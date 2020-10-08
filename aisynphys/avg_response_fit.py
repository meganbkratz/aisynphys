# coding: utf8
import sys, itertools
from collections import OrderedDict
import numpy as np
import pyqtgraph as pg
from neuroanalysis.data import TSeries, TSeriesList
from neuroanalysis.baseline import float_mode
from neuroanalysis.fitting import Psp, StackedPsp, fit_psp
import neuroanalysis.filter as filters
from aisynphys.database import default_db as db
import aisynphys.data.data_notes_db as notes_db
from aisynphys.qc import spike_qc
from aisynphys.fitting import fit_avg_pulse_response
from sqlalchemy.orm import joinedload, undefer


def get_pair_avg_fits(pair, session, notes_session=None, ui=None, max_ind_freq=50):
    """Return PSP fits to averaged responses for this pair.

    Fits are performed against average PSPs in 4 different categories: 
    IC -70mV, IC -55mV, VC -70mV, and VC -55mV. All PSPs in these categories are averaged together
    regardless of their position in a pulse train, so we expect the amplitudes of these averages to
    be affected by any short-term depression/facilitation present at the synapse. As such, these fits
    are not ideal for measuring the amplitude of the synapse; however, they do provide good estimates
    of rise time and decay tau.
    
    Operations are:
    
    - Query all pulse responses for this pair, where the pulse train frequency was 
      no faster than 50Hz
    - Sort responses by clamp mode and holding potential, with the latter in two bins: -80 to -60 mV and -60 to -45 mV.
      Responses are further separated into qc pass/fail for each bin. QC pass criteria:
        - PR must have exactly one presynaptic spike with detectable latency
        - Either PR.ex_qc_pass or .in_qc_pass must be True, depending on clamp mode / holding
    - Generate average response for qc-passed pulses responses in each mode/holding combination
    - Fit averages to PSP curve. If the latency was manually annotated for this synapse, then the curve
      fit will have its latency constrained within ±100 μs.
    - Compare to manually verified fit parameters; if these are not a close match OR if the 
      manual fits were already failed, then *fit_qc_pass* will be False.
      
    
    Returns
    -------
    results : dict
        {(mode, holding): {
            'responses': ..,
            'average': ..,
            'initial_latency': ..,
            'fit_result': ..,
            'fit_qc_pass': ..,
            'fit_qc_pass_reasons': ..,
            'expected_fit_params': ..,
            'expected_fit_pass': ..,
            'avg_baseline_noise': ..,
        }, ...}
    
    """
    prof = pg.debug.Profiler(disabled=True, delayed=False)
    prof(str(pair))
    results = {}
    
    # query and sort pulse responses with induction frequency 50Hz or slower
    records = response_query(session=session, pair=pair, max_ind_freq=max_ind_freq).all()
    prof('query prs')
    pulse_responses = [rec[0] for rec in records]

    # sort into clamp mode / holding bins
    sorted_responses = sort_responses(pulse_responses)
    prof('sort prs')

    # load expected PSP curve fit parameters from notes DB
    notes_rec = notes_db.get_pair_notes_record(pair.experiment.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id, session=notes_session)
    prof('get pair notes')

    if ui is not None:
        ui.show_pulse_responses(sorted_responses)
        ui.show_data_notes(notes_rec)
        prof('update ui')

    for (clamp_mode, holding), responses in sorted_responses.items():
        if len(responses['qc_pass']) == 0:
            results[clamp_mode, holding] = None
            continue
            
        if notes_rec is None:
            notes = None
            sign = 0
            init_latency = None
            latency_window = (0.5e-3, 8e-3)
        else:
            notes = notes_rec.notes
            if notes.get('fit_parameters') is None:
                init_latency = None
                latency_window = (0.5e-3, 8e-3)
            else:
                init_latency = notes['fit_parameters']['initial'][clamp_mode][str(holding)]['xoffset']
                latency_window = (init_latency - 100e-6, init_latency + 100e-6)
            
            # Expected response sign depends on synapse type, clamp mode, and holding:
            sign = 0
            if notes['synapse_type'] == 'ex':
                sign = -1 if clamp_mode == 'vc' else 1
            elif notes['synapse_type'] == 'in' and holding == -55:
                sign = 1 if clamp_mode == 'vc' else -1

        prof('prepare %s %s' % (clamp_mode, holding))
        fit_result, avg_response = fit_avg_pulse_response(responses['qc_pass'], latency_window, sign)
        prof('fit avg')

        # measure baseline noise
        avg_baseline_noise = avg_response.time_slice(avg_response.t0, avg_response.t0+7e-3).data.std()

        # compare to manually-verified results
        if notes is None:
            qc_pass = False
            reasons = ['no data notes entry']
            expected_fit_params = None
            expected_fit_pass = None
        elif notes['fit_pass'][clamp_mode][str(holding)] is not True:
            qc_pass = False
            reasons = ['data notes fit failed qc']
            expected_fit_params = None
            expected_fit_pass = False
        else:
            expected_fit_params = notes['fit_parameters']['fit'][clamp_mode][str(holding)]
            expected_fit_pass = True
            qc_pass, reasons = check_fit_qc_pass(fit_result, expected_fit_params, clamp_mode)
            if not qc_pass:
                print("%s %s %s: %s" % (str(pair), clamp_mode, holding,  '; '.join(reasons)))

        if ui is not None:
            ui.show_fit_results(clamp_mode, holding, fit_result, avg_response, qc_pass)

        results[clamp_mode, holding] = {
            'responses': responses,
            'average': avg_response,
            'initial_latency': init_latency,
            'fit_result': fit_result,
            'fit_qc_pass': qc_pass,
            'fit_qc_pass_reasons': reasons,
            'expected_fit_params': expected_fit_params,
            'expected_fit_pass': expected_fit_pass,
            'avg_baseline_noise': avg_baseline_noise,
        }

    return results


def response_query(session, pair, max_ind_freq=50):
    """Query pulse responses appropriate for generating nice average PSP/PSC shapes.
    
    - Only select from multipatch probes with induction frequencies <= 50Hz
    """
    q = session.query(db.PulseResponse, db.PatchClampRecording, db.StimPulse)
    
    q = q.join(db.StimPulse, db.PulseResponse.stim_pulse)
    q = q.outerjoin(db.StimSpike, db.StimSpike.stim_pulse_id==db.StimPulse.id)
    q = q.join(db.Recording, db.PulseResponse.recording)
    q = q.join(db.PatchClampRecording)
    q = q.outerjoin(db.MultiPatchProbe)
    
    q = q.filter(db.PulseResponse.pair_id == pair.id)

    mpp_count = session.query(db.MultiPatchProbe).count()
    if mpp_count == 0 or max_ind_freq is None:
        return q
    else:
        q = q.filter(db.MultiPatchProbe.induction_frequency <= max_ind_freq)
        return q

def get_pulse_response(session, expt_id, sync_rec_id, device_name, pulse_number):
    """Return a particular pulse response from it's external id combination (or None if no response if found).

    expt_id (str)       The external id of the experiment
    sync_rec_id (int)   The external id of the sync rec (for MIES data this is the sweep number)
    device_name (str)   The name of the device recording the response.
    pulse_number (int)  The index of the pulse in the sweep (starting at 0)

    """

    q = session.query(db.PulseResponse)
    q = q.join(db.Recording, db.PulseResponse.recording)
    q = q.join(db.StimPulse, db.PulseResponse.stim_pulse)
    q = q.join(db.SyncRec, db.Recording.sync_rec)
    q = q.join(db.Experiment, db.SyncRec.experiment)

    q = q.filter(db.Experiment.ext_id == expt_id)
    q = q.filter(db.SyncRec.ext_id == sync_rec_id)
    q = q.filter(db.Recording.device_name == str(device_name))
    q = q.filter(db.StimPulse.pulse_number == pulse_number)

    return q.one_or_none()

    


def sort_responses(pulse_responses):
    """Sort a list of pulse responses by clamp mode and holding potential into 4 categories: 
    (ic -70), (ic -55), (vc -70), (vc -55). Each category contains pulse responses split into
    lists for qc-pass and qc-fail.

    QC pass for this function requires that the pulse response has True for either pr.in_qc_pass
    or pr.ex_qc_pass, depending on which category the PR has been sorted into. We _also_ require
    at this stage that the PR has exactly one presynaptic spike with a detectable onset time.
    """
    ex_limits = [-80e-3, -60e-3]
    in_limits = [-60e-3, -45e-3]
    
    sorted_responses = {
        ('ic', -70): {'qc_pass': [], 'qc_fail': []},
        ('ic', -55): {'qc_pass': [], 'qc_fail': []},
        ('vc', -70): {'qc_pass': [], 'qc_fail': []},
        ('vc', -55): {'qc_pass': [], 'qc_fail': []},
    }
    qc = {False: 'qc_fail', True: 'qc_pass'}
    
    for pr in pulse_responses: 
        post_rec = pr.recording
        clamp_mode = post_rec.patch_clamp_recording.clamp_mode
        holding = post_rec.patch_clamp_recording.baseline_potential

        if in_limits[0] <= holding < in_limits[1]:
            qc_pass = qc[pr.in_qc_pass and pr.stim_pulse.n_spikes == 1 and pr.stim_pulse.first_spike_time is not None]
            sorted_responses[clamp_mode, -55][qc_pass].append(pr)
        elif ex_limits[0] <= holding < ex_limits[1]:
            qc_pass = qc[pr.ex_qc_pass and pr.stim_pulse.n_spikes == 1 and pr.stim_pulse.first_spike_time is not None]
            sorted_responses[clamp_mode, -70][qc_pass].append(pr)
    
    return sorted_responses


def check_fit_qc_pass(fit_result, expected_params, clamp_mode):
    """Return bool indicating whether a PSP fit result matches a previously accepted result, as well as
    a list of strings describing the reasons, if any.
    """
    failures = []
    fit_params = fit_result.best_values

    # decide on relative and absolute thresholds to use for comparing each parameter
    abs_amp_threshold = 40e-6 if clamp_mode == 'ic' else 1e-12
    if fit_result.nrmse() < expected_params['nrmse']:
        # if the new fit improves NRMSE, then we can be a little more forgiving about not matching the previous fit.
        thresholds = {'amp': (0.3, abs_amp_threshold*1.5), 'rise_time': (1.0, 1e-3), 'decay_tau': (3.0, 5e-3), 'xoffset': (1.0, 150e-6)}
    else:
        thresholds = {'amp': (0.2, abs_amp_threshold), 'rise_time': (0.5, 1e-3), 'decay_tau': (2.0, 5e-3), 'xoffset': (1.0, 150e-6)}

    # compare parameters
    for k, (error_threshold, abs_threshold) in thresholds.items():
        v1 = fit_params[k]
        v2 = expected_params[k]
        if v2 == 0:
            continue
        error = abs(v1-v2) / v2
        # We expect large relative errors when the values are small relative to noise,
        # and large absolute errors otherwise.
        if (error > error_threshold) and (abs(v1 - v2) > abs_threshold):
            failures.append('%s error too large (%s != %s)' % (k, v1, v2))

    return len(failures) == 0, failures


##### 2p opto average response analysis functions
##### -------------------------------------------

def response_query_2p(session, pair, include_data=False):
    """Return a list of PulseResponses associated with the given pair. This
    function eagerloads the associated records from StimPulse, Recording, 
    PatchClampRecording, SyncRecording, Pair and Experiment. (This reduces the 
        number of downstream queries which significantly speeds things up, 
        particularly for slower connections.)

    If include_data is True, then also eagerload the post_tseries and pre_tseries data."""

    q = session.query(db.PulseResponse)
    q = q.filter(db.PulseResponse.pair_id == pair.id)
    q = q.options(joinedload(db.PulseResponse.stim_pulse))
    q = q.options(joinedload(db.PulseResponse.recording))
    q = q.options(joinedload(db.PulseResponse.recording, db.Recording.patch_clamp_recording))
    q = q.options(joinedload(db.PulseResponse.recording, db.Recording.sync_rec))
    q = q.options(joinedload(db.PulseResponse.pair))
    q = q.options(joinedload(db.PulseResponse.pair, db.Pair.experiment))

    if include_data:
        q = q.options(undefer(db.PulseResponse.data))
        q = q.options(undefer(db.PulseResponse.stim_pulse, db.StimPulse.data))

    return q.all()


def sort_responses_2p(pulse_responses, exclude_empty=True):
    """Sort a list of pulse_responses into categories based on clamp mode, 
    holding potential, and stimulation power.

    Category keys are tuples of (clamp mode, holding potential, stimulation), 
    for example ('ic', -70, 30.0). Stimulation power is reported as the command 
    to the pockel cell. Stimulation power of None indicates that an electrode 
    driving an action potential was the source of stimulation.

    In order to pass qc, responses must have response.in_qc_pass or 
    response.ex_qc_pass set to True, and also have an offset distance < 10 um. 

    Return a nested dictionary:
    {(category1):{'qc_pass':[responses that pass qc], 'qc_fail':[responses that fail qc]},
    """
    ex_limits = [-80e-3, -60e-3]
    in_limits1 = [-60e-3, -45e-3]
    in_limits2 = [-10e-3, 10e-3] ## some experiments were done with Cs+ and held at 0mv
    distance_limit = 10e-6

    modes = ['vc', 'ic']
    holdings = [-70, -55, 0]
    powers = []
    for pr in pulse_responses:
        if pr.stim_pulse.meta is None:
            powers.append(None)
        else:
            powers.append(pr.stim_pulse.meta.get('pockel_cmd'))
    powers = list(set(powers))
    #powers = list(set([pr.stim_pulse.meta.get('pockel_cmd') for pr in pulse_responses if pr.stim_pulse.meta is not None else None]))

    keys = itertools.product(modes, holdings, powers)

    ### Need to differentiate between laser-stimulated pairs and electrode-electode pairs
    ## I would like to do this in a more specific way, ie: if the device type == Fidelity. -- this is in the pipeline branch of aisynphys 
    ## But that needs to wait until devices are in the db. (but also aren't they?)
    ## Also, we're going to have situations where the same pair has laser responses and 
    ##   electrode responses when we start getting 2P guided pair patching, and this will fail then
    # if pulse_responses[0].pair.pre_cell.electrode is None:  
    #     powers = list(set([pr.stim_pulse.meta.get('pockel_cmd') for pr in pulse_responses]))
    #     keys = itertools.product(modes, holdings, powers)
    # else:
    #     keys = itertools.product(modes, holdings)

    sorted_responses = OrderedDict({k:{'qc_pass':[], 'qc_fail':[]} for k in keys})

    qc = {False: 'qc_fail', True: 'qc_pass'}

    for pr in pulse_responses:
        clamp_mode = pr.recording.patch_clamp_recording.clamp_mode
        holding = pr.recording.patch_clamp_recording.baseline_potential
        power = pr.stim_pulse.meta.get('pockel_cmd') if pr.stim_pulse.meta is not None else None

        offset_distance = pr.stim_pulse.meta.get('offset_distance', 0) if pr.stim_pulse.meta is not None else 0
        if offset_distance is None: ## early photostimlogs didn't record the offset between the stimulation plane and the cell
            offset_distance = 0

        if in_limits1[0] <= holding < in_limits1[1]:
            qc_pass = qc[pr.in_qc_pass and offset_distance < distance_limit]
            sorted_responses[(clamp_mode, -55, power)][qc_pass].append(pr)

        elif in_limits2[0] <= holding < in_limits2[1]:
            qc_pass = qc[pr.in_qc_pass and offset_distance < distance_limit]
            sorted_responses[(clamp_mode, 0, power)][qc_pass].append(pr)

        elif ex_limits[0] <= holding < ex_limits[1]:
            qc_pass = qc[pr.ex_qc_pass and offset_distance < distance_limit]
            sorted_responses[(clamp_mode, -70, power)][qc_pass].append(pr)

    if not exclude_empty:
        return sorted_responses
    else:    
        # filter out categories with no responses
        filtered_responses = OrderedDict()
        for k, v in sorted_responses.items():
            if len(v['qc_fail']) + len(v['qc_pass']) > 0:
                filtered_responses[k]=v
        return filtered_responses

def sort_responses_into_categories_2p(pulse_responses, exclude_empty=True):
    """Sort a list of pulse_responses into categories based on clamp mode, 
    holding potential, and stimulation power.

    Category keys are tuples of (clamp mode, holding potential, stimulation), 
    for example ('ic', -70, 30.0). Stimulation power is reported as the command 
    to the pockel cell. Stimulation power of None indicates that an electrode 
    driving an action potential was the source of stimulation. 

    Clamp mode options are 'vc' or 'ic'
    Holding potential options are -70mV (limits -80 to -60 mV), -55mV (limits
    -60 to -45 mV) or 0mV (limits -10 to 10 mV). 
    Stimulation power options are determined by the dataset.

    If exclude_empty is True, then we remove categories that don't have any 
    responses from the returned dictionary.

    Return a dictionary:
    {(category1):[list, of, responses], 
     (category2):[list, of, responses]}
    """
    ex_limits = [-80e-3, -60e-3]
    in_limits1 = [-60e-3, -45e-3]
    in_limits2 = [-10e-3, 10e-3] ## some experiments were done with Cs+ and held at 0mv

    modes = ['vc', 'ic']
    holdings = [-70, -55, 0]
    powers = []
    for pr in pulse_responses:
        if pr.stim_pulse.meta is None:
            powers.append(None)
        else:
            powers.append(pr.stim_pulse.meta.get('pockel_cmd'))
    powers = list(set(powers))

    keys = itertools.product(modes, holdings, powers)



    sorted_responses = OrderedDict([(k,[]) for k in keys])

    for pr in pulse_responses:
        clamp_mode = pr.recording.patch_clamp_recording.clamp_mode
        holding = pr.recording.patch_clamp_recording.baseline_potential
        power = pr.stim_pulse.meta.get('pockel_cmd') if pr.stim_pulse.meta is not None else None


        if in_limits1[0] <= holding < in_limits1[1]:
            sorted_responses[(clamp_mode, -55, power)].append(pr)

        elif in_limits2[0] <= holding < in_limits2[1]:
            sorted_responses[(clamp_mode, 0, power)].append(pr)

        elif ex_limits[0] <= holding < ex_limits[1]:
            sorted_responses[(clamp_mode, -70, power)].append(pr)

    if not exclude_empty:
        return sorted_responses
    else:    
        # filter out categories with no responses
        filtered_responses = OrderedDict()
        for k, v in sorted_responses.items():
            if len(v) > 0:
                filtered_responses[k]=v
        return filtered_responses


def fit_event_2p(avg_response, clamp_mode, latencies, event_index=0):
    """Event fitting algorithm for two-photon repsonses.

    Arguments:
    ----------
    avg_response : TSeries
        The trace of the data to be fit.
    clamp_mode : str
        The clamp mode of the data. Options: 'ic', 'vc'
    latencies : list
        A sorted list of time values corresponding to the start of events in avg_response
    event_index : int | 0
        The index of the latency (in latencies) of the event to be fit. 

    Returns:
    --------
    fit : lmfit ModelResult
        The resulting PSP fit
    """
    latency = latencies[event_index]
    if latency == latencies[-1]:
        latencies.append(latency+0.05) ## create a next latency stop point
    window = [latency - 0.0002, latency+0.0002]

    data = avg_response.time_slice(latency-0.001, latencies[event_index+1])
    filtered = filters.bessel_filter(data, 6000, order=4, btype='low', bidir=True)

    ### for events right next to spike crosstalk
    #lat_index = filtered.index_at(latency)
    #if max(abs(filtered.data[:lat_index])) > max(abs(filtered.data[lat_index:])): ## there is lots going on in the baseline
    #    ## cut it off
    #    filtered = filtered[lat_index-int(0.0002/filtered.dt):]

    peak_ind = np.argwhere(max(abs(filtered.data))==abs(filtered.data))[0][0]
    peak_time = filtered.time_at(peak_ind)
    rise_time = peak_time-(latency)
    amp = filtered.value_at(peak_time) - filtered.value_at(latency)
    init_params = {'rise_time':rise_time, 'amp':amp}

    fit = fit_psp(filtered, (window[0], window[1]), clamp_mode, sign=0, exp_baseline=False, init_params=init_params, fine_search_spacing=filtered.dt, fine_search_window_width=100e-6)
    fit.opto_init_params = init_params ## attach parameters that we feed in here; fit already has an init_params that gets filled in inside fit_psp

    return fit

def get_average_response_2p(pulse_responses):
    """Given a list of pulse responses, determine whether to align them by pulse or by spike, and return the mean."""

    prl = PulseResponseList(pulse_responses)

    has_pre_data = pulse_responses[0].pre_tseries is not None

    if has_pre_data:
        return prl.post_tseries(align='spike', bsub=True).mean()
    else:
        return prl.post_tseries(align='pulse', bsub=True).mean()

def get_pair_avg_fits_opto(pair, db_session):
    '''Operations are:
    
    - Query all pulse responses for this pair, where the pulse train frequency was 
      no faster than 50Hz
    - Sort responses by clamp mode and holding potential, with the latter in two bins: -80 to -60 mV and -60 to -45 mV.
      Responses are further separated into qc pass/fail for each bin. QC pass criteria:
        - PR must have exactly one presynaptic spike with detectable latency
        - Either PR.ex_qc_pass or .in_qc_pass must be True, depending on clamp mode / holding
    - Generate average response for qc-passed pulses responses in each mode/holding combination
    - Fit averages to PSP curve. If the latency was manually annotated for this synapse, then the curve
      fit will have its latency constrained within ±100 μs.
    - Compare to manually verified fit parameters; if these are not a close match OR if the 
      manual fits were already failed, then *fit_qc_pass* will be False.

    Returns
    -------
    results : dict
        {(mode, holding, power): {
            'responses': list of PulseResponse objects that pass qc
            'average': TSeries, the average of responses,
            'initial_latency': the latency of the event to be fit,
            'all_latencies': list of user-set latency for each event
            'fit_result': fit object,
            'fit_qc_pass': Boolean, whether the fit passes,
            'fit_qc_pass_reasons': list of strings explaining why fit_qc_pass failed (or passed?),
            'expected_fit_params': dict of expected fit parameters,
            'expected_fit_pass': whether the expected fit passed qc,
            'avg_baseline_noise': ..,
        }, ...}
    '''

### notes_rec.notes:"{
#       "expt_id": "2019_05_29_exp2_TH", 
#       "pre_cell_id": "Point 32", 
#       "post_cell_id": "electrode_0", 
#       "synapse_type": "ex", 
#       "gap_junction": false, 
#       "comments": "", 
#       "categories": {
#           "('ic', -70, 40.0)": {
#               "initial_parameters": {"rise_time": 0.0014534752985781146, "amp": 0.00036418358100861396}, 
#               "fit_parameters": {"xoffset": 0.008303698442252731, "yoffset": 2.1486003695600497e-05, "rise_time": 0.0018497438161078129, "decay_tau": 0.007747602882916505, "amp": 0.00037444707755288466, "rise_power": 2, "exp_amp": 0, "exp_tau": 1, "nrmse": 0.040841399293527765},
#               "fit_pass": "True", 
#               "n_events": "3", 
#               "event_times": [0.008346524701421892, 0.011652299482281091, 0.016906683933368895]}, 
#               "fit_event_index":0
#           "('ic', -70, 50.0)": {"initial_parameters": {"rise_time": 0.0021330294285896983, "amp": 0.00034952583580172595}, "fit_parameters": {"xoffset": 0.0073498977724152925, "yoffset": -1.9205783347603367e-05, "rise_time": 0.002132751712697264, "decay_tau": 0.009671682481174188, "amp": 0.00034229503798775146, "rise_power": 2, "exp_amp": 0, "exp_tau": 1, "nrmse": 0.030614627629844873}, "fit_pass": "True", "n_events": "5", "event_times": [0.007266970571410307, 0.010716722307365367, 0.015020227464195443, 0.0245304433743027, 0.028549997232482797]}, "('ic', -70, 60.0)": null}}"
    notes_rec = notes_db.get_pair_notes_record(pair.experiment.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id)
    if notes_rec is None:
        raise Exception('No notes record saved in %s for %s. Fitting without annotated latencies is not implemented.' %(notes_db.name(), pair))
    q = response_query(db_session, pair)
    pulse_responses = [r.PulseResponse for r in q.all()]
    sorted_responses = sort_responses_2p(pulse_responses, exclude_empty=True)

    results = OrderedDict()

    for key, responses in sorted_responses.items():
        if len(responses['qc_pass']) == 0:
            results[key] = None
            continue
        results[key] == OrderedDict()
        results['responses'] = responses['qc_pass']
        avg_response = get_average_response_2p(responses['qc_pass'])
        results['average'] = average

        notes = notes_rec.notes[str(key)]
        results['initial_latency'] = notes['event_times'][notes['fit_event_index']]
        results['all_latencies'] = notes['event_times']
        clamp_mode=key[0]
        fit = fit_event_2p(avg_response, clamp_mode, notes['event_times'], event_index=notes['fit_event_index'])
        results['fit_result'] = fit

        # compare to manually-verified results
        # if notes is None: ## Removed b/c we error out early if there's no notes record because we need latencies to calculate a fit.
        #     qc_pass = False
        #     reasons = ['no data notes entry']
        #     expected_fit_params = None
        #     expected_fit_pass = None
        if notes['fit_pass'] is not True:
            qc_pass = False
            reasons = ['data notes fit failed qc']
            expected_fit_params = None
            expected_fit_pass = False
        else:
            expected_fit_params = notes['fit_parameters']
            expected_fit_pass = True
            qc_pass, reasons = check_fit_qc_pass(fit, expected_fit_params, clamp_mode)
            if not qc_pass:
                print("%s %s: %s" % (str(pair), str(key), '; '.join(reasons)))

        results['fit_qc_pass'] = qc_pass
        results['fit_qc_pass_reasons'] = reasons
        results['expected_fit_params'] = expected_fit_params
        results['expected_fit_pass'] = expected_fit_pass

        # measure baseline noise from beginning of average to 1 ms before start of fit
        avg_baseline_noise = avg_response.time_slice(avg_response.t0, notes['event_times'][notes['fit_event_index']-1e-3]).data.std()
        results['avg_baseline_noise'] = avg_baseline_noise

    return results










