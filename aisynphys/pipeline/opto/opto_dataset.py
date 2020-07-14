from __future__ import print_function
from collections import OrderedDict
import os, struct, hashlib
import numpy as np
from ... import config #, synphys_cache, lims
from ...util import timestamp_to_datetime, datetime_to_timestamp
from ..pipeline_module import DatabasePipelineModule
from .opto_experiment import OptoExperimentPipelineModule
from neuroanalysis.data.experiment import AI_Experiment
#from neuroanalysis.data.libraries import opto
from neuroanalysis.data.loaders.opto_experiment_loader import OptoExperimentLoader
from neuroanalysis.baseline import float_mode
#from ...data import BaselineDistributor, PulseStimAnalyzer #Experiment, MultiPatchDataset, MultiPatchProbe, PulseStimAnalyzer, MultiPatchSyncRecAnalyzer, BaselineDistributor
from neuroanalysis.data import PatchClampRecording
#from optoanalysis.optoadapter import OptoRecording
#from optoanalysis.data.dataset import OptoRecording
from optoanalysis.analyzers import OptoSyncRecAnalyzer
from neuroanalysis.analyzers.stim_pulse import GenericStimPulseAnalyzer, PWMStimPulseAnalyzer, PatchClampStimPulseAnalyzer
from neuroanalysis.analyzers.baseline import BaselineDistributor
import optoanalysis.power_calibration as power_cal
import optoanalysis.qc as qc
#import cProfile


class OptoDatasetPipelineModule(DatabasePipelineModule):

    name='opto_dataset'
    dependencies = [OptoExperimentPipelineModule]
    table_group = [
        'sync_rec', 
        'recording', 
        'patch_clamp_recording', 
        'test_pulse', 
        'stim_pulse', 
        'pulse_response',
        'baseline'
        ]

    # datasets are large and NWB access leaks memory -- this is probably true for optoanalysis too
    # when running parallel, each child process may run only one job before being killed
    maxtasksperchild = 1 

    # @classmethod
    # def create_db_entries(cls, job, session):
    #     cProfile.runctx('cls.create_db_entries_real(job, session)', globals(), {'cls':cls, 'job':job, 'session':session}, filename='profile.txt')

    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        job_id = job['job_id']

        # Load experiment from DB
        expt_entry = db.experiment_from_ext_id(job_id, session=session)
        elecs_by_ad_channel = {elec.device_id:elec for elec in expt_entry.electrodes}
        cell_entries = expt_entry.cells ## do this once here instead of multiple times later because it's slooooowwwww
        pairs_by_cell_id = expt_entry.pairs

        # load NWB file
        path = os.path.join(config.synphys_data, expt_entry.storage_path)
        expt = AI_Experiment(loader=OptoExperimentLoader(site_path=path))
        nwb = expt.data
        stim_log = expt.loader.load_stimulation_log()
        stim_log_version = stim_log.get('version', 0)
        if stim_log_version < 3:
            ## gonna need to load an image in order to calculate spiral size later
            from acq4.util.DataManager import getHandle

        last_stim_pulse_time = {}
        # Load all data from NWB into DB
        for srec in nwb.contents:
            temp = srec.meta.get('temperature', None)
            srec_entry = db.SyncRec(ext_id=srec.key, experiment=expt_entry, temperature=temp)
            session.add(srec_entry)

            rec_entries = {}
            all_pulse_entries = {}
            for rec in srec.recordings:
                
                # import all recordings
                electrode_entry = elecs_by_ad_channel.get(rec.device_id, None)

                rec_entry = db.Recording(
                    sync_rec=srec_entry,
                    electrode=electrode_entry,
                    start_time=rec.start_time,
                    device_name=str(rec.device_id)
                )
                session.add(rec_entry)
                rec_entries[rec.device_id] = rec_entry

                # import patch clamp recording information
                if isinstance(rec, PatchClampRecording):
                    qc_pass, qc_failures = qc.recording_qc_pass(rec)
                    pcrec_entry = db.PatchClampRecording(
                        recording=rec_entry,
                        clamp_mode=rec.clamp_mode,
                        patch_mode=rec.patch_mode,
                        stim_name=rec.stimulus.description,
                        baseline_potential=rec.baseline_potential,
                        baseline_current=rec.baseline_current,
                        baseline_rms_noise=rec.baseline_rms_noise,
                        qc_pass=qc_pass,
                        meta=None if len(qc_failures) == 0 else {'qc_failures': qc_failures},
                    )
                    session.add(pcrec_entry)

                    # import test pulse information
                    tp = rec.nearest_test_pulse
                    if tp is not None:
                        indices = tp.indices or [None, None]
                        tp_entry = db.TestPulse(
                            electrode=electrode_entry,
                            recording=rec_entry,
                            start_index=indices[0],
                            stop_index=indices[1],
                            baseline_current=tp.baseline_current,
                            baseline_potential=tp.baseline_potential,
                            access_resistance=tp.access_resistance,
                            input_resistance=tp.input_resistance,
                            capacitance=tp.capacitance,
                            time_constant=tp.time_constant,
                        )
                        session.add(tp_entry)
                        pcrec_entry.nearest_test_pulse = tp_entry

                    psa = PatchClampStimPulseAnalyzer.get(rec)
                    pulses = psa.pulse_chunks()
                    pulse_entries = {}
                    all_pulse_entries[rec.device_id] = pulse_entries
                    cell_entry = electrode_entry.cell

                    for i,pulse in enumerate(pulses):
                        # Record information about all pulses, including test pulse.
                        t0, t1 = pulse.meta['pulse_edges']
                        resampled = pulse['primary'].resample(sample_rate=20000)
                        
                        clock_time = t0 + datetime_to_timestamp(rec_entry.start_time)
                        prev_pulse_dt = clock_time - last_stim_pulse_time.get(cell_entry.ext_id, -np.inf)
                        last_stim_pulse_time[cell_entry.ext_id] = clock_time

                        pulse_entry = db.StimPulse(
                            recording=rec_entry,
                            pulse_number=pulse.meta['pulse_n'],
                            onset_time=t0,
                            amplitude=pulse.meta['pulse_amplitude'],
                            duration=t1-t0,
                            data=resampled.data,
                            data_start_time=resampled.t0,
                            #cell=electrode_entry.cell if electrode_entry is not None else None,
                            cell=cell_entry,
                            #device_name=str(rec.device_id),
                            previous_pulse_dt=prev_pulse_dt
                        )
                        session.add(pulse_entry)
                        pulse_entries[pulse.meta['pulse_n']] = pulse_entry


                #elif isinstance(rec, OptoRecording) and (rec.device_name=='Fidelity'): 
                elif rec.device_type == 'Fidelity':
                    ## This is a 2p stimulation

                    ## get cell entry
                    stim_num = rec.meta['notebook']['USER_stim_num']
                    if stim_num is None: ### this is a trace that would have been labeled as 'unknown'
                        continue
                    stim = stim_log[str(int(stim_num))]
                    if stim_log_version == 0:
                        cell_name = 'Point %i' % stim['stimulationPoint'] 
                    else:
                        cell_name = stim['stimulationPoint']['name']
                    cell_entry = cell_entries.get(cell_name)

                    ## get stimulation shape parameters
                    if stim_log_version >=3:
                        shape={'spiral_revolutions':stim['shape']['spiral revolutions'], 'spiral_size':stim['shape']['size']}
                    else:
                        ## need to calculate spiral size from reference image, cause stimlog is from before we were saving spiral size
                        shape={'spiral_revolutions':stim.get('prairieCmds', {}).get('spiralRevolutions')}
                        prairie_size = stim['prairieCmds']['spiralSize']
                        ref_image = os.path.join(expt.path, stim['prairieImage'][-23:])
                        if os.path.exists(ref_image):
                            h = getHandle(ref_image)
                            xPixels = h.info()['PrairieMetaInfo']['Environment']['PixelsPerLine']
                            pixelLength = h.info()['PrairieMetaInfo']['Environment']['XAxis_umPerPixel']
                            size = prairie_size * pixelLength * xPixels * 1e-6
                            shape['spiral_size'] = size
                        else:
                            shape['spiral_size'] = None

                    ## calculate offset_distance
                    offset = stim.get('offset')
                    if offset is not None:
                        offset_distance = (offset[0]**2 + offset[1]**2 + offset[2]**2)**0.5
                    else:
                        offset_distance = None

                    pulse_entries = {}
                    all_pulse_entries[rec.device_id] = pulse_entries

                    ospa = GenericStimPulseAnalyzer.get(rec)

                    for i, pulse in enumerate(ospa.pulses(channel='reporter')):
                        ### pulse is (start, stop, amplitude)
                    # Record information about all pulses, including test pulse.
                        #t0, t1 = pulse.meta['pulse_edges']
                        #resampled = pulse['reporter'].resample(sample_rate=20000)

                        t0, t1 = pulse[0], pulse[1]
                        if cell_entry is not None:
                            clock_time = t0 + datetime_to_timestamp(rec_entry.start_time)
                            prev_pulse_dt = clock_time - last_stim_pulse_time.get(cell_entry.ext_id, -np.inf)
                            last_stim_pulse_time[cell_entry.ext_id] = clock_time
                        pulse_entry = db.StimPulse(
                            recording=rec_entry,
                            cell=cell_entry,
                            pulse_number=i, #pulse.meta['pulse_n'],
                            onset_time=pulse[0],#rec.pulse_start_times[i], #t0,
                            amplitude=power_cal.convert_voltage_to_power(pulse[2], timestamp_to_datetime(expt_entry.acq_timestamp), expt_entry.rig_name), ## need to fill in laser/objective correctly
                            duration=pulse[1]-pulse[0],#rec.pulse_duration()[i],
                            previous_pulse_dt=prev_pulse_dt,
                            #data=resampled.data,
                            #data_start_time=resampled.t0,
                            #wavelength,
                            #light_source,
                            position=stim.get('stimPos', stim['pos']), ### older stim log versions don't have stimPos
                            #position_offset=stim['offset'],
                            #device_name=rec.device_id,
                            #qc_pass=None
                            meta = {'shape': shape,
                                    'pockel_cmd':stim.get('prairieCmds',{}).get('laserPower', [None]*100)[i],
                                    'pockel_voltage': float(pulse[2]),#rec.pulse_power()[i],
                                    'position_offset':offset,
                                    'offset_distance':offset_distance,
                                    'wavelength': 1070e-9
                                    } # TODO: put in light_source and wavelength
                            )
                        qc_pass, qc_failures = qc.opto_stim_pulse_qc_pass(pulse_entry)
                        pulse_entry.qc_pass = qc_pass
                        if not qc_pass:
                            pulse_entry.meta['qc_failures'] = qc_failures
                        if cell_entry is None:
                            pulse_entry.meta.get('warnings', []).append('No cell named "%s" was found in this experiment.' % cell_name)

                        session.add(pulse_entry)
                        pulse_entries[i] = pulse_entry


                elif 'LED' in rec.device_type:
                    #if rec.device_id == 'TTL1P_0': ## this is the ttl output to Prairie, not an LED stimulation
                    #    continue

                    ### This is an LED stimulation
                    #if rec.device_id in ['TTL1_1', 'TTL1P_1']:
                    #    lightsource = 'LED-470nm'
                    #elif rec.device_id in ['TTL1_2', 'TTL1P_2']:
                    #    lightsource = 'LED-590nm'
                    #else:
                    #    raise Exception("Don't know lightsource for device: %s" % rec.device_id)

                    pulse_entries = {}
                    all_pulse_entries[rec.device_id] = pulse_entries

                    spa = PWMStimPulseAnalyzer.get(rec)
                    pulses = spa.pulses(channel='reporter')
                    max_power=power_cal.get_led_power(timestamp_to_datetime(expt_entry.acq_timestamp), expt_entry.rig_name, rec.device_id)

                    for i, pulse in enumerate(pulses):
                        pulse_entry = db.StimPulse(
                            recording=rec_entry,
                            #cell=cell_entry, ## we're not stimulating just one cell here TODO: but maybe this should be a list of cells in the fov?
                            pulse_number=i,
                            onset_time=pulse.global_start_time,
                            amplitude=max_power*pulse.amplitude,
                            duration=pulse.duration,
                            #data=resampled.data, ## don't need data, it's just a square pulse
                            #data_start_time=resampled.t0,
                            #position=None, # don't have a 3D position, have a field
                            #device_name=rec.device_id,
                            meta = {'shape': 'wide-field', ## TODO: description of field of view
                                    'LED_voltage':str(pulse.amplitude),
                                    'light_source':rec.device_id,
                                    'pulse_width_modulation': spa.pwm_params(channel='reporter', pulse_n=i),
                                    #'position_offset':offset,
                                    #'offset_distance':offset_distance,
                                    } ## TODO: put in lightsource and wavelength
                            )
                        ## TODO: make qc function for LED stimuli
                        #qc_pass, qc_failures = qc.opto_stim_pulse_qc_pass(pulse_entry)
                        #pulse_entry.qc_pass = qc_pass
                        #if not qc_pass:
                        #    pulse_entry.meta['qc_failures'] = qc_failures

                        session.add(pulse_entry)
                        pulse_entries[i] = pulse_entry
                    
                elif rec.device_id == 'unknown': 
                    ## At the end of some .nwbs there are vc traces to check access resistance.
                    ## These have an AD6(fidelity) channel, but do not have an optical stimulation and
                    ## this channel is labeled unknown when it gets created in OptoRecording
                    pass

                elif rec.device_id== 'Prairie_Command':
                    ### this is just the TTL command sent to the laser, the actually data about when the Laser was active is in the Fidelity channel
                    pass
                    
                else:
                    raise Exception('Need to figure out recording type for %s (device_id:%s)' % (rec, rec.device_id))

            # collect and shuffle baseline chunks for each recording
            baseline_chunks = {}
            for post_rec in [rec for rec in srec.recordings if isinstance(rec, PatchClampRecording)]:
                post_dev = post_rec.device_id

                base_dist = BaselineDistributor.get(post_rec)
                chunks = list(base_dist.baseline_chunks())
                
                # generate a different random shuffle for each combination pre,post device
                # (we are not allowed to reuse the same baseline chunks for a particular pre-post pair,
                # but it is ok to reuse them across pairs)
                for pre_dev in srec.devices: 
                    # shuffle baseline chunks in a deterministic way:
                    # convert expt_id/srec_id/pre/post into an integer seed
                    seed_str = ("%s %s %s %s" % (job_id, srec.key, pre_dev, post_dev)).encode()
                    seed = struct.unpack('I', hashlib.sha1(seed_str).digest()[:4])[0]
                    rng = np.random.RandomState(seed)
                    rng.shuffle(chunks)
                    
                    baseline_chunks[pre_dev, post_dev] = chunks[:]

            baseline_qc_cache = {}
            baseline_entry_cache = {}

            ### import postsynaptic responses
            unmatched = 0
            osra = OptoSyncRecAnalyzer.get(srec)
            for stim_rec in srec.recordings:
                if stim_rec.device_type in ['Prairie_Command', 'unknown']: ### these don't actually contain data we want to use -- ignore them
                    continue
                if isinstance(stim_rec, PatchClampRecording):
                    ### exclude trying to analyze intrinsic pulses
                    stim_name = stim_rec.stimulus.description
                    if any(substr in stim_name for substr in ['intrins']):
                        continue

                for post_rec in [x for x in srec.recordings if isinstance(x, PatchClampRecording)]:
                    if stim_rec == post_rec:
                        continue

                    if 'Fidelity' in stim_rec.device_type:
                        stim_num = stim_rec.meta['notebook']['USER_stim_num']
                        if stim_num is None: ## happens when last sweep records a voltage offset - used to be labelled as 'unknown' device
                            continue
                        
                        stim = stim_log[str(int(stim_num))]
                        if stim_log_version == 0:
                            pre_cell_name = 'Point %i' % stim['stimulationPoint'] 
                        else:
                            pre_cell_name = stim['stimulationPoint']['name']

                        post_cell_name = str('electrode_'+ str(post_rec.device_id))

                        pair_entry = pairs_by_cell_id.get((pre_cell_name, post_cell_name))

                    elif 'led' in stim_rec.device_type.lower():
                        pair_entry = None

                    elif isinstance(stim_rec, PatchClampRecording):
                        pre_cell_name = str('electrode_' + str(stim_rec.device_id))
                        post_cell_name = str('electrode_'+ str(post_rec.device_id))
                        pair_entry = pairs_by_cell_id.get((pre_cell_name, post_cell_name))

                    # get all responses, regardless of the presence of a spike
                    responses = osra.get_responses(stim_rec, post_rec)
                    if len(responses) > 10:
                        raise Exception('Found more than 10 pulse responses for %s. Please investigate.'%srec)
                    for resp in responses:
                        if pair_entry is not None: ### when recordings are crappy cells are not always included in connections files so won't exist as pairs in the db, also led stimulations don't have pairs
                            if resp['ex_qc_pass']:
                                pair_entry.n_ex_test_spikes += 1
                            if resp['in_qc_pass']:
                                pair_entry.n_in_test_spikes += 1
                            
                        resampled = resp['response']['primary'].resample(sample_rate=20000)
                        resp_entry = db.PulseResponse(
                            recording=rec_entries[post_rec.device_id],
                            stim_pulse=all_pulse_entries[stim_rec.device_id][resp['pulse_n']],
                            pair=pair_entry,
                            data=resampled.data,
                            data_start_time=resampled.t0,
                            ex_qc_pass=resp['ex_qc_pass'],
                            in_qc_pass=resp['in_qc_pass'],
                            meta=None if resp['ex_qc_pass'] and resp['in_qc_pass'] else {'qc_failures': resp['qc_failures']},
                        )
                        session.add(resp_entry)

                        # find a baseline chunk from this recording with compatible qc metrics
                        got_baseline = False
                        for i, (start, stop) in enumerate(baseline_chunks[stim_rec.device_id, post_rec.device_id]):
                            key = (post_rec.device_id, start, stop)

                            # pull data and run qc if needed
                            if key not in baseline_qc_cache:
                                data = post_rec['primary'].time_slice(start, stop).resample(sample_rate=db.default_sample_rate).data
                                ex_qc_pass, in_qc_pass, qc_failures = qc.opto_pulse_response_qc_pass(post_rec, [start, stop])
                                baseline_qc_cache[key] = (data, ex_qc_pass, in_qc_pass)
                            else:
                                (data, ex_qc_pass, in_qc_pass) = baseline_qc_cache[key]

                            if resp_entry.ex_qc_pass is True and ex_qc_pass is not True:
                                continue
                            elif resp_entry.in_qc_pass is True and in_qc_pass is not True:
                                continue
                            else:
                                got_baseline = True
                                baseline_chunks[stim_rec.device_id, post_rec.device_id].pop(i)
                                break

                        if not got_baseline:
                            # no matching baseline available
                            unmatched += 1
                            continue

                        if key not in baseline_entry_cache:
                            # create a db record for this baseline chunk if it has not already appeared elsewhere
                            base_entry = db.Baseline(
                                recording=rec_entries[post_rec.device_id],
                                data=data,
                                data_start_time=start,
                                mode=float_mode(data),
                                ex_qc_pass=ex_qc_pass,
                                in_qc_pass=in_qc_pass,
                                meta=None if ex_qc_pass is True and in_qc_pass is True else {'qc_failures': qc_failures},
                            )
                            session.add(base_entry)
                            baseline_entry_cache[key] = base_entry
                        
                        resp_entry.baseline = baseline_entry_cache[key]

            if unmatched > 0:
                print("%s %s: %d pulse responses without matched baselines" % (job_id, srec, unmatched))

    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        # only need to return from syncrec table; other tables will be dropped automatically.
        db = self.database
        return session.query(db.SyncRec).filter(db.SyncRec.experiment_id==db.Experiment.id).filter(db.Experiment.ext_id.in_(job_ids)).all()

    def ready_jobs(self):
        """Return an ordered dict of all jobs that are ready to be processed (all dependencies are present)
        and the dates that dependencies were created.
        """
        db = self.database
        
        # All experiments and their creation times in the DB
        expt_module = self.pipeline.get_module('opto_experiment')
        expts = expt_module.finished_jobs()
        
        # Look up nwb file locations for all experiments
        session = db.session()
        expt_recs = session.query(db.Experiment.ext_id, db.Experiment.storage_path, db.Experiment.ephys_file).filter(db.Experiment.ephys_file != None).all()
        expt_paths = {rec.ext_id: rec for rec in expt_recs}
        session.rollback()
        
        # Return the greater of NWB mod time and experiment DB record mtime
        ready = OrderedDict()
        for expt_id, (expt_mtime, expt_success) in expts.items():
            if expt_id not in expt_paths or expt_success is False:
                # no NWB file; ignore
                continue
            rec = expt_paths[expt_id]
            ephys_file = os.path.join(config.synphys_data, rec.storage_path, rec.ephys_file)
            nwb_mtime = timestamp_to_datetime(os.stat(ephys_file).st_mtime)
            ready[rec.ext_id] = {'dep_time':max(expt_mtime, nwb_mtime), 'meta':{'source':ephys_file}}
        return ready