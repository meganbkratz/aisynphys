from __future__ import print_function
from collections import OrderedDict
import os
from ... import config #, synphys_cache, lims
from ...util import timestamp_to_datetime
from ..pipeline_module import DatabasePipelineModule
from .opto_experiment import OptoExperimentPipelineModule
from neuroanalysis.data.experiment import Experiment
from neuroanalysis.data.libraries import opto
from neuroanalysis.baseline import float_mode
#from neuroanalysis.stimuli import find_square_pulses
from ...data import BaselineDistributor, PulseStimAnalyzer #Experiment, MultiPatchDataset, MultiPatchProbe, PulseStimAnalyzer, MultiPatchSyncRecAnalyzer, BaselineDistributor
from neuroanalysis.data import PatchClampRecording
#from optoanalysis.optoadapter import OptoRecording
from optoanalysis.data.dataset import OptoRecording, OptoSyncRecAnalyzer
import optoanalysis.power_calibration as power_cal
import optoanalysis.qc as qc
from neuroanalysis.stimuli import find_noisy_square_pulses


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

    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        job_id = job['job_id']

        # Load experiment from DB
        expt_entry = db.experiment_from_ext_id(job_id, session=session)
        elecs_by_ad_channel = {elec.device_id:elec for elec in expt_entry.electrodes}

        pairs_by_cell_id = {}
        for pair in expt_entry.pairs.values():
            pre_cell_id = pair.pre_cell.ext_id
            post_cell_id = pair.post_cell.ext_id
            pairs_by_cell_id[(pre_cell_id, post_cell_id)] = pair


        # load NWB file
        path = os.path.join(config.synphys_data, expt_entry.storage_path)
        expt = Experiment(site_path=path, loading_library=opto)
        nwb = expt.data
        stim_log = expt.library.load_stimulation_log(expt)
        if stim_log['version'] < 3:
            from acq4.util.DataManager import getHandle

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
                #if electrode_entry is not None:
                rec_entry = db.Recording(
                    sync_rec=srec_entry,
                    electrode=electrode_entry,
                    start_time=rec.start_time,
                    device_name=str(rec.device_id)
                )
                session.add(rec_entry)
                rec_entries[rec.device_id] = rec_entry
                # else:
                #     opto_rec_entry = db.OpticalRecoding(
                #         sync_rec=srec_entry,
                #         device_name=str(rec.device_id),
                #         start_time=rec.start_time
                #     )
                #     session.add(opto_rec_entry)

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

                    psa = PulseStimAnalyzer.get(rec)
                    pulses = psa.pulse_chunks()
                    pulse_entries = {}
                    all_pulse_entries[rec.device_id] = pulse_entries

                    for i,pulse in enumerate(pulses):
                        # Record information about all pulses, including test pulse.
                        t0, t1 = pulse.meta['pulse_edges']
                        resampled = pulse['primary'].resample(sample_rate=20000)
                        pulse_entry = db.StimPulse(
                            recording=rec_entry,
                            pulse_number=pulse.meta['pulse_n'],
                            onset_time=t0,
                            amplitude=pulse.meta['pulse_amplitude'],
                            duration=t1-t0,
                            data=resampled.data,
                            data_start_time=resampled.t0,
                            cell=electrode_entry.cell if electrode_entry is not None else None,
                            device_name=str(rec.device_id)
                        )
                        session.add(pulse_entry)
                        pulse_entries[pulse.meta['pulse_n']] = pulse_entry

                elif isinstance(rec, OptoRecording) and (rec.device_id=='Fidelity' or 'AD6' in rec.device_id): 
                    #psa = PulseStimAnalyzer.get(rec)
                    #psa.pulses(channel='reporter')
                    #pulses = psa.pulse_chunks
                    #pulses = find_square_pulses(rec['reporter'])
                    #print('adding OptoRecording')
                    stim_num = rec.meta['notebook']['USER_stim_num']
                    stim = stim_log[str(int(stim_num))]
                    cell_entry = expt_entry.cells[stim['stimulationPoint']['name']]

                    if stim_log['version'] >=3:
                        shape={'spiral_revolutions':stim['shape']['spiral revolutions'], 'spiral_size':stim['shape']['size']}
                    else:
                        shape={'spiral_revolutions':stim.get('prairieCmds', {}).get('spiralRevolutions')}
                        prairie_size = stim['prairieCmds']['spiralSize']
                        ref_image = os.path.join(expt.path, stim['prairieImage'][-23:])
                        if os.path.exists(ref_image):
                            h = getHandle(ref_image)
                            #x=float(size*1000000)/float(xPixels)/pixelLength
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

                    pulses = find_noisy_square_pulses(rec['reporter'])
                    for i in range(len(rec.pulse_start_times)):
                    # Record information about all pulses, including test pulse.
                        #t0, t1 = pulse.meta['pulse_edges']
                        #resampled = pulse['reporter'].resample(sample_rate=20000)
                        pulse_entry = db.StimPulse(
                            recording=rec_entry,
                            cell=cell_entry,
                            pulse_number=i, #pulse.meta['pulse_n'],
                            onset_time=rec.pulse_start_times[i], #t0,
                            amplitude=power_cal.convert_voltage_to_power(rec.pulse_power()[i], timestamp_to_datetime(expt_entry.acq_timestamp), expt_entry.rig_name), ## need to fill in laser/objective correctly
                            duration=rec.pulse_duration()[i],
                            #data=resampled.data,
                            #data_start_time=resampled.t0,
                            #wavelength,
                            #light_source,
                            position=stim['stimPos'],
                            #position_offset=stim['offset'],
                            device_name=rec.device_id,
                            #qc_pass=None
                            meta = {'shape': shape,
                                    'pockel_cmd':stim.get('prairieCmds',{}).get('laserPower', [None]*100)[i],
                                    'pockel_voltage':rec.pulse_power()[i],
                                    'position_offset':offset,
                                    'offset_distance':offset_distance,
                                    }
                            )
                        qc_pass, qc_failures = qc.opto_stim_pulse_qc_pass(pulse_entry)
                        pulse_entry.qc_pass = qc_pass
                        if not qc_pass:
                            pulse_entry.meta['qc_failures'] = qc_failures

                        session.add(pulse_entry)
                        pulse_entries[i] = pulse_entry

                elif 'TTL' in rec.device_id:
                    #print('passing device:', rec.device_id)
                    pass
                    
                else:
                    raise Exception('need to figure out recording type for %s' % rec.device_id)

            ### import postsynaptic responses
            osra = OptoSyncRecAnalyzer(srec)
            for stim_rec in srec.fidelity_channels:
                for post_rec in srec.recording_channels:
                    #if pre_dev == post_dev:
                    #    continue
                    stim_num = stim_rec.meta['notebook']['USER_stim_num']
                    stim = stim_log[str(int(stim_num))]
                    pre_cell_name = stim['stimulationPoint']['name']

                    post_cell_name = 'electrode_'+ str(post_rec.device_id)

                    pair_entry = pairs_by_cell_id[(pre_cell_name, post_cell_name)]




                    # get all responses, regardless of the presence of a spike
                    responses = osra.get_photostim_responses(stim_rec, post_rec)
                    for resp in responses:
  

                        #raise Exception('stop')
                        #### NEED a way to get pair_entry - can't use device ids because pre-device is not attached to cell

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

            # generate up to 20 baseline snippets for each recording
            for dev in srec.recording_channels:
                rec = srec[dev.device_id]
                dist = BaselineDistributor.get(rec)
                for i in range(20):
                    base = dist.get_baseline_chunk(20e-3)
                    if base is None:
                        # all out!
                        break
                    start, stop = base
                    #raise Exception('stop')
                    data = rec['primary'].time_slice(start, stop).resample(sample_rate=20000).data

                    ex_qc_pass, in_qc_pass, qc_failures = qc.opto_pulse_response_qc_pass(rec, [start, stop])

                    base_entry = db.Baseline(
                        recording=rec_entries[dev.device_id],
                        data=data,
                        data_start_time=start,
                        mode=float_mode(data),
                        ex_qc_pass=ex_qc_pass,
                        in_qc_pass=in_qc_pass,
                        meta=None if ex_qc_pass is True and in_qc_pass is True else {'qc_failures': qc_failures},
                    )
                    session.add(base_entry)


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
            ready[rec.ext_id] = max(expt_mtime, nwb_mtime)
        return ready