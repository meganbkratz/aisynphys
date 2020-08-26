# coding: utf8
from __future__ import print_function, division

import os
import numpy as np
import pyqtgraph as pg
from collections import OrderedDict
from ... import config
from ..pipeline_module import DatabasePipelineModule
from .opto_experiment import OptoExperimentPipelineModule
from .opto_dataset import OptoDatasetPipelineModule
import aisynphys.data.data_notes_db as notes_db
from ...avg_response_fit import get_pair_avg_fits


class OptoSynapsePipelineModule(DatabasePipelineModule):
    """Generate fit to response average for all pairs per experiment
    """
    name = 'synapse'
    dependencies = [ExperimentPipelineModule, DatasetPipelineModule]
    table_group = ['synapse', 'avg_response_fit']
    
    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        expt_id = job['job_id']
        
        expt = db.experiment_from_ext_id(expt_id, session=session)

        for pair in expt.pair_list:

            notes_rec = notes_db.get_pair_notes_record(pair.experiment.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id)
            if notes_rec is None:
                print('   -- No notes record for %s' % pair)
                continue
            
            # update upstream pair record
            pair.has_synapse = notes_rec.notes['synapse_type'] in ('ex', 'in')
            pair.has_electrical = notes_rec.notes['gap_junction']

            # only proceed if we have a synapse here            
            if not pair.has_synapse:
                continue

            fits = get_pair_avg_fits_opto(pair, session)

            for (clamp_mode, holding, power), data in fits.items():

                rec = db.AvgResponseFit(
                    pair_id=pair.id,
                    clamp_mode=clamp_mode,
                    holding=holding,
                    laser_power_command=power,
                    nrmse=data['fit_result'].nrmse(),
                    initial_xoffset=data['initial_latency'],
                    manual_qc_pass=data['fit_qc_pass'],
                    avg_data=fit['average'],
                    



                    )
                for k in ['xoffset', 'yoffset', 'amp', 'rise_time', 'decay_tau', 'exp_amp', 'exp_tau']:
                    setattr(rec, 'fit_'+k, data['fit_result'].best_values[k])


            

        ## AvgResponseFit
        # ('pair_id', 'pair.id', 'The ID of the entry in the pair table to which these results apply', {'index': True}),
        # ('clamp_mode', 'str', 'The clamp mode "ic" or "vc"', {'index': True}),
        # ('holding', 'float', 'The holding potential -70 or -55', {'index': True}),
        # ('laser_power_command', 'float', 'The pockel cell command value for the 2p laser'),
        # ('fit_xoffset', 'float', 'Fit time from max slope of the presynaptic spike until onset of the synaptic response (seconds)'),
        # ('fit_yoffset', 'float', 'Fit constant y-offset (amps or volts)'),
        # ('fit_amp', 'float', 'Fit synaptic response amplitude (amps or volts)'),
        # ('fit_rise_time', 'float', 'Fit rise time (seconds) from response onset until peak'),
        # ('fit_rise_power', 'float', 'Fit rise exponent (usually fixed at 2)'),
        # ('fit_decay_tau', 'float', 'Fit exponential decay time constant (seconds)'),
        # ('fit_exp_amp', 'float', 'Fit baseline exponental amplitude (amps or volts)'),
        # ('fit_exp_tau', 'float', 'Fit baseline exponental decay time constant (seconds)'),
        # ('nrmse', 'float', 'Normalized RMS error of the fit residual'),
        # ('initial_xoffset', 'float', 'Initial latency supplied to fitting algorithm'),
        # ('manual_qc_pass', 'bool', 'If true, this fit passes manual verification QC'),
        # ('avg_data', 'array', 'Averaged PSP/PSC that was fit.', {'deferred': True}),
        # ('avg_data_start_time', 'float', 'Starting time of avg_data, relative to the presynaptic spike'),
        # ('n_averaged_responses', 'int', 'Number of postsynaptic responses that were averaged in avg_data'),
        # ('avg_baseline_noise', 'float', 'Standard deviation of avg_data before the presynaptic stimulus'),


    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        db = self.database
        
        q = session.query(db.Synapse)
        q = q.filter(db.Synapse.pair_id==db.Pair.id)
        q = q.filter(db.Pair.experiment_id==db.Experiment.id)
        q = q.filter(db.Experiment.ext_id.in_(job_ids))
        recs = q.all()
        
        q = session.query(db.AvgResponseFit)
        q = q.filter(db.AvgResponseFit.pair_id==db.Pair.id)
        q = q.filter(db.Pair.experiment_id==db.Experiment.id)
        q = q.filter(db.Experiment.ext_id.in_(job_ids))
        recs.extend(q.all())

        return recs

    def ready_jobs(self):
        """Return an ordered dict of all jobs that are ready to be processed (all dependencies are present)
        and the dates that dependencies were created.
        """
        dataset_module = self.pipeline.get_module('opto_dataset')
        finished_datasets = dataset_module.finished_jobs()

        # find most recent modification time listed for each experiment
        notes_recs = notes_db.db.query(notes_db.PairNotes.expt_id, notes_db.PairNotes.modification_time)
        mod_times = {}
        for rec in notes_recs:
            mod_times[rec.expt_id] = max(rec.modification_time, mod_times.get(rec.expt_id, rec.modification_time))

        # combine update times from pair_notes and finished_datasets
        ready = OrderedDict()
        for job, (mtime, success) in finished_datasets.items():
            if job in mod_times:
                mtime = max(mtime, mod_times[job])
            ready[job] = {'dep_time': mtime}
            
        return ready