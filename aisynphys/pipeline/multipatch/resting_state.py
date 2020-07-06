# coding: utf8
from __future__ import print_function, division

import numpy as np
from .pipeline_module import MultipatchPipelineModule
from .synapse import SynapsePipelineModule
from ...resting_state import resting_state_response_fits


# Minimum duration (seconds) we need to wait between stimuli to consider
# a synapse at "rest". Although some synaptic dynamics may last much longer
# than this limit, we need to make a tradeoff to have enough data to average.
# (the question here is really: how short can we make this interval without
# creating a significant bias in the average?)
minimum_rest_duration = 8.0


class RestingStatePipelineModule(MultipatchPipelineModule):
    """Measure the "resting state" amplitude of a synapse by averaging together only responses 
    that have no prior stimuli in a certain window. 

    This module generates records in the resting_state_fit table and also updates synapse.psp_amplitude
    and synapse.psc_amplitude. For QC, the synapse fields are only updated if synapse.psp_rise_time and 
    .psc_rise_time have already been set.

    """
    name = 'resting_state'
    dependencies = [SynapsePipelineModule]
    table_group = ['resting_state_fit']
    
    @classmethod
    def create_db_entries(cls, job, session):
        db = job['database']
        expt_id = job['job_id']

        expt = db.experiment_from_ext_id(expt_id, session=session)
       
        for pair in expt.pairs.values():
            if pair.has_synapse is not True:
                continue

            # get resting-state response fits for this pair            
            result = resting_state_response_fits(pair, rest_duration=minimum_rest_duration)
            if result is None:
                continue
            
            # write out a db record 
            fit_rec = db.RestingStateFit(pair=pair)
            for mode in ('ic', 'vc'):
                fit = result[mode]['fit']
                if fit is None:
                    continue
                for k,v in fit.best_values.items():
                    setattr(fit_rec, mode + '_' + k, v)
                setattr(fit_rec, mode + '_nrmse', fit.nrmse())
                setattr(fit_rec, mode + '_avg_data', result[mode]['average'].data)
                setattr(fit_rec, mode + '_avg_data_start_time', result[mode]['average'].t0)
                
                # list IDs of pulses that went into this average
                pr_ids = np.array([pr.id for pr in result[mode]['responses']]),
                setattr(fit_rec, mode + '_pulse_ids', pr_ids)
                
            session.add(fit_rec)
            
            # update synapse record IF it also has kinetics
            # (if not, then the synapse failed psp fitting to the all-pulse averages,
            # so we should assume the fit to only resting-state pulses is even worse)
            if result['ic']['fit'] is not None and pair.synapse.psp_rise_time is not None:
                pair.synapse.psp_amplitude = result['ic']['fit'].best_values['amp']
            if result['vc']['fit'] is not None and pair.synapse.psc_rise_time is not None:
                pair.synapse.psc_amplitude = result['vc']['fit'].best_values['amp']
        
    def job_records(self, job_ids, session):
        """Return a list of records associated with a list of job IDs.
        
        This method is used by drop_jobs to delete records for specific job IDs.
        """
        db = self.database
        q = session.query(db.RestingStateFit)
        q = q.filter(db.RestingStateFit.pair_id==db.Pair.id)
        q = q.filter(db.Pair.experiment_id==db.Experiment.id)
        q = q.filter(db.Experiment.ext_id.in_(job_ids))
        return q.all()
