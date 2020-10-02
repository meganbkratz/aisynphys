import pyqtgraph as pg
from aisynphys.database import default_db as db
from aisynphys.data import data_notes_db as notes_db
from sqlalchemy.orm import joinedload
from datetime import datetime

expt = db.query(db.Experiment).filter(db.Experiment.ext_id=='2020_05_27_exp2_TH').one()

def test1(expt, session=None):
    print('test1:',len(expt.pair_list), ' pairs')

    for pair in expt.pair_list[:64]:
        rec = notes_db.get_pair_notes_record(expt.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id, session=session)

def test2(expt):
    q = notes_db.db.default_session.query(notes_db.PairNotes)
    q = q.filter(notes_db.PairNotes.expt_id==expt.ext_id).all()

    print('test2:',len(expt.pair_list), ' pairs')
    for pair in expt.pair_list[:64]:
        #q1 = q.filter(notes_db.PairNotes.pre_cell_id==pair.pre_cell.ext_id)
        #q1 = q1.filter(notes_db.PairNotes.post_cell_id==pair.post_cell.ext_id)
        recs = []
        for rec in q:
            if rec.pre_cell_id == pair.pre_cell.ext_id:
                if rec.post_cell_id == pair.post_cell.ext_id:
                    recs.append(rec)

        if len(recs) == 0:
            pass
        elif len(recs) > 1:
            raise Exception("Multiple records found in pair_notes for pair %s %s %s!" % (expt_id, pre_cell_id, post_cell_id))
        pass

def test3(expt):
    print('test3:', len(expt.pair_list), ' pairs')
    for pair in expt.pair_list[:64]:
        q = notes_db.db.default_session.query(notes_db.PairNotes)
        q = q.filter(notes_db.PairNotes.expt_id==expt.ext_id)
        q = q.filter(notes_db.PairNotes.pre_cell_id==pair.pre_cell.ext_id)
        q = q.filter(notes_db.PairNotes.post_cell_id==pair.post_cell.ext_id)
        
        recs = q.all()
        if len(recs) == 0:
            pass
        elif len(recs) > 1:
            raise Exception("Multiple records found in pair_notes for pair %s %s %s!" % (expt_id, pre_cell_id, post_cell_id))
        pass


# prof = pg.debug.Profiler('main', disabled=False)
# s = notes_db.db.default_session
# prof.mark('establish default session')
# test1(expt, session=s)
# prof.mark('done with test1')
# test1(expt)
# prof.mark('done with test1')
# test2(expt)
# prof.mark('done with test2')
# test3(expt)
# prof.mark('done with test3')

# #test1(expt)
# #prof.mark('done with test1')
# #test3(expt)
# #prof.mark('done with test3')
# #test2(expt)
# #prof.mark('done with test2')
# prof.finish()


def populate(experiments=None, all_pairs=False, synapses=False, check_notes_db=False):
    """Populate the browser with a list of experiments.
    
    Parameters
    ----------
    experiments : list | None
        A list of Experiment instances. If None, then automatically query experiments from the default database.
    all_pairs : bool
        If False, then pairs with no qc-passed pulse responses are excluded
    synapses : bool
        If True, then only synaptically connected pairs are shown
    check_notes_db : bool | False
        If True, display an 'x' next to the synapse call for a pair if the pair is already in the notes db.
    """
    with pg.BusyCursor():
        # if all_pairs is set to True, all pairs from an experiment will be included regardless of whether they have data
        prof = pg.debug.Profiler('ExptBrowser.populate', disabled=False)
        
        session = db.session()
        prof.mark('got session')
        
        experiments.sort(key=lambda e: e.acq_timestamp if e.acq_timestamp is not None else 0)
        prof.mark('sorted experiments')
        for expt in experiments:
            date = expt.acq_timestamp
            date_str = datetime.fromtimestamp(date).strftime('%Y-%m-%d') if date is not None else None
            time_str = 'None' if expt.acq_timestamp is None else '%.3f'%expt.acq_timestamp
            slice = expt.slice
            expt_item = pg.TreeWidgetItem(map(str, [date_str, expt.ext_id, time_str, expt.rig_name, slice.species, expt.project_name, expt.target_region, slice.genotype, expt.acsf]))
            expt_item.expt = expt
            for pair in expt.pair_list:
                if all_pairs is False and pair.n_ex_test_spikes == 0 and pair.n_in_test_spikes == 0:
                    continue
                if synapses and not pair.has_synapse:
                    continue
                cells = '%s => %s' % (pair.pre_cell.ext_id, pair.post_cell.ext_id)
                conn = {True:"syn", False:"-", None:"?"}[pair.has_synapse]
                if check_notes_db:
                    rec = notes_db.get_pair_notes_record(expt.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id)
                    if rec is not None:
                        conn += '\t' + 'x'
                types = 'L%s %s => L%s %s' % (pair.pre_cell.target_layer or "?", pair.pre_cell.cre_type, pair.post_cell.target_layer or "?", pair.post_cell.cre_type)
                pair_item = pg.TreeWidgetItem([cells, conn, types])
                expt_item.addChild(pair_item)
                pair_item.pair = pair
                pair_item.expt = expt
                # also allow select by ext id
            prof.mark('add expt %s'%expt)

def populate_2(experiments=None, all_pairs=False, synapses=False, check_notes_db=False):
    """Populate the browser with a list of experiments.
    
    Parameters
    ----------
    experiments : list | None
        A list of Experiment instances. If None, then automatically query experiments from the default database.
    all_pairs : bool
        If False, then pairs with no qc-passed pulse responses are excluded
    synapses : bool
        If True, then only synaptically connected pairs are shown
    check_notes_db : bool | False
        If True, display an 'x' next to the synapse call for a pair if the pair is already in the notes db.
    """
    with pg.BusyCursor():
        # if all_pairs is set to True, all pairs from an experiment will be included regardless of whether they have data
        prof = pg.debug.Profiler('ExptBrowser.populate_2', disabled=False)
        
        session = db.session()
        prof.mark('got session')
        
        experiments.sort(key=lambda e: e.acq_timestamp if e.acq_timestamp is not None else 0)
        prof.mark('sorted experiments')
        for expt in experiments:
            date = expt.acq_timestamp
            date_str = datetime.fromtimestamp(date).strftime('%Y-%m-%d') if date is not None else None
            time_str = 'None' if expt.acq_timestamp is None else '%.3f'%expt.acq_timestamp
            slice = expt.slice
            expt_item = pg.TreeWidgetItem(map(str, [date_str, expt.ext_id, time_str, expt.rig_name, slice.species, expt.project_name, expt.target_region, slice.genotype, expt.acsf]))
            expt_item.expt = expt
            if check_notes_db:
                notes = notes_db.db.default_session.query(notes_db.PairNotes).filter(notes_db.PairNotes.expt_id==expt.ext_id).all()

            for pair in expt.pair_list:
                if all_pairs is False and pair.n_ex_test_spikes == 0 and pair.n_in_test_spikes == 0:
                    continue
                if synapses and not pair.has_synapse:
                    continue
                cells = '%s => %s' % (pair.pre_cell.ext_id, pair.post_cell.ext_id)
                conn = {True:"syn", False:"-", None:"?"}[pair.has_synapse]
                if check_notes_db:
                    recs = []
                    for rec in notes:
                        if rec.pre_cell_id == pair.pre_cell.ext_id:
                            if rec.post_cell_id == pair.post_cell.ext_id:
                                recs.append(rec)
                    if len(recs) > 1:
                        raise Exception("Multiple records found in pair_notes for pair %s %s %s!" % (expt.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id))
                    elif len(recs) == 1:
                        conn += '\t' + 'x'
                types = 'L%s %s => L%s %s' % (pair.pre_cell.target_layer or "?", pair.pre_cell.cre_type, pair.post_cell.target_layer or "?", pair.post_cell.cre_type)
                pair_item = pg.TreeWidgetItem([cells, conn, types])
                expt_item.addChild(pair_item)
                pair_item.pair = pair
                pair_item.expt = expt
            prof.mark('add expt %s'%expt)

prof = pg.debug.Profiler('main', disabled=False)
expts = db.query(db.Experiment).options(joinedload(db.Experiment.cell_list)).all()
prof.mark('got experiments')
#populate(expts, check_notes_db=True)
populate_2(expts, check_notes_db=True)
prof.finish()


