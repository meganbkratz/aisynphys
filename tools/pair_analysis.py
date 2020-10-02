import sys, argparse

import pyqtgraph as pg

from aisynphys.database import default_db as db
import aisynphys.data.data_notes_db as notes_db
from aisynphys.ui.pair_analysis.pair_analysis import PairAnalysisWindow
from aisynphys.ui.pair_analysis.pair_analysis_opto import OptoPairAnalysisWindow

if __name__ == '__main__':
    app = pg.mkQApp()
    prof = pg.debug.Profiler('PairAnalysis', disabled=False)
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamps', type=str, nargs='*')
    parser.add_argument('--dbg', default=False, action='store_true')
    parser.add_argument('expt_id', type=str, nargs='?', default=None)
    parser.add_argument('pre_cell_id', type=str, nargs='?', default=None)
    parser.add_argument('post_cell_id', type=str, nargs='?', default=None)

    args = parser.parse_args(sys.argv[1:])

    prof.mark('parsed args')

    if args.dbg:
        pg.dbg()
        prof.mark('started debug')

    default_session = db.session()
    notes_session = notes_db.db.session()
    prof.mark('got db sessions')
    
    #mw = PairAnalysisWindow(default_session, notes_session)
    mw = OptoPairAnalysisWindow(default_session, notes_session)
    prof.mark('created gui window')

    # timestamps = [r.acq_timestamp for r in db.query(db.Experiment.acq_timestamp).all()]
    timestamps = []
    if args.timestamps is not None:
        timestamps = args.timestamps
    elif args.expt_id is not None:
        timestamps = [args.expt_id]

    if timestamps != []:
        q = default_session.query(db.Experiment).filter(db.Experiment.acq_timestamp.in_(timestamps))
        expts = q.all()
        mw.set_expts(expts)
        prof.mark('set experiments')

    if None not in (args.expt_id, args.pre_cell_id, args.post_cell_id):
        expt = db.experiment_from_ext_id(args.expt_id)
        pair = expt.pairs[args.pre_cell_id, args.post_cell_id]
        mw.experiment_browser.select_pair(pair.id)
        prof.mark('set pair')

    if sys.flags.interactive == 0:
        app.exec_()
