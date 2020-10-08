import sys, argparse

import pyqtgraph as pg

from aisynphys.database import default_db as db
import aisynphys.data.data_notes_db as notes_db
from aisynphys.ui.pair_analysis.pair_analysis import PairAnalysisWindow
from aisynphys.ui.pair_analysis.pair_analysis_opto import OptoPairAnalysisWindow

if __name__ == '__main__':
    app = pg.mkQApp()
    parser = argparse.ArgumentParser()
    parser.add_argument('--timestamps', type=str, nargs='*')
    parser.add_argument('--dbg', default=False, action='store_true')
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('expt_id', type=str, nargs='?', default=None)
    parser.add_argument('pre_cell_id', type=str, nargs='?', default=None)
    parser.add_argument('post_cell_id', type=str, nargs='?', default=None)

    args = parser.parse_args(sys.argv[1:])

    if args.dbg:
        pg.dbg()

    default_session = db.session()
    notes_session = notes_db.db.session()
    
    modes = ['multipatch', 'opto']
    if args.mode is None:
        print('Warning: no mode specified, defaulting to multipatch. (use --mode=mode, options are %s)'%modes)
        args.mode = 'multipatch'
    if args.mode.lower() == 'multipatch':
        mw = PairAnalysisWindow(default_session, notes_session)
    elif args.mode.lower() == 'opto':
        mw = OptoPairAnalysisWindow(default_session, notes_session)
    else:
        raise Exception('Mode %s is not valid. Options are %s' % (args.mode, modes))

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

    if None not in (args.expt_id, args.pre_cell_id, args.post_cell_id):
        expt = db.experiment_from_ext_id(args.expt_id)
        pair = expt.pairs[args.pre_cell_id, args.post_cell_id]
        mw.experiment_browser.select_pair(pair.id)

    if sys.flags.interactive == 0:
        app.exec_()
