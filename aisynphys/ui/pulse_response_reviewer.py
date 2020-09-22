import sys
import pyqtgraph as pg
from aisynphys.database import default_db as db
import aisynphys.data.data_notes_db as notes_db
from neuroanalysis.ui.plot_grid import PlotGrid
from aisynphys.avg_response_fit import response_query, sort_responses, sort_responses_2p
from neuroanalysis.baseline import float_mode
from optoanalysis.qc import opto_pulse_response_qc_pass
from aisynphys.qc import pulse_response_qc_pass


class PulseResponseReviewer(pg.QtGui.QWidget):

    sigNewDataSaved = pg.QtCore.Signal(object)

    def __init__(self, db_session, notes_db_session, mode=None, pair=None):
        pg.QtGui.QWidget.__init__(self)

        self.db_session = db_session
        self.notes_db_session = notes_db_session

        self.layout = pg.QtGui.QGridLayout()
        self.layout.setContentsMargins(3,3,3,3)
        self.setGeometry(100, 50, 800, 600)
        self.setLayout(self.layout) 

        self.response_tree = pg.parametertree.ParameterTree(showHeader=False)
        self.save_btn = pg.FeedbackButton("Save to notes db")
        self.pair_label = pg.QtGui.QLabel("Current Pair: %s"%(pair))

        self.plot_grid =  PlotGrid()
        self.plot_grid.set_shape(1,1)
        self.plot_grid[0,0].setTitle('All responses')
        #self.plot_grid[1,0].setTitle('selected response')
        #self.plot_grid[2,0].setTitle('selected presynaptic spike')

        self.qc_text = pg.DataTreeWidget()
        self.qc_text.setHeaderLabels(['QC failures', 'type', 'value'])

        h_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Horizontal)
        self.layout.addWidget(h_splitter)

        v_widget = pg.QtGui.QWidget(self)
        v_layout = pg.QtGui.QVBoxLayout()
        v_widget.setLayout(v_layout)
        v_layout.addWidget(self.pair_label)
        v_layout.addWidget(self.response_tree)
        v_layout.addWidget(self.save_btn)

        h_splitter.addWidget(v_widget)

        v_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Vertical)
        v_splitter.addWidget(self.plot_grid)
        v_splitter.addWidget(self.qc_text)
        h_splitter.addWidget(v_splitter)

        self.show()

        self._mode = mode

        if pair is not None:
            self.load(pair=pair)

        self.response_tree.itemSelectionChanged.connect(self.pulse_response_selection_changed)
        self.save_btn.clicked.connect(self.save_to_db)

    @property
    def mode(self):
        known_modes = ['opto', 'multipatch']

        if self._mode is None:
            raise Exception("Error: no mode specified at initialization. Options are: %s" % known_modes)
        if self._mode not in known_modes:
            raise ValueError("Error: don't know how to use the given mode (%s). Valid options are: %s" %(self._mode, known_modes))

        return self._mode

    def load(self, pair=None, sorted_responses=None):
        self.pair=pair
        self.sorted_responses = sorted_responses
        if self.sorted_responses is None:
            if pair is None:
                raise Exception("Need to supply either pair or sorted_responses")

            q = response_query(self.db_session, pair)
            responses = [q.PulseResponse for q in q.all()]
            record = notes_db.get_pair_notes_record(pair.experiment.ext_id, pair.pre_cell.ext_id, pair.post_cell.ext_id, session=self.notes_db_session)
            if record is not None:
                user_qc = record.notes.get('user_qc_changes', [])
            else:
                user_qc = []

            if self.mode == 'opto':
                self.sorted_responses = sort_responses_2p(responses, user_qc=user_qc)
            elif self.mode == 'multipatch':
                self.sorted_responses = sort_responses(responses)
            else:
                raise Exception('Sorting responses for %s mode is not yet implemented.' % self.mode)

        pass_dict={True:'qc_pass', False:'qc_fail'}
        for key, items in self.sorted_responses.items():
            param = pg.parametertree.Parameter.create(name=str(key), type='group')
            for qc, prs in items.items():
                for pr in prs:
                    name = 'sweep_%i_pulse_%i'%(pr.recording.sync_rec.ext_id, pr.stim_pulse.pulse_number)
                    #ex_qc_pass, in_qc_pass, failures = qc_function(pr.recording.patch_clamp_recording, [pr.data_start_time, pr.post_tseries.time_values[-1]])
                    child = pg.parametertree.Parameter.create(name=name, type='list', values=['qc_pass', 'qc_fail'], value=qc, default=pass_dict[pr.ex_qc_pass or pr.in_qc_pass])
                    child.pulse_response = pr
                    child.failures = pr.meta.get('qc_failures')
                    param.addChild(child)
            self.response_tree.addParameters(param)

    def pulse_response_selection_changed(self):
        pr_param = self.response_tree.selectedItems()[0].param
        pr = pr_param.pulse_response
        has_presynaptic_data = pr.stim_pulse.first_spike_time != None

        category_param = pr_param.parent()

        self.plot_grid[0,0].clear()
        for child in category_param.children():
            #### do our own baseline subtraction and pulse alignment instead of 
            #### using PulseResponseList so that we can keep pulse responses attached to traces
            stim_time = child.pulse_response.stim_pulse.onset_time
            ts = child.pulse_response.post_tseries
            start_time = max(ts.t0, stim_time-5e-3)
            baseline_data = ts.time_slice(start_time, stim_time).data
            if len(baseline_data) == 0:
                baseline = ts.data[0]
            else:
                baseline = float_mode(baseline_data)
            ts = ts - baseline

            if has_presynaptic_data:
                stim_time=child.pulse_response.stim_pulse.first_spike_time

            ts = ts.copy(t0=ts.t0 - stim_time)

            pen = {'qc_pass':(255,255,255,100), 'qc_fail':(255,0,0,100)}[child.value()]

            self.plot_grid[0,0].plot(ts.time_values, ts.data, pen=pen)

            if child == pr_param:
                item = self.plot_grid[0,0].plot(ts.time_values, ts.data, pen=(0, 200, 100))
                item.setZValue(20)

        self.qc_text.setData(pr_param.failures)

    def save_to_db(self):
        expt_id = self.pair.experiment.ext_id
        pre_cell_id = self.pair.pre_cell.ext_id
        post_cell_id = self.pair.post_cell.ext_id

        qc_changes = []
        d = {'qc_pass': "User manually changed qc to pass.",
             'qc_fail': "User manually failed qc."}

        for category_param in self.response_tree.topLevelItems():
            for pr_param in category_param.param.children():
                if pr_param.value() != pr_param.defaultValue():
                    pr = pr_param.pulse_response
                    pr_id = (expt_id, pr.recording.sync_rec.ext_id, pr.recording.device_name, pr.stim_pulse.pulse_number)
                    qc_changes.append((pr_id, pr_param.value(), d[pr_param.value()]))
        print('1:', qc_changes)

        session = notes_db.db.session(readonly=False)
        record = notes_db.get_pair_notes_record(expt_id, pre_cell_id, post_cell_id, session=session)
        if record is None:
            entry = notes_db.PairNotes(
                expt_id=expt_id,
                pre_cell_id=pre_cell_id,
                post_cell_id=post_cell_id, 
                notes={'user_qc_changes':qc_changes},
                #modification_time=datetime.datetime.now(),
            )
            session.add(entry)
            session.commit()
        else:
            meta = record.notes
            print("meta:", meta)
            print("qc_changes:", qc_changes)
            meta.update({'user_qc_changes':qc_changes}) 
            print('meta2:', meta)
            record.notes = meta
            print('record.notes:', record.notes)
            session.commit() 
        session.close()

        self.sigNewDataSaved.emit(self)

### TODO:
# - implement reloading
# - change how we set the default qc_pass or qc_fail (use qc function instead) (currently it uses the 
#   value from sort_responses, which checks our list, so will result in oscilating behavior, I think)
#            --> changed to use value from main db (can't use qc functions because they need data objects instead of database objects)
# - finish debugging
# - maybe don't allow override of qc_fail from the qc function

if __name__ == '__main__':
    app = pg.mkQApp()
    pg.dbg()

    pr = db.default_session.query(db.PulseResponse).filter(db.PulseResponse.pair_id != None).all()[0]
    pair = pr.pair

    prr = PulseResponseReviewer(db.default_session, notes_db.db.default_session, 'opto', pair=pair)

    if sys.flags.interactive == 0:
        app.exec_()



