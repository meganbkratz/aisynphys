import pyqtgraph as pg
from collections import OrderedDict

from neuroanalysis.ui.plot_grid import PlotGrid
from neuroanalysis.fitting import fit_psp

from aisynphys.ui.pair_analysis.pair_analysis import ControlPanel, SuperLine
from aisynphys.ui.experiment_selector import ExperimentSelector
from aisynphys.ui.experiment_browser import ExperimentBrowser
from aisynphys.avg_response_fit import get_pair_avg_fits, response_query, sort_responses_opto

from aisynphys.database import default_db as db
from aisynphys.data import data_notes_db as notes_db
from aisynphys.data import PulseResponseList


class OptoPairAnalysisWindow(pg.QtGui.QWidget):

    default_latency = 2e-3
    nrmse_threshold = 4

    def __init__(self, default_session, notes_session):
        pg.QtGui.QWidget.__init__(self)

        self.db_session = default_session
        self.notes_db_session = notes_session

        self.layout = pg.QtGui.QGridLayout()
        self.layout.setContentsMargins(3,3,3,3)
        self.setLayout(self.layout) 

        self.h_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Horizontal)

        self.expt_selector = ExperimentSelector(default_session, notes_session, hashtags=[])
        self.expt_browser = ExperimentBrowser()
        self.ctrl_panel = ControlPanel()
        self.plot_grid = PlotGrid()
        self.latency_superline = SuperLine()

        self.ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.pair_param = pg.parametertree.Parameter.create(name='Current Pair', type='str', readonly=True)
        self.ptree.addParameters(self.pair_param)
        self.ptree.addParameters(self.ctrl_panel.user_params, showTop=False)

        self.fit_btn = pg.QtGui.QPushButton('Fit Responses')
        self.fit_ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.fit_ptree.addParameters(self.ctrl_panel.output_params, showTop=False)

        self.save_btn = pg.FeedbackButton('Save Analysis')

        v_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Vertical)
        for widget in [self.expt_browser, self.ptree, self.fit_btn, self.fit_ptree, self.save_btn]:
            v_splitter.addWidget(widget)

        self.h_splitter.addWidget(self.expt_selector)
        self.h_splitter.addWidget(v_splitter)
        self.h_splitter.addWidget(self.plot_grid)

        self.layout.addWidget(self.h_splitter)
        self.show()

        self.expt_selector.sigNewExperimentsRetrieved.connect(self.set_expts)
        self.expt_browser.itemSelectionChanged.connect(self.new_pair_selected)
        self.latency_superline.sigPositionChanged.connect(self.ctrl_panel.set_latency)
        self.fit_btn.clicked.connect(self.fit_responses)

    def set_expts(self, expts):
        with pg.BusyCursor():
            self.expt_browser.clear()
            has_data = self.expt_selector.data_type['Pairs with data']
            if not has_data:
                self.expt_browser.populate(experiments=expts, all_pairs=True)
            else:
                self.expt_browser.populate(experiments=expts)

    def reset(self):
        self.pulse_responses = None
        self.sorted_responses = None
        self.latency_superline.clear_lines()
        self.plot_grid.clear()
        self.ctrl_panel.fit_params.clearChildren()
        self.ctrl_panel.output_params.child('Comments', 'Hashtag').setValue('')
        self.ctrl_panel.output_params.child('Comments', '').setValue('')
        self.ctrl_panel.output_params.child('Warnings').setValue('')

    def new_pair_selected(self):
        with pg.BusyCursor():
            self.reset()
            #self.fit_compare.hide()
            #self.meta_compare.hide()
            selected = self.expt_browser.selectedItems()
            if len(selected) != 1:
                return
            item = selected[0]

            if hasattr(item, 'pair') is False:
                return
            pair = item.pair

            ## check to see if the pair has already been analyzed
            expt_id = pair.experiment.ext_id
            pre_cell_id = pair.pre_cell.ext_id
            post_cell_id = pair.post_cell.ext_id
            record = notes_db.get_pair_notes_record(expt_id, pre_cell_id, post_cell_id, session=self.notes_db_session)
            
            self.pair_param.setValue(pair)

            self.load_pair(pair)
            if record is not None:
                self.load_saved_fit(record)


    def load_pair(self, pair):
        """Pull responses from db, sort into groups and plot."""
        with pg.BusyCursor():
            self.pair = pair
            print ('loading responses for %s...' % pair)
            q = response_query(self.db_session, pair)
            self.pulse_responses = [q.PulseResponse for q in q.all()]
            print('got %d pulse responses' % len(self.pulse_responses))
                
            if pair.has_synapse is True:
                synapse_type = pair.synapse.synapse_type
            else:
                synapse_type = None
            pair_params = {'Synapse call': synapse_type, 'Gap junction call': pair.has_electrical}
            self.ctrl_panel.update_user_params(**pair_params)
            #self.ctrl_panel.update_fit_params(self.fit_params['fit'], fit_pass=True)
            

            sorted_responses = sort_responses_opto(self.pulse_responses)

            # filter out categories with no responses
            self.sorted_responses = OrderedDict()
            for k, v in sorted_responses.items():
                if len(v['qc_fail']) + len(v['qc_pass']) > 0:
                    self.sorted_responses[k]=v

            ## create plots and parameter items for each catagory in sorted responses
            self.ctrl_panel.create_new_fit_params([str(k) for k in self.sorted_responses.keys()])
            self.create_new_plots(self.sorted_responses.keys())
            self.fit_params = {key:{'initial':{}, 'fit':{}} for key in self.sorted_responses.keys()}

            self.plot_responses()

    def create_new_plots(self, categories):
        self.plot_grid.set_shape(len(categories), 1)

        unit_map={'vc':'A', 'ic':'V'}

        for i, key in enumerate(categories):
            plot = self.plot_grid[(i,0)]
            plot.setTitle(str(key))
            plot.setLabel('left', text="%dmV holding" % key[1], units=unit_map[key[0]])
            plot.setLabel('bottom', text='Time from stimulation', units='s')
            plot.addItem(self.latency_superline.new_line(self.default_latency))

    def plot_responses(self):
        qc_color = {'qc_pass': (255, 255, 255, 100), 'qc_fail': (255, 0, 0, 100)}
        
        for i, key in enumerate(self.sorted_responses.keys()):
            for qc, prs in self.sorted_responses[key].items():
                if len(prs) == 0:
                    continue
                prl = PulseResponseList(prs)
                post_ts = prl.post_tseries(align='pulse', bsub=True)
                
                for trace in post_ts:
                    item = self.plot_grid[(i,0)].plot(trace.time_values, trace.data, pen=qc_color[qc])
                    if qc == 'qc_fail':
                        item.setZValue(-10)
                    #self.items.append(item)
                if qc == 'qc_pass':
                    grand_trace = post_ts.mean()
                    item = self.plot_grid[(i,0)].plot(grand_trace.time_values, grand_trace.data, pen={'color': 'b', 'width': 2})
                    #self.items.append(item)
            self.plot_grid[(i,0)].autoRange()
            self.plot_grid[(i,0)].setXRange(-5e-3, 10e-3)

    def fit_responses(self):
        latency = self.ctrl_panel.user_params['User Latency']
        latency_window = [latency-500e-6, latency+500e-6]

        fit_color = {True: 'g', False: 'r'}

        with pg.ProgressDialog("curve fitting..", maximum=len(self.sorted_responses)) as dlg:
            for i, key in enumerate(self.sorted_responses.keys()): 
                clamp_mode = key[0]
                prs = self.sorted_responses[key]['qc_pass']
                if len(prs) == 0:
                    dlg += 1
                    continue
                    
                tsl = PulseResponseList(prs).post_tseries(align='pulse', bsub=True)
                average = tsl.mean()
                fit = fit_psp(average, latency_window, clamp_mode, baseline_like_psp=True)
                fit_ts = average.copy(data=fit.best_fit)
                    
                self.fit_params[key]['initial']['xoffset'] = latency
                self.fit_params[key]['fit']['nrmse'] = fit.nrmse()
                self.fit_params[key]['fit'].update(fit.best_values)

                fit_pass = fit.nrmse() < self.nrmse_threshold
                self.ctrl_panel.output_params.child('Fit parameters', str(key), 'Fit Pass').setValue(fit_pass)
                self.ctrl_panel.update_fit_param(key, self.fit_params[key]['fit'])

                self.plot_grid[(i,0)].plot(fit_ts.time_values, fit_ts.data, pen={'color':fit_color[fit_pass], 'width': 3})
                
                dlg += 1
                if dlg.wasCanceled():
                    raise Exception("User canceled fit")
    
        