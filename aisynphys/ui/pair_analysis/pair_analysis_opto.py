import datetime
import pyqtgraph as pg
import numpy as np
from collections import OrderedDict

from neuroanalysis.ui.plot_grid import PlotGrid
from neuroanalysis.fitting.psp import fit_psp, Psp
import neuroanalysis.filter as filters
import neuroanalysis.event_detection as ev_detect
from neuroanalysis.data.dataset import TSeries

from aisynphys.ui.pair_analysis.pair_analysis import ControlPanel, SuperLine, comment_hashtag
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

        self.selectTabWidget = pg.QtGui.QTabWidget()
        self.analyzerTabWidget = pg.QtGui.QTabWidget()
        self.analyzerTabWidget.addTab(ResponseAnalyzer(self), "Responses")

        self.expt_selector = ExperimentSelector(default_session, notes_session, hashtags=[])
        self.expt_browser = ExperimentBrowser()
        #self.ctrl_panel = ControlPanel()
        self.plot_grid = PlotGrid()
        #self.plot_grid = FitPlotter()
        self.latency_superline = SuperLine()

        self.selectTabWidget.addTab(self.expt_selector, 'Experiment Select')

        self.ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.pair_param = pg.parametertree.Parameter.create(name='Current Pair', type='str', readonly=True)
        self.pair_param.addChild({'name':'Synapse call', 'type':'list', 'values':{'Excitatory': 'ex', 'Inhibitory': 'in', 'None': None}})
        self.pair_param.addChild({'name':'Gap junction call', 'type':'bool'})
        self.ptree.addParameters(self.pair_param)
        #self.category_param = pg.parametertree.Parameter.create(name='Categories', type='group')
        #self.ptree.addParameters(self.category_param)
        #self.comment_param = pg.parametertree.Parameter.create(name='Comments', type='group', children=[
        #    {'name': 'Hashtag', 'type': 'list', 'values': comment_hashtag, 'value': ''},
        #    {'name': '', 'type': 'text'}
        #])
        #self.ptree.addParameters(self.comment_param)
        #self.comment_param.child('Hashtag').sigValueChanged.connect(self.add_text_to_comments)
        
        #self.ptree.addParameters(self.ctrl_panel.user_params, showTop=False)

        #self.fit_btn = pg.QtGui.QPushButton('Fit Responses')
        #self.fit_btn.setEnabled(False)
        self.fit_ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.category_param = pg.parametertree.Parameter.create(name='Categories', type='group')
        self.fit_ptree.addParameters(self.category_param)
        self.comment_param = pg.parametertree.Parameter.create(name='Comments', type='group', children=[
            {'name': 'Hashtag', 'type': 'list', 'values': comment_hashtag, 'value': ''},
            {'name': '', 'type': 'text'}
        ])
        self.fit_ptree.addParameters(self.comment_param)
        self.comment_param.child('Hashtag').sigValueChanged.connect(self.add_text_to_comments)
        #self.fit_ptree.addParameters(self.ctrl_panel.output_params, showTop=False)

        self.save_btn = pg.FeedbackButton('Save Analysis')

        v_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Vertical)
        for widget in [self.expt_browser, self.ptree, self.fit_ptree, self.save_btn]:
            v_splitter.addWidget(widget)

        self.selectTabWidget.addTab(v_splitter, "Pair Select")

        #self.h_splitter.addWidget(self.expt_selector)
        #self.h_splitter.addWidget(v_splitter)
        self.h_splitter.addWidget(self.selectTabWidget)
        #self.h_splitter.addWidget(self.plot_grid)
        self.h_splitter.addWidget(self.analyzerTabWidget)

        self.layout.addWidget(self.h_splitter)
        self.show()

        self.expt_selector.sigNewExperimentsRetrieved.connect(self.set_expts)
        self.expt_browser.itemSelectionChanged.connect(self.new_pair_selected)
        #self.latency_superline.sigPositionChanged.connect(self.ctrl_panel.set_latency)
        #self.ctrl_panel.synapse.sigValueChanged.connect(self.synapse_call_changed)
        #self.fit_btn.clicked.connect(self.fit_responses)
        self.save_btn.clicked.connect(self.save_to_db)

    def set_expts(self, expts):
        with pg.BusyCursor():
            self.expt_browser.clear()
            has_data = self.expt_selector.data_type['Pairs with data']
            if not has_data:
                self.expt_browser.populate(experiments=expts, all_pairs=True)
            else:
                self.expt_browser.populate(experiments=expts)

        if len(expts) > 0:
            self.selectTabWidget.setCurrentIndex(1)

    def reset(self):
        self.pulse_responses = None
        self.sorted_responses = None
        self.analyzers = {}
        #self.latency_superline.clear_lines()
        #self.plot_grid.clear()
        self.analyzerTabWidget.clear()
        #self.ctrl_panel.fit_params.clearChildren()
        #self.ctrl_panel.output_params.child('Comments', 'Hashtag').setValue('')
        #self.ctrl_panel.output_params.child('Comments', '').setValue('')
        #self.ctrl_panel.output_params.child('Warnings').setValue('')

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

    def synapse_call_changed(self):
        value = self.ctrl_panel.synapse.value()
        if value is None:
            self.fit_btn.setEnabled(False)
            ## probably also need to clear fit parameter items and plot items here
        else:
            self.fit_btn.setEnabled(True)

    def add_text_to_comments(self):
        text = self.comment_param['Hashtag']
        comments = self.comment_param['']
        update_comments = comments + text + '\n'
        self.comment_param.child('').setValue(update_comments)


    def load_pair(self, pair):
        """Pull responses from db, sort into groups and plot."""
        with pg.BusyCursor():
            self.pair = pair
            print ('loading responses for %s...' % pair)
            q = response_query(self.db_session, pair)
            self.pulse_responses = [q.PulseResponse for q in q.all()]
            print('got %d pulse responses' % len(self.pulse_responses))
                
            if pair.has_synapse is True and pair.synapse is not None:
                synapse_type = pair.synapse.synapse_type
            else:
                synapse_type = None

            self.pair_param.child('Synapse call').setValue(synapse_type)
            self.pair_param.child('Gap junction call').setValue(pair.has_electrical)
            #pair_params = {'Synapse call': synapse_type, 'Gap junction call': pair.has_electrical}
            #self.ctrl_panel.update_user_params(**pair_params)
            #self.ctrl_panel.update_fit_params(self.fit_params['fit'], fit_pass=True)
            

            sorted_responses = sort_responses_opto(self.pulse_responses)

            # filter out categories with no responses
            self.sorted_responses = OrderedDict()
            for k, v in sorted_responses.items():
                if len(v['qc_fail']) + len(v['qc_pass']) > 0:
                    self.sorted_responses[k]=v

            ## create plots and parameter items for each catagory in sorted responses
            #self.ctrl_panel.create_new_fit_params([str(k) for k in self.sorted_responses.keys()])
            #self.create_new_plots(self.sorted_responses.keys())
            self.create_new_analyzers(self.sorted_responses.keys())
            self.fit_params = {key:{'initial':{}, 'fit':{}} for key in self.sorted_responses.keys()}

            self.plot_responses()

    # def create_new_plots(self, categories):
    #     self.plot_grid.set_shape(len(categories), 1)

    #     unit_map={'vc':'A', 'ic':'V'}

    #     for i, key in enumerate(categories):
    #         plot = self.plot_grid[(i,0)]
    #         plot.setTitle(str(key))
    #         plot.setLabel('left', text="%dmV holding" % key[1], units=unit_map[key[0]])
    #         plot.setLabel('bottom', text='Time from stimulation', units='s')
    #         plot.addItem(self.latency_superline.new_line(self.default_latency))

    def create_new_analyzers(self, categories):
        for i, key in enumerate(categories):
            self.analyzers[key] = ResponseAnalyzer(host=self, key=key)
            self.analyzerTabWidget.addTab(self.analyzers[key], str(key))
            self.category_param.addChild({'name':str(key), 'type':'group', 'children':[
                {'name':'status', 'type':'str', 'value': 'not analyzed', 'readOnly':True},
                {'name':'number_of_events', 'type':'str', 'readOnly':True, 'visible':False},
                {'name':'user_passed_fit', 'type':'str', 'readOnly':True, 'visible':False}
                ]})
            self.analyzers[key].sigNewAnalysisAvailable.connect(self.got_new_analysis)

    def got_new_analysis(self, result):
           # res = {'category_name':self.key,
           #     'fit': self.current_fit,
           #     'fit_pass':self.event_params.child('event_0')['Fit Pass'],
           #     'n_events':len(self.event_params.children()),
           #     'event_times':[p['user_latency'] for p in self.event_params.children()]}
        param = self.category_param.child(str(result['category_name']))
        param.child('status').setValue('done')
        param.child('number_of_events').setValue(str(result['n_events']))
        param.child('number_of_events').show()
        param.child('user_passed_fit').setValue(str(result['fit_pass']))
        param.child('user_passed_fit').show()

        param.fit = result['fit']
        param.event_times = result['event_times']

    def plot_responses(self):
        #qc_color = {'qc_pass': (255, 255, 255, 100), 'qc_fail': (255, 0, 0, 100)}
        
        for i, key in enumerate(self.sorted_responses.keys()):
            self.analyzers[key].plot_responses(self.sorted_responses[key])
            # for qc, prs in self.sorted_responses[key].items():
            #     if len(prs) == 0:
            #         continue
            #     prl = PulseResponseList(prs)
            #     post_ts = prl.post_tseries(align='pulse', bsub=True)
                
            #     for trace in post_ts:
            #         item = self.plot_grid[(i,0)].plot(trace.time_values, trace.data, pen=qc_color[qc])
            #         if qc == 'qc_fail':
            #             item.setZValue(-10)
            #         #self.items.append(item)
            #     if qc == 'qc_pass':
            #         grand_trace = post_ts.mean()
            #         item = self.plot_grid[(i,0)].plot(grand_trace.time_values, grand_trace.data, pen={'color': 'b', 'width': 2})
            #         #self.items.append(item)
            # self.plot_grid[(i,0)].autoRange()
            # self.plot_grid[(i,0)].setXRange(-5e-3, 10e-3)

    # def fit_responses(self):
    #     latency = self.ctrl_panel.user_params['User Latency']
    #     latency_window = [latency-500e-6, latency+500e-6]

    #     fit_color = {True: 'g', False: 'r'}

    #     with pg.ProgressDialog("curve fitting..", maximum=len(self.sorted_responses)) as dlg:
    #         for i, key in enumerate(self.sorted_responses.keys()): 
    #             clamp_mode = key[0]
    #             prs = self.sorted_responses[key]['qc_pass']
    #             if len(prs) == 0:
    #                 dlg += 1
    #                 continue
                    
    #             tsl = PulseResponseList(prs).post_tseries(align='pulse', bsub=True)
    #             average = tsl.mean()
    #             #fit = fit_psp(average, latency_window, clamp_mode, baseline_like_psp=True)
    #             fits = self.fit_psps_opto(average, latency_window, clamp_mode, baseline_like_psp=True)
    #             d = np.zeros(average.data.shape, average.data.dtype)
    #             for j, fit in fits.items():
    #                 v = fit.values
    #                 fit_data = Psp.psp_func(average.time_values, v['xoffset'], v['yoffset'], v['rise_time'], v['decay_tau'], v['amp'], v['rise_power'])
    #                 self.plot_grid[(i,0)].plot(average.time_values, fit_data, pen=pg.intColor(j, minHue=100))
    #                 d += fit_data

    #             self.plot_grid[(i,0)].plot(average.time_values, d, pen='r')


    #             #fit_ts = average.copy(data=fit.best_fit)
                    
    #             #self.fit_params[key]['initial']['xoffset'] = latency
    #             #self.fit_params[key]['fit']['nrmse'] = fit.nrmse()
    #             #self.fit_params[key]['fit'].update(fit.best_values)

    #             #fit_pass = fit.nrmse() < self.nrmse_threshold
    #             #self.ctrl_panel.output_params.child('Fit parameters', str(key), 'Fit Pass').setValue(fit_pass)
    #             #self.ctrl_panel.update_fit_param(key, self.fit_params[key]['fit'])

    #             #self.plot_grid[(i,0)].plot(fit_ts.time_values, fit_ts.data, pen={'color':fit_color[fit_pass], 'width': 3})
                
    #             dlg += 1
    #             if dlg.wasCanceled():
    #                 raise Exception("User canceled fit")

    #     self.generate_warnings()



    # def fit_psps_opto(self, average, latency_window, clamp_mode, baseline_like_psp=True):

    #     tau=20e-3 ## for epsps, 50ms for ipsps

    #     filtered = filters.bessel_filter(average, 4000, order=2, btype='low', bidir=True)
    #     exp_dec = ev_detect.exp_deconvolve(filtered, tau)

    #     threshold = 0.3*max(exp_dec.data)

    #     ## index, length, sum, peak, peak_index, time, duration, area, peak_time
    #     events = ev_detect.threshold_events(exp_dec, threshold)
    #     sign=1
    #     fits={}

    #     for i, ev in enumerate(events):
    #         window = [ev['time']-0.002, ev['time']+0.002]

    #         if i == len(events)-1:
    #             data = exp_dec.time_slice(ev['time']-0.0001, None)
    #         else:
    #             data = exp_dec.time_slice(ev['time']-0.0001, events[i+1]['time'])

    #         d = np.concatenate([np.zeros(100), data.data, np.zeros(300)])
    #         tv = np.concatenate([np.arange(data.t0-data.dt*100, data.t0-data.dt/10., data.dt), data.time_values, np.arange(data.time_values[-1]+data.dt, data.time_values[-1]+data.dt*301, data.dt)])
    #         data = TSeries(data=d, time_values=tv)
    #         data = ev_detect.exp_reconvolve(data, tau)
    #         fits[i] = fit_psp(data, window, clamp_mode, sign, exp_baseline=False)

    #     return fits

    def load_saved_fit(self, record):
        raise Exception('implement me!')

    def generate_warnings(self):
        self.warnings = None

    def save_to_db(self):
        fit_pass = {}
        for key in self.sorted_responses.keys():
            fit_pass[key] = self.ctrl_panel.output_params['Fit parameters', str(key), 'Fit Pass']

        expt_id = self.pair.experiment.ext_id
        pre_cell_id = self.pair.pre_cell.ext_id
        post_cell_id = self.pair.post_cell.ext_id
        meta = {
            'expt_id': expt_id,
            'pre_cell_id': pre_cell_id,
            'post_cell_id': post_cell_id,
            'synapse_type': self.ctrl_panel.user_params['Synapse call'],
            'gap_junction': self.ctrl_panel.user_params['Gap junction call'],
            'comments': self.ctrl_panel.output_params['Comments', ''],
        }

        if self.ctrl_panel.user_params['Synapse call'] is not None:
            meta.update({
            'fit_parameters': self.fit_params,
            'fit_pass': fit_pass,
            'fit_warnings': self.warnings,
            })
        
        session = notes_db.db.session(readonly=False)
        record = notes_db.get_pair_notes_record(expt_id, pre_cell_id, post_cell_id, session=session)

        if record is None:
            entry = notes_db.PairNotes(
                expt_id=expt_id,
                pre_cell_id=pre_cell_id,
                post_cell_id=post_cell_id, 
                notes=meta,
                modification_time=datetime.datetime.now(),
            )
            session.add(entry)
            session.commit()
        else:
            self.print_pair_notes(meta, record)
            msg = pg.QtGui.QMessageBox.question(None, "Pair Analysis", 
                "The record you are about to save conflicts with what is in the Pair Notes database.\nYou can see the differences highlighted in red.\nWould you like to overwrite?",
                pg.QtGui.QMessageBox.Yes | pg.QtGui.QMessageBox.No)
            if msg == pg.QtGui.QMessageBox.Yes:
                record.notes = meta
                record.modification_time = datetime.datetime.now()
                session.commit() 
            else:
                raise Exception('Save Cancelled')
        session.close()
    

class ResponseAnalyzer(pg.QtGui.QWidget):

    sigNewAnalysisAvailable = pg.Qt.QtCore.Signal(object)

    def __init__(self, host=None, key=None):
        pg.QtGui.QWidget.__init__(self)
        layout = pg.QtGui.QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        self.setLayout(layout)

        h_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Horizontal)
        layout.addWidget(h_splitter)

        self.responses = None ## holder for {pass:[tseries], fail:[tseries]}
        self.average_response = None ## holder for the average response
        self.deconvolved = None ## holder for the deconvolved average response
        self.deconvolved_events = None ## holder for the recordarray of events found in the deconvolved trace

        self.key = key
        self.clamp_mode = self.key[0] if self.key is not None else None
        self.host = host
        self.param_tree = pg.parametertree.ParameterTree()
        self.plot_grid=PlotGrid()
        #self.fit_btn = pg.QtGui.QPushButton("Fit responses")
        self.add_btn = pg.QtGui.QPushButton("Add analysis")
        self.add_btn.clicked.connect(self.add_analysis_btn_clicked)

        v_widget = pg.QtGui.QWidget()
        v_layout = pg.QtGui.QVBoxLayout()
        
        v_layout.setSpacing(3)
        v_widget.setLayout(v_layout)
        v_layout.addWidget(self.param_tree)
        #v_layout.addWidget(self.fit_btn)
        v_layout.addWidget(self.add_btn)
        v_layout.setContentsMargins(0,0,0,0)
        v_widget.setContentsMargins(0,0,0,0)

        self.plot_grid.set_shape(2,1)
        self.plot_grid[0,0].setTitle('data')
        self.plot_grid[1,0].setTitle('processing')

        h_splitter.addWidget(v_widget)
        h_splitter.addWidget(self.plot_grid)

        self.processing_params = pg.parametertree.Parameter.create(name='Processing', type='group', children=[
            {'name':'pre-deconvolve bessel', 'type':'float', 'value':6e3, 'suffix':'Hz', 'siPrefix':True, 'dec':True},
            {'name':'deconvolve tau', 'type':'float', 'value':15e-3, 'suffix':'s', 'siPrefix':True, 'step':1e-3},
            {'name':'post-deconvolve bessel', 'type':'float', 'value':1e3, 'suffix':'Hz', 'siPrefix':True, 'dec':True},
            {'name':'event_threshold_fraction', 'type':'float', 'value':0.3, 'step':0.1}
            ])
        self.param_tree.addParameters(self.processing_params)
        self.event_params = pg.parametertree.Parameter.create(name="Events", type='group')
        self.param_tree.addParameters(self.event_params)
        #self.event_params.sigAddNew.connect(self.add_event_param)

        for param in self.processing_params.children():
            param.sigValueChanged.connect(self.process_data)

    def add_event_param(self, name, latency):
        visible=int(name[-1])==0
        color = pg.intColor(len(self.event_params.children())+1)
        param = pg.parametertree.Parameter.create(name=name, type='group', children=[
            LatencyParam(name='user_latency', value=latency, color=color),
            {'name': 'display_color', 'type':'color', 'value':color, 'readOnly':True},
            {'name': 'Fit parameter', 'type': 'str', 'readonly': True, 'value': 'Amplitude, Latency, Rise time, Decay tau, NRMSE', 'visible':visible},
            {'name': 'Fit value', 'type': 'str', 'readonly': True, 'visible':visible},
            {'name': 'Fit Pass', 'type': 'bool', 'visible':visible},
            {'name': 'Warning', 'type':'str', 'readonly':True, 'value':'', 'visible':False},
            {'name': 'Fit event', 'type':'action', 'visible':visible}
            ])
        self.event_params.addChild(param)
        param.child('Fit event').sigActivated.connect(self.fit_event)
        self.plot_grid[(0,0)].addItem(param.child('user_latency').line)
        return param

    def fit_event(self, btn_param):
        event_param = btn_param.parent()
        i = event_param.event_number
        latency = event_param['user_latency']

        window = [latency - 0.0002, latency+0.0002]
        ev = self.deconvolved_events[i]

        ##### Snip out the section of the deconvolved trace containing the event, reconvolve it and fit. - currently not working too well
        # if i == len(self.deconvolved_events)-1:
        #     data = self.deconvolved.time_slice(ev['time']-0.0001, None)
        # else:
        #     data = self.deconvolved.time_slice(ev['time']-0.0001, self.deconvolved_events[i+1]['time'])

        # n = int(0.05/data.dt)
        # d = np.concatenate([np.zeros(100), data.data, np.zeros(n)])
        # tv = np.concatenate([np.arange(data.t0-data.dt*100, data.t0-data.dt/10., data.dt), data.time_values, np.arange(data.time_values[-1]+data.dt, data.time_values[-1]+data.dt*(n+1), data.dt)])
        # data = TSeries(data=d, time_values=tv)
        # data = ev_detect.exp_reconvolve(data, self.processing_params['deconvolve tau'])
        # fit = fit_psp(data, window, self.clamp_mode, sign=0, exp_baseline=False)

        # v = fit.values

        #### Option2 - fit snippet of original data, but feed fitting algorithm good guesses for amp and rise time 
        ## basically make travis's measurments, then try to fit to get tau
        ## this seems good so far -- for some data small differences in latency window create big differences in the fits (it turns out this is resolved by making the search spacing for the fit finer), but for most the fits are pretty resilient
        ##                              -> maybe we want to fit with some different rise_time/amp seeds and see how similar the fits are, similar -> probably good, variation -> don't trust
        if i == len(self.deconvolved_events)-1: ## last event
            data = self.average_response.time_slice(latency-0.001, latency+0.05)
        else:
            for param in self.event_params.children():
                if param.event_number == i+1:
                    next_latency = param['user_latency']
                    break
            data = self.average_response.time_slice(latency-0.001, next_latency)

        pre_bessel = self.processing_params['pre-deconvolve bessel']
        filtered = filters.bessel_filter(data, pre_bessel, order=4, btype='low', bidir=True)

        #print("-------------")
        #print("latency:", latency)
        fits = {}
        #for x in [-0.00009, 0, 0.000090]:
        for x in [0]: ### I think this issue was resolved by adding the fine fit parameters to fit_psp()
            peak_ind = np.argwhere(max(filtered.data)==filtered.data)[0][0]
            peak_time = filtered.time_at(peak_ind)
            rise_time = peak_time-(latency+x)
            amp = filtered.value_at(peak_time) - filtered.value_at(latency+x)
            fit = fit_psp(filtered, (window[0]+x, window[1]+x), self.clamp_mode, sign=0, exp_baseline=False, init_params={'rise_time':rise_time, 'amp':amp}, fine_search_spacing=filtered.dt, fine_search_window_width=100e-6)
            #fit=fit_psp(filtered, (window[0]+x, window[1]+x), self.clamp_mode, sign=0, exp_baseline=False)
            #print('init_rise_time:%f, init_amp:%f, tau:%f, x:%f, rise:%s, amp:%s'%(rise_time, amp, fit.values['decay_tau'], fit.values['xoffset'], fit.values['rise_time'], fit.values['amp']))
            fits[x] = fit

        ## check to see how different decay_tau is, flag fit as bad if too high -- I think this was resolved
        # taus = [f.values['decay_tau'] for f in fits.values()]
        # print('tau stdev:', np.std(taus))
        # if np.std(taus) > 0.001: ## don't really know yet what value this should be
        #     bad_fit=True
        # else:
        #     bad_fit=False

        ## plot fit
        self.current_fit = fits[0]
        v = fits[0].values
        y=Psp.psp_func(self.average_response.time_values,v['xoffset'], v['yoffset'], v['rise_time'], v['decay_tau'], v['amp'], v['rise_power'])
        self.plot_grid[(0,0)].plot(self.average_response.time_values, y, pen=event_param['display_color'])

        ## display fit params
        self.update_fit_param_display(event_param, fit)
        # self.set_bad_fit(event_param, bad_fit, taus)


    def update_fit_param_display(self, param, fit):
        names = ['amp', 'xoffset', 'rise_time', 'decay_tau', 'nrmse']
        if self.clamp_mode == 'vc':
            suffix = ['A', 's', 's', 's', '']
        elif self.clamp_mode == 'ic':
            suffix = ['V', 's', 's', 's', '']

        format_list = []
        for p in zip(names, suffix):
            if fit.values.get(p[0]) is None:
                format_list.append('nan')
            else:
                value = fit.values[p[0]]
                if p[0] == 'nrmse':
                    p_format = ('%0.2f' % value)
                else:
                    p_format = pg.siFormat(value, suffix=p[1])
                format_list.append(p_format)
        output = ", ".join(format_list)

        param.child('Fit value').setValue(output)

    # def set_bad_fit(self, param, bad_fit, taus):
    #     if bad_fit:
    #         param.child('Fit Pass').setValue(False)
    #         for item in param.child('Fit Pass').items.keys():
    #             item.setDisabled(True)
    #         param.child('Warning').setValue('Fit was not robust. Different decay taus found:'+str([pg.siFormat(t, suffix='s') for t in taus]))
    #         param.child('Warning').show()
    #     else:
    #         for item in param.child('Fit Pass').items.keys():
    #             if item.isDisabled():
    #                 item.setDisabled(False)
    #         param.child('Warning').setValue('')
    #         param.child('Warning').hide()

    def update_events(self, events):
        for ch in self.event_params.children():
            self.plot_grid[(0,0)].removeItem(ch.child('user_latency').line)
        self.event_params.clearChildren()

        times = events['time']
        zeroth = np.argwhere(times == times[times > 0].min())

        for i, ev in enumerate(events):
            param = self.add_event_param('event_%i'%(i-zeroth), ev['time'])
            param.event_number=i



    def plot_responses(self, responses):
        self.responses = responses
        qc_color = {'qc_pass': (255, 255, 255, 100), 'qc_fail': (255, 0, 0, 100)}
        
        for qc, prs in responses.items():
            if len(prs) == 0:
                continue
            prl = PulseResponseList(prs)
            post_ts = prl.post_tseries(align='pulse', bsub=True)
            
            for trace in post_ts:
                item = self.plot_grid[(0,0)].plot(trace.time_values, trace.data, pen=qc_color[qc])
                if qc == 'qc_fail':
                    item.setZValue(-10)
                #self.items.append(item)
            if qc == 'qc_pass':
                self.average_response = post_ts.mean()
                item = self.plot_grid[(0,0)].plot(self.average_response.time_values, self.average_response.data, pen={'color': 'b', 'width': 2})
                #self.items.append(item)
        self.plot_grid[(0,0)].autoRange()
        self.process_data()

    def process_data(self):

        plot = self.plot_grid[(1,0)]
        plot.clear()

        plot.plot(self.average_response.time_values, self.average_response.data)

        pre_bessel = self.processing_params['pre-deconvolve bessel']
        post_bessel = self.processing_params['post-deconvolve bessel']
        tau = self.processing_params['deconvolve tau']
        threshold_frac = self.processing_params['event_threshold_fraction']


        pre_filtered = filters.bessel_filter(self.average_response, pre_bessel, order=4, btype='low', bidir=True)
        self.deconvolved = ev_detect.exp_deconvolve(pre_filtered, tau)
        post_filtered = filters.bessel_filter(self.deconvolved, post_bessel, order=1, btype='low', bidir=True)

        plot.plot(pre_filtered.time_values, pre_filtered.data, pen='b')
        plot.plot(self.deconvolved.time_values, self.deconvolved.data, pen='r')
        plot.plot(post_filtered.time_values, post_filtered.data, pen='g')

        threshold = threshold_frac*max(post_filtered.data)
        plot.plot(post_filtered.time_values, [threshold]*len(post_filtered), pen='y')

        ## index, length, sum, peak, peak_index, time, duration, area, peak_time
        self.deconvolved_events = ev_detect.threshold_events(post_filtered, threshold) ## find events in deconvolved trace
        plot.addItem(pg.VTickGroup(self.deconvolved_events['time']))

        self.update_events(self.deconvolved_events)

        # dec_times = list(dec_events['time']) + [None]
        # event_times = []
        # for i in range(len(dec_events['time'])):
        #     ev = deconvolved.time_slice(dec_times[i], dec_times[i+1])
        #     d = np.concatenate([np.full((100), ev.data[0]), ev.data, np.full((int(0.1/ev.dt)), ev.data[-1])])
        #     tv = np.concatenate([np.arange(ev.t0-ev.dt*100, ev.t0-ev.dt/10., ev.dt), ev.time_values, np.arange(ev.time_values[-1]+ev.dt, ev.time_values[-1]+0.1+ev.dt, ev.dt)])
        #     data1 = TSeries(data=d, time_values=tv)
        #     data = ev_detect.exp_reconvolve(data1, tau)
        #     thresh = (max(data.data)-data.data[-1])*threshold_frac ## need to figure this out for negative going events too
        #     event = ev_detect.threshold_events(data, thresh)
        #     if len(event) > 1:
        #         raise Exception('stop')
        #     event_times.append(event['time'][0])


        
        # plot.addItem(pg.VTickGroup(event_times, (0.7, 1)))

    def add_analysis_btn_clicked(self):
        res = {'category_name':self.key,
               'fit': self.current_fit,
               'fit_pass':self.event_params.child('event_0')['Fit Pass'],
               'n_events':len(self.event_params.children()),
               'event_times':[p['user_latency'] for p in self.event_params.children()]}
        self.sigNewAnalysisAvailable.emit(res)







class LatencyParam(pg.parametertree.parameterTypes.SimpleParameter):

    def __init__(self, *args, **kargs):
        kargs.update({'siPrefix':True, 'suffix':'s', 'type':'float', 'step':100e-6})
        pg.parametertree.parameterTypes.SimpleParameter.__init__(self, *args, **kargs)

        self.line = pg.InfiniteLine(pos=self.value(), pen=kargs.get('color'), movable=True)

        self.sigValueChanged.connect(self.update_line_position)
        self.line.sigPositionChanged.connect(self.update_param_value)

    def update_line_position(self, param):
        with pg.SignalBlock(self.line.sigPositionChanged, self.update_param_value):
            self.line.setValue(param.value())

    def update_param_value(self, line):
        with pg.SignalBlock(self.sigValueChanged, self.update_line_position):
            self.setValue(line.value())












        