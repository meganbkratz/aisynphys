import datetime, itertools
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
from aisynphys.avg_response_fit import get_pair_avg_fits, response_query

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
        self.setGeometry(280, 130, 1500, 900)
        self.setLayout(self.layout) 

        self.h_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Horizontal)

        self.selectTabWidget = pg.QtGui.QTabWidget()
        self.analyzerTabWidget = pg.QtGui.QTabWidget()
        self.analyzerTabWidget.addTab(ResponseAnalyzer(self), "Responses")

        self.expt_selector = ExperimentSelector(default_session, notes_session, hashtags=[])
        self.expt_browser = ExperimentBrowser()
        self.plot_grid = PlotGrid()
        self.latency_superline = SuperLine()

        self.selectTabWidget.addTab(self.expt_selector, 'Experiment Select')
        self.selectTabWidget.addTab(self.expt_browser, "Pair Select")

        self.ptree = pg.parametertree.ParameterTree(showHeader=False)
        self.pair_param = pg.parametertree.Parameter.create(name='Current Pair', type='str', readonly=True)
        self.pair_param.addChild({'name':'Synapse call', 'type':'list', 'values':{'':'not specified', 'Excitatory': 'ex', 'Inhibitory': 'in', 'None': None, "TBD":'tbd'}, 'value':''})
        self.pair_param.addChild({'name':'Gap junction call', 'type':'bool'})
        self.ptree.addParameters(self.pair_param)
        self.category_param = pg.parametertree.Parameter.create(name='Categories', type='group')
        self.ptree.addParameters(self.category_param)
        self.comment_param = pg.parametertree.Parameter.create(name='Comments', type='group', children=[
            {'name': 'Hashtag', 'type': 'list', 'values': comment_hashtag, 'value': ''},
            {'name': '', 'type': 'text'}
        ])
        self.ptree.addParameters(self.comment_param)
        self.comment_param.child('Hashtag').sigValueChanged.connect(self.add_text_to_comments)

        self.save_btn = pg.FeedbackButton('Save Analysis')

        v_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Vertical)
        for widget in [self.ptree, self.save_btn]:
            v_splitter.addWidget(widget)


        self.selectTabWidget.addTab(v_splitter, "Current Pair")

        self.h_splitter.addWidget(self.selectTabWidget)
        self.h_splitter.addWidget(self.analyzerTabWidget)

        self.layout.addWidget(self.h_splitter)
        self.show()

        self.expt_selector.sigNewExperimentsRetrieved.connect(self.set_expts)
        self.expt_browser.itemSelectionChanged.connect(self.new_pair_selected)

        self.save_btn.clicked.connect(self.save_to_db)

    def set_expts(self, expts):
        with pg.BusyCursor():
            self.expt_browser.clear()
            has_data = self.expt_selector.data_type['Pairs with data']
            if not has_data:
                self.expt_browser.populate(experiments=expts, all_pairs=True, check_notes_db=True)
            else:
                self.expt_browser.populate(experiments=expts, check_notes_db=True)

        if len(expts) > 0:
            self.selectTabWidget.setCurrentIndex(1)

    def reset(self):
        self.pulse_responses = None
        self.sorted_responses = None
        self.analyzers = {}
        self.analyzerTabWidget.clear()
        self.category_param.clearChildren()

    def new_pair_selected(self):
        with pg.BusyCursor():
            self.reset()
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
            self.selectTabWidget.setCurrentIndex(2)

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
                synapse_type = 'not specified'

            self.pair_param.child('Synapse call').setValue(synapse_type)
            self.pair_param.child('Gap junction call').setValue(pair.has_electrical)
            

            sorted_responses = self.sort_responses(self.pulse_responses)

            # filter out categories with no responses
            self.sorted_responses = OrderedDict()
            for k, v in sorted_responses.items():
                if len(v['qc_fail']) + len(v['qc_pass']) > 0:
                    self.sorted_responses[k]=v

            self.create_new_analyzers(self.sorted_responses.keys())

            self.plot_responses()

    def load_saved_fit(self, record):
        notes = record.notes
        self.pair_param.child('Synapse call').setValue(notes.get('synapse_type', 'not specified'))
        self.pair_param.child('Gap junction call').setValue(notes.get('gap_junction', False))
        self.comment_param.child('').setValue(notes.get('comments', ''))

        saved_categories = list(notes['categories'].keys()) # sanity check

        for cat in self.analyzers.keys():
            data = notes['categories'].get(str(cat), None)
            saved_categories.remove(str(cat))
            if data is not None:
                p = self.category_param.child(str(cat))
                p.child('status').setValue('previously analyzed')
                p.child('number_of_events').setValue(data['n_events'])
                p.child('number_of_events').show()
                p.child('user_passed_fit').setValue(data['fit_pass'])
                p.child('user_passed_fit').show()
                self.analyzers[cat].load_saved_data(data)

        if len(saved_categories) > 0: # sanity check
            raise Exception("Previously saved categories %s, but no analyzer was found for these."%saved_categories)

    def sort_responses(self, pulse_responses):
        ex_limits = [-80e-3, -60e-3]
        in_limits1 = [-60e-3, -45e-3]
        in_limits2 = [-10e-3, 10e-3] ## some experiments were done with Cs+ and held at 0mv
        distance_limit = 10e-6

        modes = ['vc', 'ic']
        holdings = [-70, -55, 0]
        powers = []
        for pr in pulse_responses:
            if pr.stim_pulse.meta is None:
                powers.append(None)
            else:
                powers.append(pr.stim_pulse.meta.get('pockel_cmd'))
        powers = list(set(powers))
        #powers = list(set([pr.stim_pulse.meta.get('pockel_cmd') for pr in pulse_responses if pr.stim_pulse.meta is not None else None]))

        keys = itertools.product(modes, holdings, powers)

        ### Need to differentiate between laser-stimulated pairs and electrode-electode pairs
        ## I would like to do this in a more specific way, ie: if the device type == Fidelity. -- this is in the pipeline branch of aisynphys 
        ## But that needs to wait until devices are in the db. (but also aren't they?)
        ## Also, we're going to have situations where the same pair has laser responses and 
        ##   electrode responses when we start getting 2P guided pair patching, and this will fail then
        # if pulse_responses[0].pair.pre_cell.electrode is None:  
        #     powers = list(set([pr.stim_pulse.meta.get('pockel_cmd') for pr in pulse_responses]))
        #     keys = itertools.product(modes, holdings, powers)

        # else:
        #     keys = itertools.product(modes, holdings)

        sorted_responses = OrderedDict({k:{'qc_pass':[], 'qc_fail':[]} for k in keys})

        qc = {False: 'qc_fail', True: 'qc_pass'}

        for pr in pulse_responses:
            clamp_mode = pr.recording.patch_clamp_recording.clamp_mode
            holding = pr.recording.patch_clamp_recording.baseline_potential
            power = pr.stim_pulse.meta.get('pockel_cmd') if pr.stim_pulse.meta is not None else None

            offset_distance = pr.stim_pulse.meta.get('offset_distance', 0) if pr.stim_pulse.meta is not None else 0
            if offset_distance is None: ## early photostimlogs didn't record the offset between the stimulation plane and the cell
                offset_distance = 0

            if in_limits1[0] <= holding < in_limits1[1]:
                qc_pass = qc[pr.in_qc_pass and offset_distance < distance_limit]
                sorted_responses[(clamp_mode, -55, power)][qc_pass].append(pr)

            elif in_limits2[0] <= holding < in_limits2[1]:
                qc_pass = qc[pr.in_qc_pass and offset_distance < distance_limit]
                sorted_responses[(clamp_mode, 0, power)][qc_pass].append(pr)

            elif ex_limits[0] <= holding < ex_limits[1]:
                qc_pass = qc[pr.ex_qc_pass and offset_distance < distance_limit]
                sorted_responses[(clamp_mode, -70, power)][qc_pass].append(pr)

        return sorted_responses


    def create_new_analyzers(self, categories):
        for i, key in enumerate(categories):
            self.analyzers[key] = ResponseAnalyzer(host=self, key=key)
            self.analyzerTabWidget.addTab(self.analyzers[key], str(key))
            self.category_param.addChild({'name':str(key), 'type':'group', 'children':[
                {'name':'status', 'type':'str', 'value': 'not analyzed', 'readonly':True},
                {'name':'number_of_events', 'type':'str', 'readonly':True, 'visible':False},
                {'name':'user_passed_fit', 'type':'str', 'readonly':True, 'visible':False}
                ]})
            self.analyzers[key].sigNewAnalysisAvailable.connect(self.got_new_analysis)

    def got_new_analysis(self, result):
        param = self.category_param.child(str(result['category_name']))
        param.child('status').setValue('done')
        param.child('number_of_events').setValue(str(result['n_events']))
        param.child('number_of_events').show()
        if result['n_events'] > 0:
            param.child('user_passed_fit').setValue(str(result['fit_pass']))
            param.child('user_passed_fit').show()

        param.fit = result['fit']
        param.event_times = result['event_times']
        param.initial_params = result['initial_params']

    def plot_responses(self):        
        for i, key in enumerate(self.sorted_responses.keys()):
            self.analyzers[key].plot_responses(self.sorted_responses[key])


    def generate_warnings(self):
        self.warnings = None

    def save_to_db(self):

        try:
            synapse_call = self.pair_param['Synapse call']
            if synapse_call == 'not specified':
                raise Exception("Please make a synapse call before saving to db.")

            expt_id = self.pair.experiment.ext_id
            pre_cell_id = self.pair.pre_cell.ext_id
            post_cell_id = self.pair.post_cell.ext_id
            meta = {
                'expt_id': self.pair.experiment.ext_id,
                'pre_cell_id': self.pair.pre_cell.ext_id,
                'post_cell_id': self.pair.post_cell.ext_id,
                'synapse_type': synapse_call,
                'gap_junction': self.pair_param['Gap junction call'],
                'comments': self.comment_param[''],
                'categories':{str(key):None for key in self.sorted_responses.keys()}
            }

            if synapse_call is not None:
                for key in self.sorted_responses.keys():
                    param = self.category_param.child(str(key))
                    if param['status'] in ['done', 'previously analyzed']:
                        meta['categories'][str(key)]={
                            'initial_parameters':param.initial_params,
                            'fit_parameters': param.fit,
                            'fit_pass': param['user_passed_fit'],
                            'n_events': param['number_of_events'],
                            'event_times':param.event_times
                            }
            
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
                #self.print_pair_notes(meta, record.notes)
                #msg = pg.QtGui.QMessageBox.question(None, "Pair Analysis", 
                #    "The record you are about to save conflicts with what is in the Pair Notes database.\nYou can see the differences highlighted in red.\nWould you like to overwrite?",
                #    pg.QtGui.QMessageBox.Yes | pg.QtGui.QMessageBox.No)
                msg = CompareDialog("There is already a record for %s in the %s database.\nYou can see the differences highlighted in red.\nWould you like to overwrite?"%(self.pair, notes_db.name), OrderedDict([('Previously saved',record.notes), ('New',meta)]))
                if msg == pg.QtGui.QDialog.Accepted:
                    record.notes = meta
                    record.modification_time = datetime.datetime.now()
                    session.commit() 
                else:
                    raise Exception('Save Cancelled')
            session.close()
            self.save_btn.success('Saved.')

        except:
            self.save_btn.failure('Error')
            raise

class CompareDialog(pg.QtGui.QDialog):


    def __init__(self, message, data, execute=True):
        """A dialog box that displays a message and a DiffTreeWidget. An Okay button
        is connected to the dialogs accept slot and a Cancel button is connected to 
        reject.

        Parameters
        ----------
        message : str
            The message to be displayed at the top of the dialog.
        data : dict
            The data to be compared. This should be a dict with 2 entries. The key 
            for each entry should a label for the data, and the value should be the
            data to compare. These values are passed directly to DiffTreeWidget.setData().
        execute : bool | True
            If True, start self.exec_() at the end of __init__ 
        """
        pg.QtGui.QDialog.__init__(self, None)
        self.setSizeGripEnabled(True)

        layout = pg.QtGui.QGridLayout()
        self.setLayout(layout)
        text = pg.QtGui.QLabel(message)

        okBtn = pg.QtGui.QPushButton("Okay")
        cancelBtn = pg.QtGui.QPushButton('Cancel')
        cancelBtn.setDefault(True)

        self.compareTree = pg.DiffTreeWidget()
        def sizeHint():
            return pg.QtCore.QSize(1000,600)

        self.compareTree.sizeHint = sizeHint

        layout.addWidget(text, 0,0,1,3)
        layout.addWidget(self.compareTree, 1, 0, 1, 3)
        layout.addWidget(cancelBtn, 2, 0)
        layout.addWidget(okBtn, 2, 2)
        layout.setRowStretch(1, 10)

        layout.setSpacing(3)
        layout.setContentsMargins(0,0,0,0)
        self.setContentsMargins(0,0,0,0)

        if len(data) != 2:
            raise Exception("data argument should be a {name1:data1, name2:data2}")

        vals = []
        for i, k in enumerate(data.keys()):
            self.compareTree.trees[i].setHeaderLabels([k, 'type', 'value'])
            vals.append(data[k])

        self.compareTree.setData(vals[0], vals[1])

        cancelBtn.clicked.connect(self.reject)
        okBtn.clicked.connect(self.accept)

        if execute:
            self.exec_()




        


    

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
        self.event_counter = 0

        self.key = key
        self.clamp_mode = self.key[0] if self.key is not None else None
        self.has_presynaptic_data = (key != None) and (key[2] == None)
        self.host = host
        self.param_tree = pg.parametertree.ParameterTree()
        self.plot_grid=PlotGrid()
        self.add_btn = pg.FeedbackButton("Add analysis")
        self.add_btn.clicked.connect(self.add_analysis_btn_clicked)

        v_widget = pg.QtGui.QWidget()
        v_layout = pg.QtGui.QVBoxLayout()
        
        v_layout.setSpacing(3)
        v_widget.setLayout(v_layout)
        v_layout.addWidget(self.param_tree)
        v_layout.addWidget(self.add_btn)
        v_layout.setContentsMargins(0,0,0,0)
        v_widget.setContentsMargins(0,0,0,0)

        self.plot_grid.set_shape(2,1)
        self.plot_grid[0,0].setTitle('data')
        self.plot_grid[1,0].setTitle('presynaptic trace')
        self.plot_grid.grid.ci.layout.setRowStretchFactor(0, 2)
        if not self.has_presynaptic_data:
            self.plot_grid[1,0].hide()

        h_splitter.addWidget(v_widget)
        h_splitter.addWidget(self.plot_grid)

        self.event_params = pg.parametertree.Parameter.create(name="Events", type='group', addText='Add event')
        self.param_tree.addParameters(self.event_params)
        self.event_params.sigAddNew.connect(self.add_event_param)

    def load_saved_data(self, data):
        for time in data['event_times']:
            p = self.add_event_param()
            p.child('user_latency').setValue(time)

        for p in self.event_params.children():
            if p._should_have_fit:
                self.plot_fit(p, data['fit_parameters'], data['initial_parameters'])
                self.update_fit_param_display(p, data['fit_parameters'])
                p.child('Fit passes qc').setValue({'':None, 'True':True, 'False':False}.get(data['fit_pass']))

    def add_event_param(self):
        n = len(self.event_params.children())
        if n == 0:
            latency = 10e-3
            visible = True
        else:
            latency = max(self.event_params.children()[-1]['user_latency']+2e-3, 10e-3)
            visible = self.event_params.children()[-1]['user_latency'] < 0

        color = pg.intColor(self.event_counter)
        self.event_counter += 1
        param = pg.parametertree.Parameter.create(name='event_%i'%n, type='group', removable=True, children=[
            LatencyParam(name='user_latency', value=latency, color=color),
            {'name': 'display_color', 'type':'color', 'value':color, 'readonly':True},
            {'name': 'Fit results', 'type':'group', 'readonly':True, 'visible':visible, 'expanded':True, 'children':[
                {'name':'amplitude', 'type':'str', 'readonly':True},
                {'name':'latency', 'type':'str', 'readonly':True},
                {'name':'rise time', 'type':'str', 'readonly':True},
                {'name':'decay tau', 'type':'str', 'readonly':True},
                {'name':'NRMSE', 'type':'str', 'readonly':True}]},
            {'name': 'Fit passes qc', 'type': 'list', 'values':{'': None, 'True':True, 'False':False}, 'value':'', 'visible':visible},
            {'name': 'Warning', 'type':'str', 'readonly':True, 'value':'', 'visible':False},
            {'name': 'Fit event', 'type':'action', 'visible':visible, 'renamable':False}
            ])
        self.event_params.addChild(param)
        param.child('Fit event').sigActivated.connect(self.fit_event)
        param.child('user_latency').sigValueChanged.connect(self.event_latency_changed)
        self.plot_grid[(0,0)].addItem(param.child('user_latency').line)
        param.sigRemoved.connect(self.event_param_removed)
        return param

    def event_param_removed(self, param):
        if hasattr(param, '_fit_plot_item'):
            self.plot_grid[(0,0)].removeItem(param._fit_plot_item)

        line = param.child('user_latency').line
        self.plot_grid[(0,0)].removeItem(line)

        if len(self.event_params.children()) > 0:
            for i, p in enumerate(self.event_params.children()):
                p.setName('event_%i'%i)
            ## trigger re-evaluation of which events should be fit
            self.event_latency_changed(self.event_params.children()[0].child('user_latency')) 

    def event_latency_changed(self, latency_param):
        i = int(latency_param.parent().name()[-1])
        if i > 0:
            pre = self.event_params.child('event_%i'%(i-1))
            bounds = pre.child('user_latency').line.bounds()
            pre.child('user_latency').line.setBounds([bounds[0], latency_param.value()])
        if i < len(self.event_params.children())-1:
            post = self.event_params.child('event_%i'%(i+1))
            bounds = post.child('user_latency').line.bounds()
            post.child('user_latency').line.setBounds([latency_param.value(), bounds[1]])

        first_event=None
        for p in self.event_params.children():
            if p['user_latency'] > 0:
                first_event = p.name()
                break
        for p in self.event_params.children():
            if p.name() != first_event:
                self.show_fit_params(p, False)
            else:
                self.show_fit_params(p, True)

    def show_fit_params(self, param, show):
        """*param* is an event param, *show* is boolean"""
        param.child('Fit results').show(show)
        param.child('Fit passes qc').show(show)
        param.child('Fit event').show(show)
        if show:
            param._should_have_fit = True
        else:
            param._should_have_fit = False
            if hasattr(param, '_fit_plot_item'):
                self.plot_grid[(0,0)].removeItem(param._fit_plot_item)
            for name in ['amplitude', 'latency', 'rise time', 'decay tau', 'NRMSE']:
                param.child('Fit results').child(name).setValue('')


    def fit_event(self, btn_param):
        event_param = btn_param.parent()
        i = int(event_param.name()[-1])
        latency = event_param['user_latency']

        window = [latency - 0.0002, latency+0.0002]

        #### fit snippet of original data, but feed fitting algorithm good guesses for amp and rise time 
        ## basically make travis's measurments, then try to fit to get tau
        if i == len(self.event_params.children())-1: ## last event
            data = self.average_response.time_slice(latency-0.001, latency+0.05)
        else:
            for param in self.event_params.children():
                if int(param.name()[-1]) == i+1:
                    next_latency = param['user_latency']
                    break
            data = self.average_response.time_slice(latency-0.001, next_latency)

        filtered = filters.bessel_filter(data, 6000, order=4, btype='low', bidir=True)

        peak_ind = np.argwhere(max(abs(filtered.data))==abs(filtered.data))[0][0]
        peak_time = filtered.time_at(peak_ind)
        rise_time = peak_time-(latency)
        amp = filtered.value_at(peak_time) - filtered.value_at(latency)
        init_params = {'rise_time':rise_time, 'amp':amp}
        fit = fit_psp(filtered, (window[0], window[1]), self.clamp_mode, sign=0, exp_baseline=False, init_params=init_params, fine_search_spacing=filtered.dt, fine_search_window_width=100e-6)

        self.plot_fit(event_param, fit.values, init_params)
        ## display fit params
        self.update_fit_param_display(event_param, fit.values)

    def plot_fit(self, event_param, fit_values, init_values):
        ## plot fit
        v = fit_values
        y=Psp.psp_func(self.average_response.time_values,v['xoffset'], v['yoffset'], v['rise_time'], v['decay_tau'], v['amp'], v['rise_power'])

        if hasattr(event_param, '_fit_plot_item'):
            self.plot_grid[(0,0)].removeItem(event_param._fit_plot_item)
        event_param._fit_plot_item = self.plot_grid[(0,0)].plot(self.average_response.time_values, y, pen=event_param['display_color'])
        event_param._fit_values = v
        event_param._initial_fit_guesses = init_values
        

    def update_fit_param_display(self, param, v):
        if self.clamp_mode == 'vc':
            param.child('Fit results').child('amplitude').setValue(pg.siFormat(v['amp'], suffix='A'))
        elif self.clamp_mode == 'ic':
            param.child('Fit results').child('amplitude').setValue(pg.siFormat(v['amp'], suffix='V'))

        param.child('Fit results').child('latency').setValue(pg.siFormat(v['xoffset'], suffix='s'))
        param.child('Fit results').child('rise time').setValue(pg.siFormat(v['rise_time'], suffix='s'))
        param.child('Fit results').child('decay tau').setValue(pg.siFormat(v['decay_tau'], suffix='s'))

        nrmse = v.get('nrmse')
        if nrmse is None:
            param.child('Fit results').child('NRMSE').setValue('nan')
        else:
            param.child('Fit results').child('NRMSE').setValue('%0.2f'%nmrse)


    def plot_responses(self, responses):
        self.responses = responses
        qc_color = {'qc_pass': (255, 255, 255, 100), 'qc_fail': (255, 0, 0, 100)}
        
        for qc, prs in responses.items():
            if len(prs) == 0:
                continue
            prl = PulseResponseList(prs)
            if not self.has_presynaptic_data:
                post_ts = prl.post_tseries(align='pulse', bsub=True)
            else:
                post_ts = prl.post_tseries(align='spike', bsub=True)
                pre_ts = prl.pre_tseries(align='spike', bsub=True)
            
            for trace in post_ts:
                item = self.plot_grid[(0,0)].plot(trace.time_values, trace.data, pen=qc_color[qc])
                if qc == 'qc_fail':
                    item.setZValue(-10)
            if qc == 'qc_pass':
                self.average_response = post_ts.mean()
                item = self.plot_grid[(0,0)].plot(self.average_response.time_values, self.average_response.data, pen={'color': 'b', 'width': 2})

            if self.has_presynaptic_data:
                for pr, spike in zip(prl, pre_ts):
                    pre_qc = 'qc_pass' if pr.stim_pulse.n_spikes == 1 else 'qc_fail'
                    item = self.plot_grid[(1,0)].plot(spike.time_values, spike.data, pen=qc_color[qc])
                    if qc == 'qc_fail':
                        item.setZValue(-10)


        self.plot_grid[(0,0)].autoRange()
        self.plot_grid[(0,0)].setLabel('bottom', text='Time from stimulus', units='s')
        self.plot_grid[(0,0)].setLabel('left', units={'ic':'V', 'vc':'A'}.get(self.clamp_mode))

        if self.has_presynaptic_data:
            self.plot_grid[(1,0)].autoRange()
            self.plot_grid[(1,0)].setLabel('bottom', units='s')
            self.plot_grid[(1,0)].setLabel('left', units='V')

    def add_analysis_btn_clicked(self):
        try:

            fit = None
            fit_pass = None
            initial_params=None
            evs = [] ## sanity check that we only have one event fit
            for p in self.event_params.children():
                if p._should_have_fit:
                    fit_pass = p['Fit passes qc']
                    if fit_pass is None:
                        raise Exception('Please specify whether fit passes qc for %s'%p.name())

                    evs.append(p.name())
                    fit = p._fit_values
                    initial_params=p._initial_fit_guesses
                    
            if len(evs) > 1:
                ### need to figure out why this is happening
                raise Exception('Error: More than one fit found. This is a bug')

            res = {'category_name':self.key,
                   'fit': fit,
                   'fit_pass':fit_pass,
                   'initial_params':initial_params,
                   'n_events':len(self.event_params.children()),
                   'event_times':[p['user_latency'] for p in self.event_params.children()]}
            self.sigNewAnalysisAvailable.emit(res)
            self.add_btn.success('Added.')

        except:
            self.add_btn.failure('Error')
            raise



class LatencyParam(pg.parametertree.parameterTypes.SimpleParameter):

    def __init__(self, *args, **kargs):
        kargs.update({'siPrefix':True, 'suffix':'s', 'type':'float', 'step':100e-6})
        pg.parametertree.parameterTypes.SimpleParameter.__init__(self, *args, **kargs)

        self.line = pg.InfiniteLine(pos=self.value(), pen=kargs.get('color'), movable=True, hoverPen=pg.mkPen('y', width=2))

        self.sigValueChanged.connect(self.update_line_position)
        self.line.sigPositionChanged.connect(self.update_param_value)

    def update_line_position(self, param):
        with pg.SignalBlock(self.line.sigPositionChanged, self.update_param_value):
            self.line.setValue(param.value())

    def update_param_value(self, line):
        with pg.SignalBlock(self.sigValueChanged, self.update_line_position):
            self.setValue(line.value())














        