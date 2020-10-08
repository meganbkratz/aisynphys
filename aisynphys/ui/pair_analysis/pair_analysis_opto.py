import datetime, itertools
import pyqtgraph as pg
import numpy as np
from collections import OrderedDict

from neuroanalysis.ui.plot_grid import PlotGrid
from neuroanalysis.fitting.psp import fit_psp, Psp
import neuroanalysis.filter as filters
from neuroanalysis.baseline import float_mode
import neuroanalysis.event_detection as ev_detect
from neuroanalysis.data.dataset import TSeries, TSeriesList

from aisynphys.ui.pair_analysis.pair_analysis import ControlPanel, SuperLine, comment_hashtag
from aisynphys.ui.experiment_selector import ExperimentSelector
from aisynphys.ui.experiment_browser import ExperimentBrowser
from aisynphys.avg_response_fit import response_query_2p, sort_responses_2p, get_average_response_2p, fit_event_2p, sort_responses_into_categories_2p

from aisynphys.database import default_db as db
from aisynphys.data import data_notes_db as notes_db
from aisynphys.data import PulseResponseList


### list of available reasons for excluding a response from the average that is fit.
EXCLUSION_REASONS = ["Response includes spontaneous event",
                    "Response doesnt include an event",
                    "Response is an outlier",
                    "Unable to determine spike time",
                    "Response failed automated qc"]


class OptoPairAnalysisWindow(pg.QtGui.QWidget):

    default_latency = 2e-3
    nrmse_threshold = 4

    def __init__(self, default_session, notes_session):
        pg.QtGui.QWidget.__init__(self)

        self.db_session = default_session
        self.notes_db_session = notes_session

        self.layout = pg.QtGui.QGridLayout()
        self.layout.setContentsMargins(3,3,3,3)
        self.setGeometry(100, 50, 1100, 700)
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
        v_splitter.setStretchFactor(0,5)


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
            self.pulse_responses = response_query_2p(self.db_session, pair, include_data=True)
            print('got %d pulse responses' % len(self.pulse_responses))
                
            if pair.has_synapse is True and pair.synapse is not None:
                synapse_type = pair.synapse.synapse_type
            else:
                synapse_type = 'not specified'

            self.pair_param.child('Synapse call').setValue(synapse_type)
            self.pair_param.child('Gap junction call').setValue(pair.has_electrical)
            

            #self.sorted_responses = sort_responses_2p(self.pulse_responses, exclude_empty=True)
            self.sorted_responses = sort_responses_into_categories_2p(self.pulse_responses, exclude_empty=True)

            # # filter out categories with no responses
            # self.sorted_responses = OrderedDict()
            # for k, v in sorted_responses.items():
            #     if len(v['qc_fail']) + len(v['qc_pass']) > 0:
            #         self.sorted_responses[k]=v

            self.create_new_analyzers(self.sorted_responses.keys())
            self.set_responses()

    def load_saved_fit(self, record):

        notes = record.notes
        self.pair_param.child('Synapse call').setValue(notes.get('synapse_type', 'not specified'))
        self.pair_param.child('Gap junction call').setValue(notes.get('gap_junction', False))
        self.comment_param.child('').setValue(notes.get('comments', ''))

        saved_categories = list(notes['categories'].keys()) # sanity check

        for cat in self.analyzers.keys():
            data = notes['categories'].get(str(cat), {})
            saved_categories.remove(str(cat))
            if len(data) > 0:
                p = self.category_param.child(str(cat))
                p.child('status').setValue('previously analyzed')
                p.child('number_of_events').setValue(data['n_events'])
                p.child('number_of_events').show()
                p.child('user_passed_fit').setValue(data['fit_pass'])
                p.child('user_passed_fit').show()
                p.result = data
                #p.fit = data['fit_parameters']
                #p.initial_params = data['initial_parameters']
                #p.event_times = data['event_times']
                #p.included_responses = data.get('included_responses')
                #p.excluded_responses = data.get('excluded_responses')
                self.analyzers[cat].load_saved_data(data)

        if len(saved_categories) > 0: # sanity check
            raise Exception("Previously saved categories %s, but no analyzer was found for these."%saved_categories)

    # def reload_pulse_responses(self):
    #     record = notes_db.get_pair_notes_record(self.pair.experiment.ext_id, self.pair.pre_cell.ext_id, self.pair.post_cell.ext_id, session=self.notes_db_session)
    #     if record is None:
    #         return

    #     user_qc = record.notes.get('user_qc_changes', None)
    #     if user_qc is None:
    #         return

    #     self.sorted_responses = sort_responses_2p(self.pulse_responses, exclude_empty=True, user_qc=user_qc)
    #     self.plot_responses()

    # @classmethod
    # def sort_responses(cls, pulse_responses):
    #     ex_limits = [-80e-3, -60e-3]
    #     in_limits1 = [-60e-3, -45e-3]
    #     in_limits2 = [-10e-3, 10e-3] ## some experiments were done with Cs+ and held at 0mv
    #     distance_limit = 10e-6

    #     modes = ['vc', 'ic']
    #     holdings = [-70, -55, 0]
    #     powers = []
    #     for pr in pulse_responses:
    #         if pr.stim_pulse.meta is None:
    #             powers.append(None)
    #         else:
    #             powers.append(pr.stim_pulse.meta.get('pockel_cmd'))
    #     powers = list(set(powers))
    #     #powers = list(set([pr.stim_pulse.meta.get('pockel_cmd') for pr in pulse_responses if pr.stim_pulse.meta is not None else None]))

    #     keys = itertools.product(modes, holdings, powers)

    #     ### Need to differentiate between laser-stimulated pairs and electrode-electode pairs
    #     ## I would like to do this in a more specific way, ie: if the device type == Fidelity. -- this is in the pipeline branch of aisynphys 
    #     ## But that needs to wait until devices are in the db. (but also aren't they?)
    #     ## Also, we're going to have situations where the same pair has laser responses and 
    #     ##   electrode responses when we start getting 2P guided pair patching, and this will fail then
    #     # if pulse_responses[0].pair.pre_cell.electrode is None:  
    #     #     powers = list(set([pr.stim_pulse.meta.get('pockel_cmd') for pr in pulse_responses]))
    #     #     keys = itertools.product(modes, holdings, powers)

    #     # else:
    #     #     keys = itertools.product(modes, holdings)

    #     sorted_responses = OrderedDict({k:{'qc_pass':[], 'qc_fail':[]} for k in keys})

    #     qc = {False: 'qc_fail', True: 'qc_pass'}

    #     for pr in pulse_responses:
    #         clamp_mode = pr.recording.patch_clamp_recording.clamp_mode
    #         holding = pr.recording.patch_clamp_recording.baseline_potential
    #         power = pr.stim_pulse.meta.get('pockel_cmd') if pr.stim_pulse.meta is not None else None

    #         offset_distance = pr.stim_pulse.meta.get('offset_distance', 0) if pr.stim_pulse.meta is not None else 0
    #         if offset_distance is None: ## early photostimlogs didn't record the offset between the stimulation plane and the cell
    #             offset_distance = 0

    #         if in_limits1[0] <= holding < in_limits1[1]:
    #             qc_pass = qc[pr.in_qc_pass and offset_distance < distance_limit]
    #             sorted_responses[(clamp_mode, -55, power)][qc_pass].append(pr)

    #         elif in_limits2[0] <= holding < in_limits2[1]:
    #             qc_pass = qc[pr.in_qc_pass and offset_distance < distance_limit]
    #             sorted_responses[(clamp_mode, 0, power)][qc_pass].append(pr)

    #         elif ex_limits[0] <= holding < ex_limits[1]:
    #             qc_pass = qc[pr.ex_qc_pass and offset_distance < distance_limit]
    #             sorted_responses[(clamp_mode, -70, power)][qc_pass].append(pr)

    #     return sorted_responses


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

        # param.fit = result['fit']
        # param.event_times = result['event_times']
        # param.initial_params = result['initial_params']
        # param.fit_event_index = result['fit_event_index']
        # param.included_responses = result['included_responses']
        # param.excluded_responses = result['excluded_responses']
        param.result = result

    def set_responses(self):        
        for i, key in enumerate(self.sorted_responses.keys()):
            self.analyzers[key].set_responses(self.sorted_responses[key])


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

            session = notes_db.db.session(readonly=False)
            record = notes_db.get_pair_notes_record(expt_id, pre_cell_id, post_cell_id, session=session)
            meta = {} if record is None else record.notes.copy()

            new_meta = {
                'expt_id': self.pair.experiment.ext_id,
                'pre_cell_id': self.pair.pre_cell.ext_id,
                'post_cell_id': self.pair.post_cell.ext_id,
                'synapse_type': synapse_call,
                'gap_junction': self.pair_param['Gap junction call'],
                'comments': self.comment_param[''],
                'categories':{str(key):{} for key in self.sorted_responses.keys()}
            }

            for key in self.sorted_responses.keys():
                param = self.category_param.child(str(key))
                if param['status'] in ['done', 'previously analyzed']:
                    new_meta['categories'][str(key)].update(param.result)
                        # {
                        # 'included_responses':param.result.get('included_responses'),
                        # 'excluded_responses':param.result.get('excluded_responses')
                        # })
                    #if synapse_call is not None:
                        #new_meta['categories'][str(key)].update(
                            # {
                            # 'initial_parameters':param.initial_params,
                            # 'fit_parameters': param.fit,
                            # 'fit_pass': param['user_passed_fit'],
                            # 'n_events': param['number_of_events'],
                            # 'event_times':param.event_times,
                            # 'fit_event_index':param.fit_event_index})
                            

            meta.update(new_meta)

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
                if msg.result() == pg.QtGui.QDialog.Accepted:
                    record.notes = meta
                    record.modification_time = datetime.datetime.now()
                    session.commit() 
                else:
                    session.rollback()
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
        layout.setSpacing(3)
        self.setLayout(layout)

        h_splitter = pg.QtGui.QSplitter(pg.QtCore.Qt.Horizontal)
        h_splitter.setContentsMargins(0,0,0,0)
        layout.addWidget(h_splitter)

        self.responses = None ## holder for {pass:[tseries], fail:[tseries]}
        self.average_response = None ## holder for the average response
        self.avgPlotItem = None
        self.event_counter = 0
        self.colors = {'failed':(255,0,0,100),
                       'excluded':(255, 150, 0, 100), 
                       'included':(255,255,255,100), 
                       'selected':(0,255,0,255)}

        self.key = key
        self.mode = {-70:'excitatory', -55:'inhibitory', 0:'inhibitory'}[self.key[1]] if self.key is not None else None
        self.clamp_mode = self.key[0] if self.key is not None else None
        self.has_presynaptic_data = (key != None) and (key[2] == None)
        self.host = host
        self.event_param_tree = pg.parametertree.ParameterTree(showHeader=False)
        self.response_param_tree = pg.parametertree.ParameterTree(showHeader=False)
        self.plot_grid=PlotGrid()
        self.add_btn = pg.FeedbackButton("Add analysis")
        self.add_btn.clicked.connect(self.add_analysis_btn_clicked)

        v_widget = pg.QtGui.QWidget()
        v_layout = pg.QtGui.QVBoxLayout()

        self.tabWidget = pg.QtGui.QTabWidget()
        self.tabWidget.setContentsMargins(0,0,0,0)
        tab_layout = pg.QtGui.QGridLayout()
        tab_layout.setContentsMargins(0,0,0,0)
        self.tabWidget.setLayout(tab_layout)
        self.tabWidget.addTab(self.event_param_tree, "Events")
        self.tabWidget.addTab(self.response_param_tree, "Responses")
        
        v_layout.setSpacing(3)
        v_widget.setLayout(v_layout)
        v_layout.addWidget(self.tabWidget)
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
        self.event_param_tree.addParameters(self.event_params)

        self.response_param = pg.parametertree.Parameter.create(name="Responses", type='group')
        self.offset_param = pg.parametertree.Parameter.create(name="Plot w/ offset", type='bool', value=False)
        self.offset_param.sigValueChanged.connect(self.plot_responses)
        self.response_param_tree.addParameters(self.offset_param)
        self.response_param_tree.addParameters(self.response_param)
        self.response_param_tree.currentItemChanged.connect(self.responseParamSelectionChanged)

        try:
            self.event_params.sigAddNew.connect(self.add_event_param)
        except AttributeError:
            raise Exception('PairAnalysis requires Pyqtgraph 0.11.0 or greater. (current version is %s)'%str(pg.__version__))

    def load_saved_data(self, data):
        ### exclude traces that were excluded before 
        ###  - don't re-include traces if they've since been excluded 
        ###    -- but maybe we should indicate that the fit's no longer good?
        for param in self.response_param.children():
            if param.value():
                ext_id = str(param.pulse_response.ext_id)
                if ext_id in data['excluded_responses'].keys():
                    param.setValue(False)
                    param.child('exclusion reasons').setValue(data['excluded_responses'][ext_id])

        for time in data['event_times']:
            p = self.add_event_param()
            p.child('user_latency').setValue(time)

        for p in self.event_params.children():
            if p._should_have_fit:
                self.plot_fit(p, data['fit'])
                p._fit_values = data['fit']
                p._initial_fit_guesses = data['initial_params']
                self.update_fit_param_display(p, data['fit'])
                p.child('Fit passes qc').setValue(data['fit_pass'])

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
        param.child('Fit event').sigActivated.connect(self.fit_event_btn_clicked)
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

    def fit_event_btn_clicked(self, btn_param):
        event_param = btn_param.parent()
        i = int(event_param.name()[-1])
        latency = event_param['user_latency']

        latencies = [p['user_latency'] for p in self.event_params.children()]

        fit = fit_event_2p(self.average_response, self.clamp_mode, latencies, i)

        fit_vals = fit.values
        fit_vals.update(nrmse=fit.nrmse())

        ## store these here so we can save them later
        event_param._fit_values = fit_vals
        event_param._initial_fit_guesses = fit.opto_init_params

        ### update plot and param display
        self.plot_fit(event_param, fit_vals)
        self.update_fit_param_display(event_param, fit_vals)



    # def fit_event(self, btn_param):
    #     event_param = btn_param.parent()
    #     i = int(event_param.name()[-1])
    #     latency = event_param['user_latency']

    #     window = [latency - 0.0002, latency+0.0002]

    #     #### fit snippet of original data, but feed fitting algorithm good guesses for amp and rise time 
    #     ## basically make travis's measurments, then try to fit to get tau
    #     if i == len(self.event_params.children())-1: ## last event
    #         data = self.average_response.time_slice(latency-0.001, latency+0.05)
    #     else:
    #         for param in self.event_params.children():
    #             if int(param.name()[-1]) == i+1:
    #                 next_latency = param['user_latency']
    #                 break
    #         data = self.average_response.time_slice(latency-0.001, next_latency)

    #     filtered = filters.bessel_filter(data, 6000, order=4, btype='low', bidir=True)

    #     lat_index = filtered.index_at(latency)
    #     if max(abs(filtered.data[:lat_index])) > max(abs(filtered.data[lat_index:])): ## there is lots going on in the baseline
    #         ## cut it off
    #         filtered = filtered[lat_index-int(0.0002/filtered.dt):]
    #     #ev = filtered[lat_index:]
    #     peak_ind = np.argwhere(max(abs(filtered.data))==abs(filtered.data))[0][0]
    #     peak_time = filtered.time_at(peak_ind)
    #     rise_time = peak_time-(latency)
    #     amp = filtered.value_at(peak_time) - filtered.value_at(latency)
    #     init_params = {'rise_time':rise_time, 'amp':amp}
    #     fit = fit_psp(filtered, (window[0], window[1]), self.clamp_mode, sign=0, exp_baseline=False, init_params=init_params, fine_search_spacing=filtered.dt, fine_search_window_width=100e-6)

    #     fit_vals = fit.values
    #     fit_vals.update(nrmse=fit.nrmse())
    #     self.plot_fit(event_param, fit_vals, init_params)
    #     ## display fit params
    #     self.update_fit_param_display(event_param, fit_vals)


    def plot_fit(self, event_param, fit_values):
        ## plot fit
        v = fit_values
        y=Psp.psp_func(self.average_response.time_values,v['xoffset'], v['yoffset'], v['rise_time'], v['decay_tau'], v['amp'], v['rise_power'])

        if hasattr(event_param, '_fit_plot_item'):
            self.plot_grid[(0,0)].removeItem(event_param._fit_plot_item)
        event_param._fit_plot_item = self.plot_grid[(0,0)].plot(self.average_response.time_values, y, pen=event_param['display_color'])
        event_param._fit_plot_item.setZValue(20)

        

    def update_fit_param_display(self, param, v):
        if self.clamp_mode == 'vc':
            param.child('Fit results').child('amplitude').setValue(pg.siFormat(v['amp'], suffix='A'))
        elif self.clamp_mode == 'ic':
            param.child('Fit results').child('amplitude').setValue(pg.siFormat(v['amp'], suffix='V'))

        param.child('Fit results').child('latency').setValue(pg.siFormat(v['xoffset'], suffix='s'))
        param.child('Fit results').child('rise time').setValue(pg.siFormat(v['rise_time'], suffix='s'))
        param.child('Fit results').child('decay tau').setValue(pg.siFormat(v['decay_tau'], suffix='s'))

        nrmse=v.get('nrmse')
        if nrmse is None:
            param.child('Fit results').child('NRMSE').setValue('nan')
        else:
            param.child('Fit results').child('NRMSE').setValue('%0.2f'%nrmse)

    def clear_fit(self):
        for event_param in self.event_params.children():
            if event_param._should_have_fit:
                if hasattr(event_param, '_fit_plot_item'):
                    self.plot_grid[(0,0)].removeItem(event_param._fit_plot_item)
                for name in ['amplitude', 'latency', 'rise time', 'decay tau', 'NRMSE']:
                    event_param.child('Fit results').child(name).setValue('')
                event_param.child('Fit passes qc').setValue('')

    def set_responses(self, responses):
        """Supply this response analyzer with a list of pulse responses"""
        self.responses = responses
        global EXCLUSION_REASONS
        qc_check = {'inhibitory':'in_qc_pass', 'excitatory':'ex_qc_pass'}.get(self.mode)
        for pr in responses:
            expt_id, sweep_n, dev_name, pulse_n = pr.ext_id
            name="sweep %i: pulse %i" % (sweep_n, pulse_n)
            qc_pass = getattr(pr, qc_check)
            param = pg.parametertree.Parameter.create(name=name, type='bool', value=qc_pass, expanded=False, children=[
                {'name':'exclusion reasons', 'type':'list', 'value':'None', 'values':['None']+ EXCLUSION_REASONS}])
            if not qc_pass:
                param.child('exclusion reasons').setValue('Response failed automated qc')
                param.child('exclusion reasons').setOpts(readonly=True)
                param.setOpts(readonly=True)
            param.pulse_response = pr
            self.response_param.addChild(param)
            param.sigValueChanged.connect(self.response_inclusion_changed)

        ## align responses by spike or pulse - give each param a .aligned_post_tseries and .aligned_pre_tseries
        for param in self.response_param.children():
            self.align_response(param)

        self.plot_responses()

    def align_response(self, param):
        pr = param.pulse_response

        for name in ['pre', 'post']:
            ts = getattr(pr, name+'_tseries')
            if ts is None:
                setattr(param, 'aligned_'+name+'_tseries', None)
                continue
            stim_time = pr.stim_pulse.onset_time

            ## do baseline subtraction
            start_time = max(ts.t0, stim_time-5e-3)
            baseline_data = ts.time_slice(start_time, stim_time).data
            if len(baseline_data) == 0:
                baseline = ts.data[0]
            else:
                baseline = float_mode(baseline_data)
            ts = ts - baseline

            ## align to pulse or spike
            if self.has_presynaptic_data:
                stim_time = pr.stim_pulse.first_spike_time
                if stim_time is None:
                    setattr(param, 'aligned_'+name+'_tseries', None)
                    with pg.SignalBlock(param.sigValueChanged, self.response_inclusion_changed):
                        param.setValue(False)
                        param.setOpts(readonly=True)
                        param.child('exclusion reasons').setValue('Unable to determine spike time')
                        param.child('exclusion reasons').setOpts(readonly=True)
                        #raise Exception('stop')
                    continue
            ts = ts.copy(t0=ts.t0 - stim_time)
            setattr(param, 'aligned_'+name+'_tseries', ts)

    def response_inclusion_changed(self):
        self.clear_fit()
        self.plot_responses()

    def plot_responses(self):
        for param in self.response_param.children():
            if hasattr(param, 'plotDataItem'):
                self.plot_grid[(0,0)].removeItem(param.plotDataItem)
        self.plot_grid[(1,0)].clear() 
        if self.avgPlotItem is not None:
            self.plot_grid[(0,0)].removeItem(self.avgPlotItem)       

        included = []
        if self.offset_param.value():
            if self.clamp_mode == 'ic':
                step = 1e-3
            else:
                step = 15e-12
            offset = np.flip(np.linspace(step, step*len(self.response_param.children()), len(self.response_param.children())), axis=0)
        else:
            offset = np.zeros(len(self.response_param.children()))



        for i, param in enumerate(self.response_param.children()):
            if param.value():
                included.append(param)

            if param.aligned_post_tseries is not None:
                item = self.plot_grid[(0,0)].plot(param.aligned_post_tseries.time_values, param.aligned_post_tseries.data+offset[i])
                item.curve.setClickable(True)
                param.plotDataItem = item
                item.sigClicked.connect(self.traceClicked)
            if param.aligned_pre_tseries is not None:
                item = self.plot_grid[(1,0)].plot(param.aligned_pre_tseries.time_values, param.aligned_pre_tseries.data)
                item.curve.setClickable(True)
                param.spikePlotDataItem = item
                item.sigClicked.connect(self.traceClicked)

        post_ts = TSeriesList(map(lambda x: x.aligned_post_tseries, included))
        self.average_response = post_ts.mean()
        self.avgPlotItem = self.plot_grid[(0,0)].plot(self.average_response.time_values, self.average_response.data, pen={'color': 'b', 'width': 2})
        self.avgPlotItem.setZValue(15)

        self.plot_grid[(0,0)].autoRange()
        self.plot_grid[(0,0)].setLabel('bottom', text='Time from stimulus', units='s')
        self.plot_grid[(0,0)].setLabel('left', units={'ic':'V', 'vc':'A'}.get(self.clamp_mode))

        if self.has_presynaptic_data:
            self.plot_grid[(1,0)].autoRange()
            self.plot_grid[(1,0)].setLabel('bottom', units='s')
            self.plot_grid[(1,0)].setLabel('left', units='V')

        self.recolorTraces(None)


    def traceClicked(self, item):
        self.recolorTraces(item)
        for param in self.response_param.children():
            if not hasattr(param, 'plotDataItem'):
                continue
            if (param.plotDataItem == item) or (getattr(param, 'spikePlotDataItem', None)==item):
                for paramItem in param.items.keys():
                    with pg.SignalBlock(paramItem.treeWidget().currentItemChanged, self.responseParamSelectionChanged):
                        paramItem.treeWidget().setCurrentItem(paramItem)


    def responseParamSelectionChanged(self):
        param = self.response_param_tree.currentItem().param
        traceItem = getattr(param, 'plotDataItem', None) 
        self.recolorTraces(traceItem)


    def recolorTraces(self, item):
        for param in self.response_param.children():
            if param.aligned_post_tseries is None: ## means we don't have a trace that we can align or plot
                continue
            if (param.plotDataItem is item) or (getattr(param, 'spikePlotDataItem', 'null_place_holder') is item): ## need to default to a different thing than None so that none is not passed to recolorTraces
                param.plotDataItem.setPen(self.colors['selected'])
                param.plotDataItem.setZValue(10)
                if hasattr(param, 'spikePlotDataItem'):
                    param.spikePlotDataItem.setPen(self.colors['selected'])
                    param.spikePlotDataItem.setZValue(10)
            elif param.value():
                param.plotDataItem.setPen(self.colors['included'])
                if hasattr(param, 'spikePlotDataItem'):
                    param.spikePlotDataItem.setPen(self.colors['included'])         
            elif not param.opts['readonly']:
                param.plotDataItem.setPen(self.colors['excluded'])
                if hasattr(param, 'spikePlotDataItem'):
                    param.spikePlotDataItem.setPen(self.colors['excluded'])
            else:
                param.plotDataItem.setPen(self.colors['failed'])
                if hasattr(param, 'spikePlotDataItem'):
                    param.spikePlotDataItem.setPen(self.colors['failed'])

            if not ((param.plotDataItem is item) or (getattr(param, 'spikePlotDataItem', 'null_place_holder') is item)):
                param.plotDataItem.setZValue(0)
                if hasattr(param, 'spikePlotDataItem'):
                    param.spikePlotDataItem.setZValue(0)

    def get_response_lists(self):
        """Return a tuple with a list of ext_ids of included responses and a 
        dict of {response_ext_id:[exclusion_reasons]} for excluded responses."""

        included = []
        excluded = {}
        for param in self.response_param.children():
            if param.value():
                included.append(str(param.pulse_response.ext_id))
            else:
                excluded[str(param.pulse_response.ext_id)]=param.child('exclusion reasons').value()

        return (included, excluded)

    def add_analysis_btn_clicked(self):
        try:

            fit = None
            fit_pass = None
            initial_params=None
            event_index=None
            evs = [] ## sanity check that we only have one event fit
            for i, p in enumerate(self.event_params.children()):
                if p._should_have_fit:
                    fit_pass = p['Fit passes qc']
                    if fit_pass is None:
                        raise Exception('Please specify whether fit passes qc for %s'%p.name())

                    evs.append(p.name())
                    fit = p._fit_values
                    initial_params=p._initial_fit_guesses
                    event_index = i
                    
            if len(evs) > 1:
                ### need to figure out why this is happening
                raise Exception('Error: More than one fit found. This is a bug')

            included, excluded = self.get_response_lists()

            res = {'category_name':self.key,
                   'fit': fit,
                   'fit_pass':fit_pass,
                   'initial_params':initial_params,
                   'n_events':len(self.event_params.children()),
                   'event_times':[p['user_latency'] for p in self.event_params.children()],
                   'fit_event_index':event_index,
                   'included_responses':included,
                   'excluded_responses':excluded}
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














        