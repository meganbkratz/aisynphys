import pyqtgraph as pg
from aisynphys.database import default_db as db

class ExperimentSelector(pg.QtGui.QWidget):
    """A widget for selecting experiments from a synphys database. 
    Emits 'sigNewExperimentsRetrieved' with a list of experiments."""

    sigNewExperimentsRetrieved = pg.QtCore.Signal(object) # a list of db experiments

    def __init__(self, db_session, notes_session, hashtags=None):
        pg.QtGui.QWidget.__init__(self)
        self.db_session = db_session
        self.notes_db_session = notes_session

        layout = pg.QtGui.QVBoxLayout()
        layout.setContentsMargins(3,3,3,3)
        self.setLayout(layout)

        self.select_ptree = pg.parametertree.ParameterTree(showHeader=False)
        if hashtags is None:
            hashtags = ['']
        self.hash_select = pg.parametertree.Parameter.create(name='Hashtags', type='group', children=
            [{'name': 'With multiple selected:', 'type': 'list', 'values': ['Include if any appear', 'Include if all appear'], 'value': 'Include if any appear'}]+
            [{'name': '#', 'type': 'bool'}] +
            [{'name': ht, 'type': 'bool'} for ht in hashtags[1:]])

        self.rigs = db.query(db.Experiment.rig_name).distinct().all()
        self.operators = db.query(db.Experiment.operator_name).distinct().all()

        self.rig_select = pg.parametertree.Parameter.create(name='Rig', type='group', children=[{'name': str(rig[0]), 'type': 'bool'} for rig in self.rigs])
        self.operator_select = pg.parametertree.Parameter.create(name='Operator', type='group', children=[{'name': str(operator[0]), 'type': 'bool'} for operator in self.operators])
        self.data_type = pg.parametertree.Parameter.create(name='Reduce data to:', type='group', children=[
            {'name': 'Pairs with data', 'type': 'bool', 'value': True},
            {'name': 'Synapse is None', 'type': 'bool'}])

        [self.select_ptree.addParameters(param) for param in [self.data_type, self.rig_select, self.operator_select, self.hash_select]]

        layout.addWidget(self.select_ptree)

        self.getExptBtn = pg.QtGui.QPushButton("Get Experiments")

        layout.addWidget(self.getExptBtn)
        layout.setStretch(0, 10)

        self.getExptBtn.clicked.connect(self.get_experiments)

    def set_callback(self, fn):
        self.callback_fn = fn

    def get_experiments(self):
        print('db:', db)
        print('self.db_session', self.db_session)
        prof = pg.debug.Profiler('ExptSelector.get_experiments', disabled=False)
        expt_query = self.db_session.query(db.Experiment)
        prof.mark('expt_query 1')
        synapse_none = self.data_type['Synapse is None']
        if synapse_none:
            subquery = db.query(db.Pair.experiment_id).filter(db.Pair.has_synapse==None).subquery()
            prof.mark('query 2')
            expt_query = expt_query.filter(db.Experiment.id.in_(subquery))
            prof.mark('filter 3')
        selected_rigs = [rig.name() for rig in self.rig_select.children() if rig.value() is True]
        if len(selected_rigs) != 0:
            expt_query = expt_query.filter(db.Experiment.rig_name.in_(selected_rigs))
            prof.mark('filter 4')
        selected_operators = [operator.name() for operator in self.operator_select.children() if operator.value()is True]
        if len(selected_operators) != 0:
            expt_query = expt_query.filter(db.Experiment.operator_name.in_(selected_operators))
            prof.mark('filter 5')
        selected_hashtags = [ht.name() for ht in self.hash_select.children()[1:] if ht.value() is True]
        if len(selected_hashtags) != 0:
            timestamps = self.get_expts_hashtag(selected_hashtags)
            prof.mark('hashtags 6')
            expt_query = expt_query.filter(db.Experiment.ext_id.in_(timestamps))
            prof.mark('filter 7')
        prof.mark('pre all 7b')
        expts = expt_query.all()
        prof.mark('all 8')
        #print(expts[5])
        #import cProfile
        #cProfile.runctx('self.sigNewExperimentsRetrieved.emit(expts)', globals(), {'self':self, 'expts':expts}, filename='sigProfile.txt')
        self.sigNewExperimentsRetrieved.emit(expts)
        #self.callback_fn(expts)
        prof.mark('emit signal 9')

    def get_expts_hashtag(self, selected_hashtags):
        q = self.notes_db_session.query(notes_db.PairNotes)
        pairs_to_include = []
        note_pairs = q.all()
        note_pairs.sort(key=lambda p: p.expt_id)
        for p in note_pairs:
            comments = p.notes.get('comments')
            if comments is None:
                continue    
            if len(selected_hashtags) == 1:
                hashtag = selected_hashtags[0]
                if hashtag == '#':
                    if hashtag in comments and all([ht not in comments for ht in comment_hashtag[1:]]):
                        print(p.expt_id, p.pre_cell_id, p.post_cell_id, comments)
                        pairs_to_include.append(p)
                else:
                    if hashtag in comments:
                        print(p.expt_id, p.pre_cell_id, p.post_cell_id, comments)
                        pairs_to_include.append(p)
                
            if len(selected_hashtags) > 1:
                hashtag_present = [ht in comments for ht in selected_hashtags]
                or_expts = self.hash_select['With multiple selected:'] == 'Include if any appear'
                and_expts = self.hash_select['With multiple selected:'] == 'Include if all appear'
                if or_expts and any(hashtag_present):
                    print(p.expt_id, p.pre_cell_id, p.post_cell_id, comments)
                    pairs_to_include.append(p)
                if and_expts and all(hashtag_present):
                    print(p.expt_id, p.pre_cell_id, p.post_cell_id, comments)
                    pairs_to_include.append(p)

        return set([pair.expt_id for pair in pairs_to_include])


