import pyqtgraph as pg
from aisynphys.database import default_db as db
from sqlalchemy.orm import joinedload

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

    def get_experiments(self):
        expt_query = self.db_session.query(db.Experiment)
        synapse_none = self.data_type['Synapse is None']
        if synapse_none:
            subquery = db.query(db.Pair.experiment_id).filter(db.Pair.has_synapse==None).subquery()
            expt_query = expt_query.filter(db.Experiment.id.in_(subquery))
        selected_rigs = [rig.name() for rig in self.rig_select.children() if rig.value() is True]
        if len(selected_rigs) != 0:
            expt_query = expt_query.filter(db.Experiment.rig_name.in_(selected_rigs))
        selected_operators = [operator.name() for operator in self.operator_select.children() if operator.value()is True]
        if len(selected_operators) != 0:
            expt_query = expt_query.filter(db.Experiment.operator_name.in_(selected_operators))
        selected_hashtags = [ht.name() for ht in self.hash_select.children()[1:] if ht.value() is True]
        if len(selected_hashtags) != 0:
            timestamps = self.get_expts_hashtag(selected_hashtags)
            expt_query = expt_query.filter(db.Experiment.ext_id.in_(timestamps))

        expts = expt_query.options(joinedload(db.Experiment.cell_list)).all()
        self.sigNewExperimentsRetrieved.emit(expts)

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


