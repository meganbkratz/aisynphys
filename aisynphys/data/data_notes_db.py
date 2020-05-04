"""
A database holding results of manual analyses
"""

from aisynphys.database.database import declarative_base, make_table
from aisynphys.database import Database, NoDatabase
from aisynphys import config

DataNotesORMBase = declarative_base()


PairNotes = make_table(
    name='pair_notes',
    comment="Manually verified fits to average synaptic responses",
    columns=[
        ('expt_id', 'str', 'Unique experiment identifier (acq_timestamp)', {'index': True}),
        ('pre_cell_id', 'str', 'external id of presynaptic cell'),
        ('post_cell_id', 'str', 'external id of postsynaptic cell'),
        ('notes','object', 'pair data dict which includes synapse call, initial fit parameters, output fit parameters, comments, etc'),
        ('modification_time', 'datetime', 'Last modification time for each record.'),
    ],
    ormbase=DataNotesORMBase,
)

if config.synphys_db_host is None:
    db = NoDatabase("Cannot access data_notes; no DB specified in config.synphys_db_host")
else:
    if not hasattr(config, 'notes_db_name'):
        name = "data_notes"
        print('Please add "notes_db_name:"data_notes"" (or whatever you want the notes_db name to be) to aisynphys/config.yml')
    else:
        name = config.notes_db_name
    db = Database(config.synphys_db_host, config.synphys_db_host_rw, name, DataNotesORMBase)


def get_pair_notes_record(expt_id, pre_cell_id, post_cell_id, session=None):
    if session is None:
        session = db.default_session

    q = session.query(PairNotes)
    q = q.filter(PairNotes.expt_id==expt_id)
    q = q.filter(PairNotes.pre_cell_id==pre_cell_id)
    q = q.filter(PairNotes.post_cell_id==post_cell_id)
    
    recs = q.all()
    if len(recs) == 0:
        return None
    elif len(recs) > 1:
        raise Exception("Multiple records found in pair_notes for pair %s %s %s!" % (expt_id, pre_cell_id, post_cell_id))
    return recs[0]
