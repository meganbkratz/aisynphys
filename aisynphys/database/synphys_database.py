import datetime
from sqlalchemy.orm import aliased, contains_eager, selectinload
from collections import OrderedDict
from .database import Database
from .schema import schema_version, default_sample_rate
from ..synphys_cache import get_db_path, list_db_versions


class SynphysDatabase(Database):
    """Augments the Database class with convenience methods for querying the synphys database.
    """
    default_sample_rate = default_sample_rate
    schema_version = schema_version

    mouse_projects = ["mouse V1 coarse matrix", "mouse V1 pre-production"]
    human_projects = ["human coarse matrix"]


    @classmethod
    def load_sqlite(cls, sqlite_file, readonly=True):
        """Return a SynphysDatabase instance connected to an existing sqlite file.
        """
        ro_host = 'sqlite:///'
        rw_host = None if readonly else ro_host
        return cls(ro_host, rw_host, db_name=sqlite_file)
    
    @classmethod
    def list_versions(cls):
        """Return a list of the available database versions.
        """
        return list(list_db_versions().keys())
    
    @classmethod
    def load_version(cls, db_version):
        """Load a named database version.
        
        Available database names can be listed using :func:`list_versions`.
        The database file will be downloaded and cached, if an existing cache file is not found.

        Example::

            >>> from aisynphys.database import SynphysDatabase
            >>> SynphysDatabase.list_versions()
            ['synphys_r1.0_2019-08-29_small.sqlite', 'synphys_r1.0_2019-08-29_medium.sqlite', 'synphys_r1.0_2019-08-29_full.sqlite']
            >>> db = SynphysDatabase.load_version('synphys_r1.0_2019-08-29_small.sqlite')
            Downloading http://api.brain-map.org/api/v2/well_known_file_download/937779595 =>
              /home/luke/docs/aisynphys/doc/cache/database/synphys_r1.0_2019-08-29_small.sqlite
              [####################]  100.00% (73.13 MB / 73.1 MB)  4.040 MB/s  0:00:00 remaining
              done.

        """
        db_file = get_db_path(db_version)
        db = SynphysDatabase.load_sqlite(db_file, readonly=False)
        # instantiate any new tables that have been added to the schema but don't exist in the db file
        db.create_tables()
        return db

    def __init__(self, ro_host, rw_host, db_name):
        from .schema import ORMBase
        Database.__init__(self, ro_host, rw_host, db_name, ORMBase)
        
    def create_tables(self, tables=None):
        """This method is used when initializing a new database or new tables within an existing database.

        Extends :func:`Database.create_tables` to include an extra `Metadata` table.
        """
        Database.create_tables(self, tables=tables)
        
        # initialize or verify db version
        mrec = self.metadata_record()
        if mrec is None:
            mrec = self.Metadata(meta={
                'db_version': schema_version,
                'creation_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'origin': "Allen Institute for Brain Science / Synaptic Physiology",
            })
            s = self.session(readonly=False)
            s.add(mrec)
            s.commit()
        else:
            ver = mrec.meta['db_version']
            assert ver == schema_version, "Database has unsupported schema version %s (expected %s)"%(ver, schema_version)
    
    @property
    def metadata(self):
        return self.metadata_record.meta.copy()
        
    def metadata_record(self, session=None):
        session = session or self.default_session
        recs = session.query(self.Metadata).all()
        if len(recs) == 0:
            return None
        elif len(recs) > 1:
            raise Exception("Multiple metadata records found.")
        return recs[0]
        
    def slice_from_timestamp(self, ts, session=None):
        session = session or self.default_session
        slices = session.query(self.Slice).filter(self.Slice.acq_timestamp==ts).all()
        if len(slices) == 0:
            raise KeyError("No slice found for timestamp %0.3f" % ts)
        elif len(slices) > 1:
            raise KeyError("Multiple slices found for timestamp %0.3f" % ts)
        
        return slices[0]

    def experiment_from_timestamp(self, ts, session=None):
        session = session or self.default_session
        expts = session.query(self.Experiment).filter(self.Experiment.acq_timestamp==ts).all()
        if len(expts) == 0:
            # For backward compatibility, check for timestamp truncated to 2 decimal places
            for expt in session.query(self.Experiment).all():
                if abs((expt.acq_timestamp - ts)) < 0.01:
                    return expt
            
            raise KeyError("No experiment found for timestamp %0.3f" % ts)
        elif len(expts) > 1:
            raise RuntimeError("Multiple experiments found for timestamp %0.3f" % ts)
        
        return expts[0]

    def experiment_from_ext_id(self, ext_id, session=None):
        session = session or self.default_session
        expts = session.query(self.Experiment).filter(self.Experiment.ext_id==ext_id).all()
        if len(expts) == 0:
            raise KeyError('No experiment found for ext_id %s' %ext_id)
        elif len(expts) > 1:
            raise RuntimeError("Multiple experiments found for ext_id %s" %ext_id)

        return expts[0]

    def slice_from_ext_id(self, ext_id, session=None):
        session = session or self.default_session
        slices = session.query(self.Slice).filter(self.Slice.ext_id==ext_id).all()
        if len(slices) == 0:
            raise KeyError("No slice found for ext_id %s" % ext_id)
        elif len(slices) > 1:
            raise KeyError("Multiple slices found for ext_id %s" % ext_id)
        
        return slices[0]

    def list_experiments(self, session=None):
        session = session or self.default_session
        return session.query(self.Experiment).all()

    def pair_query(self, pre_class=None, post_class=None, synapse=None, synapse_type=None, electrical=None, 
                   project_name=None, acsf=None, age=None, species=None, distance=None, internal=None, 
                   preload=(), session=None):
        """Generate a query for selecting pairs from the database.

        Parameters
        ----------
        pre_class : :class:`aisynphys.cell_class.CellClass` | None
            Filter for pairs where the presynaptic cell belongs to this class
        post_class : :class:`aisynphys.cell_class.CellClass` | None
            Filter for pairs where the postsynaptic cell belongs to this class
        synapse : bool | None
            Include only pairs that are (or are not) connected by a chemical synapse
        synapse_type : str | None
            Include only synapses of a particular type ('ex' or 'in')
        electrical : bool | None
            Include only pairs that are (or are not) connected by an electrical synapse (gap junction)
        project_name : str | list | None
            Value(s) to match from experiment.project_name (e.g. "mouse V1 coarse matrix" or "human coarse matrix")
        acsf : str | list | None
            Filter for ACSF recipe name(s)
        age : tuple | None
            (min, max) age ranges to filter for. Either limit may be None to disable
            that check.
        species : str | None
            Species ('mouse' or 'human') to filter for
        distance : tuple | None
            (min, max) intersomatic distance in meters
        internal : str | list | None
            Electrode internal solution recipe name(s)
        preload : list
            List of strings specifying resources to preload along with the queried pairs. 
            This can speed up performance in cases where these would otherwise be 
            individually queried later on. Options are "cell" (includes cell, morphology, and patch_seq),
            "synapse" (includes synapse, resting_statem dynamics, and synapse_prediction).
        """
        session = session or self.default_session
        pre_cell = aliased(self.Cell, name='pre_cell')
        post_cell = aliased(self.Cell, name='post_cell')
        pre_morphology = aliased(self.Morphology, name='pre_morphology')
        post_morphology = aliased(self.Morphology, name='post_morphology')
        pre_patch_seq = aliased(self.PatchSeq, name='pre_patch_seq')
        post_patch_seq = aliased(self.PatchSeq, name='post_patch_seq')
        query = session.query(
            self.Pair,
        )
        query = query.join(pre_cell, pre_cell.id==self.Pair.pre_cell_id)
        query = query.join(post_cell, post_cell.id==self.Pair.post_cell_id)
        query = query.outerjoin(pre_morphology, pre_morphology.cell_id==pre_cell.id)
        query = query.outerjoin(post_morphology, post_morphology.cell_id==post_cell.id)
        query = query.outerjoin(pre_patch_seq, pre_patch_seq.cell_id==pre_cell.id)
        query = query.outerjoin(post_patch_seq, post_patch_seq.cell_id==post_cell.id)
        query = query.join(self.Experiment, self.Pair.experiment_id==self.Experiment.id)
        query = query.outerjoin(self.Slice, self.Experiment.slice_id==self.Slice.id) ## don't want to drop all pairs if we don't have slice or connection strength entries
        query = query.outerjoin(self.SynapsePrediction)
        query = query.outerjoin(self.Synapse)
        query = query.outerjoin(self.Dynamics)

        if pre_class is not None:
            query = pre_class.filter_query(query, pre_cell, db=self)

        if post_class is not None:
            query = post_class.filter_query(query, post_cell, db=self)

        if synapse is not None:
            query = query.filter(self.Pair.has_synapse==synapse)

        if synapse_type is not None:
            query = query.filter(self.Synapse.synapse_type==synapse_type)

        if electrical is not None:
            query = query.filter(self.Pair.has_electrical==electrical)

        if project_name is not None:
            if isinstance(project_name, str):
                query = query.filter(self.Experiment.project_name==project_name)
            else:
                query = query.filter(self.Experiment.project_name.in_(project_name))

        if acsf is not None:
            if isinstance(acsf, str):
                query = query.filter(self.Experiment.acsf==acsf)
            else:
                query = query.filter(self.Experiment.acsf.in_(acsf))

        if age is not None:
            if age[0] is not None:
                query = query.filter(self.Slice.age>=age[0])
            if age[1] is not None:
                query = query.filter(self.Slice.age<=age[1])

        if distance is not None:
            if distance[0] is not None:
                query = query.filter(self.Pair.distance>=distance[0])
            if distance[1] is not None:
                query = query.filter(self.Pair.distance<=distance[1])

        if species is not None:
            query = query.filter(self.Slice.species==species)

        if internal is not None:
            if isinstance(internal, str):
                query = query.filter(self.Experiment.internal==internal)
            else:
                query = query.filter(self.Experiment.internal.in_(internal))
        
        if 'cell' in preload:
            query = query.add_entity(pre_cell)
            query = query.add_entity(post_cell)
            query = query.add_entity(pre_morphology)
            query = query.add_entity(post_morphology)
            query = query.add_entity(pre_patch_seq)
            query = query.add_entity(post_patch_seq)
            query = query.options(
                contains_eager(self.Pair.pre_cell, alias=pre_cell), 
                contains_eager(self.Pair.post_cell, alias=post_cell), 
                contains_eager(pre_cell.morphology, alias=pre_morphology), 
                contains_eager(post_cell.morphology, alias=post_morphology), 
                contains_eager(pre_cell.patch_seq, alias=pre_patch_seq), 
                contains_eager(post_cell.patch_seq, alias=post_patch_seq), 
            )

        if 'synapse' in preload:
            query = query.add_entity(self.Synapse)
            query = query.options(
                contains_eager(self.Pair.synapse),
                selectinload(self.Pair.resting_state_fit), 
                selectinload(self.Pair.dynamics), 
                selectinload(self.Pair.synapse_prediction), 
            )

        # package the aliased cells
        query.pre_cell = pre_cell
        query.post_cell = post_cell

        return query

    def matrix_pair_query(self, pre_classes, post_classes, columns=None, pair_query_args=None):
        """Returns the concatenated result of running pair_query over every combination
        of presynaptic and postsynaptic cell class.
        """
        if pair_query_args is None:
            pair_query_args = {}

        pairs = None
        for pre_name, pre_class in pre_classes.items():
            for post_name, post_class in post_classes.items():
                pair_query = self.pair_query(
                    pre_class=pre_class,
                    post_class=post_class,
                    **pair_query_args
                )
                
                if columns is not None:
                    pair_query = pair_query.add_columns(*columns)
                
                df = pair_query.dataframe()
                df['pre_class'] = pre_name
                df['post_class'] = post_name
                if pairs is None:
                    pairs = df
                else:
                    pairs = pairs.append(df)
        
        return pairs

    def __getstate__(self):
        """Allows DB to be pickled and passed to subprocesses.
        """
        return {
            'ro_host': self.ro_host, 
            'rw_host': self.rw_host, 
            'db_name': self.db_name,
        }

    def __setstate__(self, state):
        self.__init__(ro_host=state['ro_host'], rw_host=state['rw_host'], db_name=state['db_name'])
