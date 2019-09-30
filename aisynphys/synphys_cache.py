import os, sys, glob
from collections import OrderedDict
from . import config
from .util import sync_file, dir_timestamp, interactive_download


_cache = None
def get_cache():
    global _cache
    if _cache is None:
        _cache = SynPhysCache()
    return _cache


class SynPhysCache(object):
    """Maintains a local cache of files from the synphys raw data repository.
    """
    def __init__(self, local_path=config.cache_path, remote_path=config.synphys_data):
        self._pip_yamls = None
        self._nwbs = None
        self._expts = None
        
        # If a relative path is given, then interpret it as relative to home
        if not os.path.isabs(local_path):
            local_path = os.path.join(os.path.dirname(__file__), '..', local_path)

        self.local_path = None if local_path is None else os.path.abspath(local_path)
        self.remote_path = os.path.abspath(remote_path)
        
    def list_experiments(self):
        if self._expts is None:
            yamls = self.list_pip_yamls()
            site_dirs = sorted([os.path.dirname(yml) for yml in yamls], reverse=True)
            self._expts = OrderedDict([('%0.3f'%dir_timestamp(site_dir), site_dir) for site_dir in site_dirs])
        return self._expts

    def list_nwbs(self):
        if self._nwbs is None:
            self._nwbs = glob.glob(os.path.join(self.remote_path, '*', 'slice_*', 'site_*', '*.nwb'))
        return self._nwbs
    
    def list_pip_yamls(self):
        if self._pip_yamls is None:
            self._pip_yamls = glob.glob(os.path.join(self.remote_path, '*', 'slice_*', 'site_*', 'pipettes.yml'))
        return self._pip_yamls

    def get_cache(self, filename):
        if self.local_path is None:
            return os.path.join(self.remote_path, filename)
        
        filename = os.path.abspath(filename)
        if not filename.startswith(self.remote_path):
            raise Exception("Requested file %s is not inside %s" % (filename, self.remote_path))

        rel_filename = filename[len(self.remote_path):].lstrip(os.sep)
        path, _ = os.path.split(rel_filename)
        
        local_path = os.path.join(self.local_path, path)
        local_filename = os.path.join(self.local_path, rel_filename)
        
        if config.grow_cache:
            self._mkdir(local_path)
            sync_file(filename, local_filename)        
        
        if os.path.exists(local_filename):
            return local_filename
        else:
            return filename
        
    def _mkdir(self, path):
        if not os.path.isdir(path):
            root, _ = os.path.split(path)
            if root != '':
                self._mkdir(root)
            os.mkdir(path)


def list_db_versions():
    """Return a dictionary listing database versions that are available for download.
    """
    return {
        'synphys_r1.0_2019-08-29_small.sqlite': {'url': 'http://iwarehouse/api/v2/well_known_file_download/937779595'},
        'synphys_r1.0_2019-08-29_medium.sqlite': {'url': 'http://iwarehouse/api/v2/well_known_file_download/937780246'},
        'synphys_r1.0_2019-08-29_full.sqlite': {'url': 'http://iwarehouse/api/v2/well_known_file_download/937780286'},
    }


def get_db_path(db_version):
    """Return the filesystem path of a known database file.
    
    If the file does not exist locally, then it will be downloaded before returning
    the path.
    """
    cache_path = os.path.join(config.cache_path, 'database')
    cache_file = os.path.join(cache_path, db_version)
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        versions = list_db_versions()
        if db_version not in versions:
            raise KeyError("Unknown database version; options are: %s" % str(list(versions.keys())))
        url = versions[db_version]['url']
        interactive_download(url, cache_file)
        
    return cache_file
