import os, sys, glob, traceback, time
import sync_rigs_to_server as mp_sync
from aisynphys import config

def find_all_days(root):
    d = '[0-9]'
    sep = '[.,_,-]'
    days = glob.glob(os.path.join(root, d*4+sep+d*2+sep+d*2+'*'))
    days.sort(reverse=True)
    days = list(filter(os.path.isdir, days))

    all_dirs = set(filter(os.path.isdir, glob.glob(os.path.join(root, '*'))))
    skipped = all_dirs - set(days)
    return days, list(skipped)

def get_server_path(day_dir):
    """Given a directory handle to an experiment storage folder on a rig,
    return the path to the corresponding location on the server, relative
    to the server's root storage path.
    server_storage_path/rig_name/computer/day
    """
    p, day = os.path.split(day_dir)
    p = os.path.abspath(p)
    for rig, machines in config.rig_data_paths.items():
        for mach in machines:
            if os.path.abspath(mach['primary']) == p:
                return os.path.join(rig, mach['name'], day)

    raise Exception("couldn't find server path for %s"%day_dir)



def sync_day_folders(paths):
    log = []
    changed_paths = []
    for day_dir in paths:
        try:
            changes = sync_day(day_dir)
            if len(changes) > 0:
                log.append((day_dir, changes))
                changed_paths.append(day_dir)
        except Exception:
            exc = traceback.format_exc()
            print(exc)
            log.append((day_dir, [], exc, []))
    return log, changed_paths

def sync_day(day_dir):
    """Synchronize all files for an experiment day on a single machine to the 
    server. If files are split between two machines (for example one computer 
    running acq4 and one running PrairieView) this should be called once for 
    each machine.

    Argument must be the path of an experiment day folder. All files within the
    day folder will be recursively copied. 

    Return a list of changes made.
    """
    #site_dh = getDirHandle(site_dir)
    changes = []
    #slice_dh = site_dh.parent()
    #expt_dh = slice_dh.parent()
    
    now = time.strftime('%Y-%m-%d_%H:%M:%S')
    mp_sync.log("========== %s : Sync %s to server" % (now, day_dir))
    skipped = 0
    
    try:
        # Decide how the top-level directory will be named on the remote server
        # (it may already be there from a previous slice/site, or the current
        # name may already be taken by another rig.)

        ## real:
        server_expt_path = os.path.join(config.synphys_data, get_server_path(day_dir))
        
        mp_sync.log("    using server path: %s" % server_expt_path)
        skipped += mp_sync._sync_paths_recursive(day_dir, server_expt_path, changes)
        
        # # Copy slice files if needed
        # server_slice_path = os.path.join(server_expt_path, slice_dh.shortName())
        # skipped += _sync_paths(slice_dh.name(), server_slice_path, changes)

        # # Copy site files if needed
        # server_site_path = os.path.join(server_slice_path, site_dh.shortName())
        # skipped += _sync_paths(site_dh.name(), server_site_path, changes)
        
        mp_sync.log("    Done; skipped %d files." % skipped)
        
    except Exception:
        err = traceback.format_exc()
        changes.append(('error', day_dir, err))
        mp_sync.log(err)

    return changes

def sync_all(source='archive'):
    """Synchronize all known rig data paths to the server

    *source* should be either 'primary' or 'archive', referring to the paths
    specified in config.rig_data_paths.
    """
    log = []
    synced_paths = []
    # Loop over all rigs
    for rig_name, data_paths in config.rig_data_paths.items():
        # Each rig may have multiple paths to check
        for data_path in data_paths:
            data_path = data_path[source]

            # Get a list of all experiments stored in this path
            paths, skipped_dirs = find_all_days(data_path)
            print("Found %i day directorys. Skipping:%s" %(len(paths), skipped_dirs))

            # synchronize files for each experiment to the server
            new_log, changed_paths = sync_day_folders(paths)
            log.extend(new_log)
            synced_paths.append((rig_name, data_path, len(changed_paths), len(paths)))
    
    return log, synced_paths

if __name__ == '__main__':
    
    paths = sys.argv[1:]
    if len(paths) == 0:
        # Synchronize all known rig data paths
        log, synced_paths = sync_all(source='primary')
        print("==========================\nSynchronized files from:")
        for rig_name, data_path, n_expts_changed, n_expts_found in synced_paths:
            print("%s  :  %s  (%d/%d expts updated)" % (rig_name, data_path, n_expts_changed, n_expts_found))

    else:
        # synchronize just the specified path(s)
        log = sync_experiments(paths)
    
    errs = [change for site in log for change in site[1] if change[0] == 'error']
    print("\n----- DONE ------\n   %d errors" % len(errs))
    
    for err in errs:
        print(err[1], err[2])