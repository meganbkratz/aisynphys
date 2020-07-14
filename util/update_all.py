"""
Runs all pipeline analysis stages on a daily schedule. 

Why not use cron like a normal human being?
Because I like having the script run in the console where I cam monitor and
debug problems more easily.
"""

import os, sys, time, argparse
from datetime import datetime, timedelta
from collections import OrderedDict


def delay(hour=2):
    """Sleep until *hour*"""
    now = datetime.now()
    tomorrow = now + timedelta(days=1)
    next_run = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 3, 0)
    delay = (next_run - now).total_seconds()

    print("Sleeping %d seconds until %s.." % (delay, next_run))
    time.sleep(delay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run all analysis pipeline stages to import / analyze new data on a schedule.")
    parser.add_argument('--now', default=False, action='store_true', help="Run once immediately before starting scheduled updates.")
    parser.add_argument('--skip', default='', help="comma-separated list of stages to skip")
    args = parser.parse_args(sys.argv[1:])

    if not args.now:
        delay()

    date = datetime.today().strftime("%Y-%m-%d")
    stages = OrderedDict([
        ('backup_notes',        ('daily',  'pg_dump -d data_notes -h 10.128.36.109 -U postgres  > data_notes_backups/data_notes_%s.pgsql'%date, 'backup data notes DB')),
        ('sync',                ('daily',  'python util/sync_rigs_to_server.py', 'sync raw data to server')),
        ('patchseq report',     ('daily',  'python util/patchseq_reports.py --daily', 'patchseq report')),
        ('pipeline',            ('daily',  'python util/analysis_pipeline.py multipatch all --update --retry', 'run analysis pipeline')),
        ('vacuum',              ('daily',  'python util/database.py --vacuum', 'vacuum database')),
        ('bake sqlite',         ('daily',  'python util/bake_sqlite.py small medium', 'bake sqlite')),
        ('bake sqlite full',    ('weekly', 'python util/bake_sqlite.py full', 'bake sqlite full')),
    ])

    skip = [] if args.skip == '' else args.skip.split(',')
    for name in skip:
        if name not in stages:
            print("Unknown stage %r. Options are: %r" % (name, list(stages.keys())))
            sys.exit(-1)

    while True:
        logfile = 'update_logs/' + time.strftime('%Y-%m-%d_%H-%M-%S') + '.log'
        for name, (when, cmd, msg) in stages.items():
            if when == 'weekly' and datetime.today().weekday() != 5:
                continue
                
            time_str = time.strftime('%Y-%M-%d %H:%M')
            full_cmd = cmd + " 2>&1 | tee -a " + logfile
            msg = ("======================================================================================\n" 
                   "    [%s]  %s\n"
                   "    %s\n"
                   "======================================================================================\n") % (time_str, msg, full_cmd)
            print(msg)
            open(logfile, 'a').write(msg)
            
            if name in skip:
                msg = "   [ skipping ]\n"
                print(msg)
                open(logfile, 'a').write(msg)
                
                skip.remove(name)  # only skip once
                continue

            os.system(full_cmd)
        
        delay()


