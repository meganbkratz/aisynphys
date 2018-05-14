import os, sys, glob, datetime, csv, re
from acq4.util.DataManager import getDirHandle
from multipatch_analysis import config
from multipatch_analysis.constants import INTERNAL_RECIPES
from multipatch_analysis.experiment import Experiment

slice_csv_file = 'Slice_Preparation_Record.csv'
osm_csv_file = 'MultiPatch Throughput.xlsx - Solution Making.csv'

# read all dissection times
recs = list(csv.reader(open(slice_csv_file).readlines()))
diss_columns = {
    'id_col': recs[0].index('LabTracks ID'),
    'date_col': recs[0].index('Date Sliced'),
    'sac_col': recs[0].index('Time Mouse Sacrificed'),
    'time_col': recs[0].index('heated ACSF1 incub. Comp.')}
diss_times = {rec[diss_columns['id_col']]: rec for rec in recs[1:]}

# read solution making
osm_recs = list(csv.reader(open(osm_csv_file).readlines()))
osm_columns = {
    'osm_date_col': osm_recs[0].index('Date'),
    'recipe_col': osm_recs[0].index('Recipe'),
    'osm_col': osm_recs[0].index('osmolarity')}
osm_dates = {osm_rec[osm_columns['osm_date_col']]: osm_rec for osm_rec in osm_recs[1:]}

def solution_check(dh, expt_date, columns):
    solution = dh.info().get('solution', None)
    osm = dh.info().get('solution_osm', None)
    date = expt_date.strftime('%#m/%#d/%Y')
    try:
        osm_entry = osm_dates[date]
    except KeyError:
        print("\tSolution date not in Multipatch_throughput sheet, check notes\r")
        return

    recipe = osm_entry[columns['recipe_col']]
    sheet_osm = osm_entry[columns['osm_col']]
    if solution is None:
        print("\tSet aCSF: %s" % recipe)
        return

    calcium = solution.split('m')[0]
    if calcium not in recipe:
        print("\taCSF disagreement: %s != %s" % (solution, recipe))

    if osm is None:
        print("\tSet osmolarity: %s" % sheet_osm)
        return

    if sheet_osm != osm:
        print("\tOsmolarity disagreement: %s != %s" % (osm, sheet_osm))
        return


def dissection_check(dh, sub_id, expt_date, columns):
    if sub_id is None:
        print("\tNo animal_ID for %s" % dh.path)
        return
    dis_rec = diss_times.get(sub_id, None)
    if dis_rec is None:
        print("\tNo tissue processing record")
        return

    date = dis_rec[columns['date_col']]
    sac = dis_rec[columns['sac_col']]
    time = dis_rec[columns['time_col']]
    sac_time = datetime.datetime.strptime(date + ' ' + sac, '%m/%d/%Y %I:%M %p')
    if not time:
        print("\tNo time record, sacrifice time was %s" % sac)
        return
    if time[-2:] in ('AM', 'PM'):
        dis_time = datetime.datetime.strptime(date + ' ' + time, '%m/%d/%Y %I:%M %p')
    else:
        dis_time = datetime.datetime.strptime(date + ' ' + time, '%m/%d/%Y %I:%M')
        if dis_time < sac_time:
            dis_time = dis_time + datetime.timedelta(seconds=12 * 3600)

    if (dis_time - sac_time).total_seconds() > 3600:
        print("\tSac. / dissection times for are too far apart")
        return
    if dis_time <= sac_time:
        print("\tDissection before sac\r")
        return

    if expt_date != dis_time.date():
        if (expt_date - dis_time.date()).days != 1:
            print("\tExpt date %s does not match dissection date %s" % (expt_date, dis_time.date()))
            return
        dis_time_1 = "{d.year}-{d.month}-{d.day} {d.hour}:{d.minute:02d}".format(d=dis_time)
    else:
        dis_time_1 = "{d.hour}:{d.minute:02d}".format(d=dis_time)

    dis_time_2 = dh.info().get('time_of_dissection', '')  # recorded by rig operator
    if dis_time_2 == '':
        print("\tSet dissection time: %s" % dis_time_1)
        return
    else:
        if dis_time_2 != dis_time_1:
            print("\tDissection time disagreement:  %r != %r" % (dis_time_2, dis_time_1))
            return

def project_check(dh, sub_id, expt_date, species):
    mouse_prod = datetime.date(2017, 10, 01)
    project = dh.info().get('project', None)
    if project is None:
        if species is None and sub_id is None:
            print("\tNo specimen, can't set project code")
            return
        if species.lower() == 'human':
            print("\tSet Project Code: human coarse matrix")
            return
        if expt_date < mouse_prod:
            print("\tSet Project Code: mouse V1 pre-production")
        else:
            print("\tSet Project Code: mouse V1 coarse matrix")

def region_check(dh, species):
    region = dh.info().get('target_region', None)
    if species is None:
        print("\tCan't set target region")
        return
    if species == 'mouse':
        if region is None:
            print("Set target region: V1")
            return
        if region != 'V1':
            print("\tTarget region mismatch: %s != V1" % region)

def internal_check(dh, species, genotype):
    internal = dh.info().get('internal', None)
    internal_dye = dh.info().get('internal_dye', None)
    if internal is not None and internal not in INTERNAL_RECIPES:
        print("\tInternal mismatch: %s not in recipe list" % internal)
    elif internal is None:
        print("\tSet Internal: Standard K-Gluc")

    if internal_dye is None and species is None:
        print("\tCan't set internal dye")
        return
    if internal_dye is None and species.lower() == 'human':
        print("\tSet internal dye: AF488")
        return
    if internal_dye is None and genotype is not None:
        if len(genotype.split(';')) < 3:
            print("\tSet internal dye: AF488, %s looks likes single transgenic" % genotype)
        elif len(genotype.split(';')) >= 3:
            print("\tSet internal dye: Cascade Blue, %s looks liked quad" % genotype)
        else:
            print("\tCan't parse genotype %s, set internal dye manually" % genotype)

def rig_check(dh):
    rig = dh.info().get('rig_name', None)
    if rig in (None, ''):
        #dh.setInfo(rig_name=config.rig_name)
        print("\tSet Rig: %s" % config.rig_name)
        return
    if rig != config.rig_name:
        print("\tRig mismatch: %s != %s" % (rig, config.rig_name))


root = sys.argv[1]

# find all subject folders that contain at least one site folder
sites = glob.glob(os.path.join(root, '*', 'slice_*', 'site_*'))
checked_days = set()
checked_slices = set()

for path in sites:
    site_dh = getDirHandle(path)
    slice_dh = site_dh.parent()
    day_dh = slice_dh.parent()
    sub_id = day_dh.info().get('animal_ID', None)
    expt_date = datetime.datetime.fromtimestamp(day_dh.info()['__timestamp__']).date()
    species = day_dh.info().get('species', None)
    if species != 'human':
        genotype = species
    if sub_id is not None and species is None:
        try:
            species = day_dh.info().get('LIMS_specimen_info')['organism']
            genotype = day_dh.info().get('LIMS_specimen_info')['genotype']
        except TypeError:
            try:
                species = day_dh.info().get('LIMS_donor_info')['organism']
            except TypeError:
                guess = sub_id[0]
                if guess == 'H':
                    species = 'human'
                else:
                    species = 'mouse'

    print("Experiment Date: %s\nAnimal ID: %s" % (expt_date, sub_id))
    if day_dh not in checked_days:
        rig_check(day_dh)
        dissection_check(day_dh, sub_id, expt_date, diss_columns)
        solution_check(day_dh, expt_date, osm_columns)
        region_check(day_dh, species)
        internal_check(day_dh, species, genotype)
        checked_days.add(day_dh)

    if slice_dh not in checked_slices:
        project_check(slice_dh, sub_id, expt_date, species)
        checked_slices.add(slice_dh)


