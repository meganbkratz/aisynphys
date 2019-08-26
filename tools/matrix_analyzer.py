import sys, argparse
import pyqtgraph as pg
from multipatch_analysis.database import default_db as db
from multipatch_analysis.matrix_analyzer import MatrixAnalyzer
from collections import OrderedDict

if __name__ == '__main__':

    app = pg.mkQApp()
    pg.dbg()
    # pg.setConfigOption('background', 'w')
    # pg.setConfigOption('foreground', 'k')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    args = parser.parse_args(sys.argv[1:])
    analyzer_mode = args.mode if args.mode is not None else 'internal'

    session = db.session()
    
    # Define cell classes
    cell_class_groups = OrderedDict([
        ('Mouse All Cre-types by layer', [
            {'cre_type': 'unknown', 'target_layer': '2/3','cortical_layer': '2/3'},
            #{'pyramidal': True, 'target_layer': '2/3'},
            {'cre_type': 'pvalb', 'target_layer': '2/3', 'cortical_layer': '2/3'},
            {'cre_type': 'sst', 'target_layer': '2/3', 'cortical_layer': '2/3'},
            {'cre_type': 'vip', 'target_layer': '2/3', 'cortical_layer': '2/3'},
           # {'cre_type': 'rorb', 'target_layer': '4'},
            {'cre_type': 'nr5a1', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'pvalb', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'sst', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'vip', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'sim1', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'tlx3', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'pvalb', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'sst', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'vip', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'ntsr1', 'target_layer': '6', 'cortical_layer': '6'},
            {'cre_type': 'pvalb', 'target_layer': '6', 'cortical_layer': '6'},
            {'cre_type': 'sst', 'target_layer': '6', 'cortical_layer': '6'},
            {'cre_type': 'vip', 'target_layer': '6', 'cortical_layer': '6'},
        ]),

        ('Mouse Layer 2/3', [
            {'cre_type': 'unknown', 'target_layer': '2/3', 'cortical_layer': '2/3'},
            #{'pyramidal': True, 'target_layer': '2/3'},
            # {'dendrite_type': 'spiny', 'target_layer': '2/3', 'cortical_layer': '2/3'},
            {'cre_type': 'pvalb', 'target_layer': '2/3', 'cortical_layer': '2/3'},
            {'cre_type': 'sst', 'target_layer': '2/3', 'cortical_layer': '2/3'},
            {'cre_type': 'vip', 'target_layer': '2/3', 'cortical_layer': '2/3'},
        ]),
        
        ('Mouse Layer 4', [
            {'cre_type': 'nr5a1', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'pvalb', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'sst', 'target_layer': '4', 'cortical_layer': '4'},
            {'cre_type': 'vip', 'target_layer': '4', 'cortical_layer': '4'},
        ]),

        ('Mouse Layer 5', [
            {'cre_type': ('sim1', 'fam84b'), 'target_layer': '5', 'display_names': ('L5', 'PT\nsim1, fam84b'), 'cortical_layer': '5'},
            {'cre_type': 'tlx3', 'target_layer': '5', 'display_names': ('L5', 'IT\ntlx3'), 'cortical_layer': '5'},
            {'cre_type': 'pvalb', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'sst', 'target_layer': '5', 'cortical_layer': '5'},
            {'cre_type': 'vip', 'target_layer': '5', 'cortical_layer': '5'},
        ]),

        ('Mouse Layer 6', [
            {'cre_type': 'ntsr1', 'target_layer': '6', 'cortical_layer': '6'},
            {'cre_type': 'pvalb', 'target_layer': '6', 'cortical_layer': '6'},
            {'cre_type': 'sst', 'target_layer': '6', 'cortical_layer': '6'},
            {'cre_type': 'vip', 'target_layer': '6', 'cortical_layer': '6'},
        ]),

        ('Mouse Inhibitory Cre-types',[
            {'cre_type': 'pvalb'},
            {'cre_type': 'sst'},
            {'cre_type': 'vip'},
        ]),
 
        ('Mouse Excitatory Cre-types', [
            # {'pyramidal': True, 'target_layer': '2/3'},
            {'cre_type': 'unknown', 'target_layer': '2/3'},
            {'cre_type': 'nr5a1', 'target_layer': '4'},
            {'cre_type': 'sim1', 'target_layer': '5'},
            {'cre_type': 'tlx3', 'target_layer': '5'},
            {'cre_type': 'ntsr1', 'target_layer': '6'},
        ]),

        ('Mouse E-I Cre-types', [
            {'cre_type': ('unknown', 'nr5a1', 'tlx3', 'sim1', 'ntsr1'), 'display_names': ('', 'Excitatory\nunknown, nr5a1,\ntlx3, sim1, ntsr1')},
            {'cre_type': ('pvalb', 'sst', 'vip'), 'display_names': ('', 'Inhibitory\npvalb, sst, vip')},
        ]),

        ('Mouse E-I Cre-types by layer',[
            # {'pyramidal': True, 'target_layer': '2/3'},
            {'cre_type': 'unknown', 'target_layer': '2/3'},
            {'cre_type': ('pvalb', 'sst', 'vip'), 'target_layer': '2/3', 'display_names': ('L2/3', 'Inhibitory\npvalb, sst, vip'), 'cortical_layer': '2/3'},
            {'cre_type': 'nr5a1', 'target_layer': '4'},
            {'cre_type': ('pvalb', 'sst', 'vip'), 'target_layer': '4', 'display_names': ('L4', 'Inhibitory\npvalb, sst, vip'), 'cortical_layer': '4'},
            {'cre_type': 'sim1', 'target_layer': '5'},
            {'cre_type': 'tlx3', 'target_layer': '5'},
            {'cre_type': ('pvalb', 'sst', 'vip'), 'target_layer': '5', 'display_names': ('L5', 'Inhibitory\npvalb, sst, vip'), 'cortical_layer': '5'},
            {'cre_type': 'ntsr1', 'target_layer': '6'},
            {'cre_type': ('pvalb', 'sst', 'vip'), 'target_layer': '6', 'display_names': ('L6', 'Inhibitory\npvalb, sst, vip'), 'cortical_layer': '6'},     
        ]),

        ('Pyramidal / Nonpyramidal by layer', [
            {'pyramidal': True, 'target_layer': '2'},
            {'pyramidal': False, 'target_layer': '2'},
            {'pyramidal': True, 'target_layer': '3'},
            {'pyramidal': False, 'target_layer': '3'},
            {'pyramidal': True, 'target_layer': '4'},
            {'pyramidal': False, 'target_layer': '4'},
            {'pyramidal': True, 'target_layer': '5'},
            {'pyramidal': False, 'target_layer': '5'},
            {'pyramidal': True, 'target_layer': '6'},
            {'pyramidal': False, 'target_layer': '6'},
        ]),

        ('Pyramidal by layer', [
            {'pyramidal': True, 'target_layer': '2'}, 
            {'pyramidal': True, 'target_layer': '3'},
            {'pyramidal': True, 'target_layer': '4'},
            {'pyramidal': True, 'target_layer': '5'},
            {'pyramidal': True, 'target_layer': '6'},
        ]),

        ('All cells by layer', [
            {'target_layer': '2'},
            {'target_layer': '3'},
            {'target_layer': '4'},
            {'target_layer': '5'},
            {'target_layer': '6'},
        ]),

        ('2P-Opto cre types', [
            {'cre_type':'ntsr1'},
            #{'cre_type':'unknown'},
            {'cre_type':'sst'},
            {'cre_type':'tlx3'},
            {'cre_type':'rorb'},
            {'cre_type':'scnn1a'}])
    ])

    if analyzer_mode == 'external':
        groups = ['Mouse All Cre-types by layer', 'Mouse Inhibitory Cre-types', 'Mouse Excitatory Cre-types', 'Mouse E-I Cre-types by layer', 'Pyramidal by layer', 'All cells by layer']
        cell_class_groups = {g:cell_class_groups[g] for g in groups}

    maz = MatrixAnalyzer(session=session, cell_class_groups=cell_class_groups, default_preset='None', preset_file='matrix_analyzer_presets.json', analyzer_mode=analyzer_mode)

    if sys.flags.interactive == 0:
        app.exec_()