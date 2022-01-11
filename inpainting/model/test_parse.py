import dj_parser 
import argparse 

from icecream import ic 

path = '/home/nrbennet/methods/dl/alphafold/filtering/notebooks/pdl1/initial_guess_predictions_look_success/pdl1_normal_HHH_b2_05903_0001600013_0000006_0_0001_af2pred_0001.pdb' 

ic(path)

parsed = dj_parser.parse_multichain_pdb(path)
ic(parsed)

breaks = dj_parser.getChainbreaks(parsed)
ic(breaks)
