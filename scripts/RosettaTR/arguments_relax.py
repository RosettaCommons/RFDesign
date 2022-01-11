import argparse

def get_args(params):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("NPZ", type=str, help="input distograms and anglegrams (NN predictions)")
    parser.add_argument("PDB_IN", type=str, help="input model (in PDB format)")
    parser.add_argument("PDB_OUT", type=str, help="output model (in PDB format)")
    parser.add_argument('--roll', dest='roll', action='store_true', help='circularly shift 6d coordinate arrays by 1')
    parser.add_argument('-bb', type=str, dest='bb', default='', help='predicted backbone torsions')
    parser.add_argument('--orient', dest='use_orient', action='store_true', help='use orientations')
    parser.add_argument('-sg', type=str, dest='sg', default='', help='window size and order for a Savitzky-Golay filter (comma-separated)')

    args = parser.parse_args()

    params['NPZ'] = args.NPZ
    params['ROLL'] = args.roll
    params['USE_ORIENT'] = args.use_orient
    params['SG'] = args.sg
    params['BB'] = args.bb
    

    return args
