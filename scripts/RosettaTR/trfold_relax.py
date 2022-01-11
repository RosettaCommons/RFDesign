#!/software/conda/envs/pyrosetta/bin/python3.7
import sys,os,json
import tempfile
import numpy as np
from arguments_relax import *
from utils import *
from pyrosetta import *
from pyrosetta.rosetta.protocols.minimization_packing import MinMover


def main():
    ########################################################
    # process inputs
    ########################################################
    # read params
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(scriptdir + '/data/params.json') as jsonfile:
        params = json.load(jsonfile)
    
    # get command line arguments
    args = get_args(params)
    print(args)
    if os.path.exists(args.PDB_OUT):
        return
    
    # init PyRosetta
    init_cmd = list()
    init_cmd.append("-multithreading:interaction_graph_threads 1 -multithreading:total_threads 1")
    init_cmd.append("-hb_cen_soft -mute all")
    init_cmd.append("-detect_disulf -detect_disulf_tolerance 2.0") # detect disulfide bonds based on Cb-Cb distance (CEN mode) or SG-SG distance (FA mode)
    init_cmd.append("-relax:dualspace true -relax::minimize_bond_angles -default_max_cycles 200")
    #init_cmd.append("-mute all")
    init(" ".join(init_cmd))
    
    # Load the pose
    print(args.PDB_IN)
    pose0 = pose_from_file(args.PDB_IN)
    
    # read and process restraints & sequence
    seq = pose0.sequence()
    L = len(seq)
    params['seq'] = seq
    #rst = gen_rst(params) 
    L1 = len(pose0.chain_sequence(1)) # binder is chain A
    rst = gen_rst(params, L1) # upweight interface interactions if input is complex

    print(params)
    
    ########################################################
    # full-atom refinement
    ########################################################
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)
    
    # First round: Repeat 2 torsion space relax w/ strong disto/anglogram constraints
    sf_fa_round1 = create_score_function('ref2015_cart')
    sf_fa_round1.set_weight(rosetta.core.scoring.atom_pair_constraint, 3.0)
    sf_fa_round1.set_weight(rosetta.core.scoring.dihedral_constraint, 1.0)
    sf_fa_round1.set_weight(rosetta.core.scoring.angle_constraint, 1.0)
    sf_fa_round1.set_weight(rosetta.core.scoring.pro_close, 0.0)
    
    relax_round1 = rosetta.protocols.relax.FastRelax(sf_fa_round1, "%s/data/relax_round1.txt"%scriptdir)
    relax_round1.set_movemap(mmap)
    
    print('relax: First round... (focused on torsion space relaxation)')
    params['PCUT'] = 0.15
    pose0.remove_constraints()
    add_rst(pose0, rst, 3, len(seq), params, nogly=True, use_orient=True)
    relax_round1.apply(pose0)
    
    # Set options for disulfide tolerance -> 0.5A
    print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
    rosetta.basic.options.set_real_option('in:detect_disulf_tolerance', 0.5)
    print (rosetta.basic.options.get_real_option('in:detect_disulf_tolerance'))
    
    sf_fa = create_score_function('ref2015_cart')
    sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, 0.1)
    sf_fa.set_weight(rosetta.core.scoring.dihedral_constraint, 0.0)
    sf_fa.set_weight(rosetta.core.scoring.angle_constraint, 0.0)
    
    relax_round2 = rosetta.protocols.relax.FastRelax(sf_fa, "%s/data/relax_round2.txt"%scriptdir)
    relax_round2.set_movemap(mmap)
    relax_round2.cartesian(True)
    relax_round2.dualspace(True)
    
    print('relax: Second round... (cartesian space)')
    params['PCUT'] = 0.30 # To reduce the number of pair restraints..
    pose0.remove_constraints()
    pose0.conformation().detect_disulfides() # detect disulfide bond again w/ stricter cutoffs
    # To reduce the number of constraints, only pair distances are considered w/ higher prob cutoffs
    add_rst(pose0, rst, 3, len(seq), params, nogly=True, use_orient=False, p12_cut=params['PCUT'])
    # Instead, apply CA coordinate constraints to prevent drifting away too much (focus on local refinement?)
    add_crd_rst(pose0, L, std=1.0, tol=2.0)
    relax_round2.apply(pose0)
    
    # Re-evaluate score w/o any constraints
    scorefxn_min=create_score_function('ref2015_cart')
    scorefxn_min.score(pose0)
   
    print("Writing relaxed pdb")
    pose0.dump_pdb(args.PDB_OUT)
    print("Done writing")

if __name__ == '__main__':
    main()
