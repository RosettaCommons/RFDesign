# Pipeline scripts for hallucination
Jue Wang (juewang@post.harvard.edu)    

This folder contains scripts for preparing inputs and postprocessing and
scoring results from hallucination (`design.py`)

## Usage

After a hallucination or inpainting run, first generate a relaxed model with sidechains:

    ./trf_relax.sh FOLDER

where FOLDER contains hallucination results (.fas, .pdb, .npz, .trb files).
This will require the .pdb and .npz files. After this completes, there will be
a folder `FOLDER/trf_relax` containing pdbs of the relaxed structures.

Generally, it is best to operate on these structures rather than the raw
backbone-only models from `design.py`, because these relaxed structures will be
more accurate (and have sidechains).  Wait until all jobs have finished, then
do any of the following. 

We usually filter on:

    - RMSD of the motif between reference structure and AF2 prediction
      (contig_rmsd_af2, or interface_rmsd_af2)
    - Predicted lDDT by AF2 (af2_lddt)
    - Radius of gyration (rog)

We also often filter on net charge, SAP score, and others, but those have not
been integrated into this workflow yet.


### AlphaFold2

To make AlphaFold2 predictions and compute RMSDs against the hallucination
design model and the template structure, run (on GPU node):

    ./af2_metrics.py FOLDER/trf_relax

If you're on the head node, you can submit a GPU job for AF2 metrics using:

    sbatch -p gpu --mem 12g --gres=gpu:rtx2080:1 --wrap="af2_metrics.py FOLDER/trf_relax"

This outputs AF2 models to `FOLDER/trf_relax/af2/` and metrics to
`FOLDER/af2_metrics.csv`.  The script automatically uses a template PDB found
in the .trb file corresponding to each design. If you would like to specify the
template, you can do:
  
    ./af2_metrics.py --template TEMPLATE_PDB FOLDER/trf_relax
  
Passing a file with space-delimited residue numbers (as numbered in the
reference structure) to `--interface_res` will also output an RMSD only on
these positions.
      
    ./af2_metrics.py --template TEMPLATE_PDB --interface_res RESNUM_FILE FOLDER/trf_relax

If you run `af2_metrics.py` a second time, it will use any existing pdb files
in the `af2/` subfolder rather than making new predictions. To redo the
predictions, delete the `af2` folder (or specific pdbfiles).

### Pyrosetta metrics

Running any of: 

    ./pyrosetta_metrics.py FOLDER/trf_relax
    ./pyrosetta_metrics.py --template REFERENCE.pdb FOLDER/trf_relax
    ./pyrosetta_metrics.py --template REFERENCE.pdb --interface_res A163-181 FOLDER/trf_relax

will calculate RMSD between the hallucination (RoseTTAFold) design model and
the reference structure, as well as radius of gyration, secondary structure,
topology (i.e. HHH or HEEH).


### Compile metrics

When you have computed all metrics, you can load the resulting tables using custom code, or most of the metrics can be combined into a single file by using

    ./compile_metrics.py FOLDER/trf_relax

The result will be a single file `FOLDER/trf_relax/combined_metrics.csv`. 


### Align models in PyMOL

To make a pymol session with designs aligned to reference structures on the
constrained regions, use `pymol_align.py`. For example:

    ./pymol_align.py -- -o OUTPUT.pse FOLDER/*pdb

Will make a session called OUTPUT.pse in the current folder containing the
original structure from REFERENCE.pdb with all the designs FOLDER/*.pdb aligned
to it. The `--` is needed to bypass PyMOL argument processing.

The above command automatically loads the reference pdb using its path as
stored in the 1st .trb file that is processed. If you would like to specify the
reference pdb (e.g. if the path in the .trb is broken), you can do:

    ./pymol_align.py --template REFERENCE.pdb -o OUTPUT.pse FOLDER/*pdb

If the designs contain multiple chains (i.e. a receptor you've designed in the
context of), `pymol_align.py` won't be able to handle aligning all the
residues. But you can still align on the constrained motif only, by excluding
the receptor chain from the alignment:

    ./pymol_align.py --template REFERENCE.pdb -o OUTPUT.pse --exclude_chain R FOLDER/*pdb

See code of pymol_align.py for a few more options. 


### Legacy scripts

Previously, we computed RMSD, DAN-lDDT (Deep Accuracy Net), and TM-score to
template using this command, which is a wrapper for `pyrosetta_metrics.py`,
`lddt.sh`, and `get_tmscores.py`. Now we use AF2 plDDT rather than DAN-lDDT, so
this combined script isn't often used.

    ./calc_metrics.sh FOLDER/trf_relax

