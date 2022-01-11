# RosettaFold joint seq-str modelling + inpainting script 
David Juergens (davidcj@uw.edu)    
Jue Wang (jue@uw.edu)    
Sidney Lisanza (lisanza@uw.edu)    
Joe Watson (jwatson3@uw.edu)

## Installation

See instructions in `rfdesign/README.md`

## Usage

Remove residues 20-25 from chain A of 2KL8.pdb and rebuild their structure and sequence,
using 10 iterations of recycling the structure, outputting 1 design to `output/`.

    conda activate SE3
    python inpaint.py --task hal --checkpoint v02 --pdb=input/2KL8.pdb --out=output/inpaint_test \
                      --window=A,20,25 --num_designs 1 \
                      --inf_method=multi_shot --n_iters 10 --recycle_str_mode both \
                      --clamp_1d 0.1 --dump_all

The above command should actually run if you start in the working folder `test/` (and run `../inpaint.py`).


## Output

By default, only a .trb file of metadata will be output for each design, and a
.csv for sequence recovery for all designs. Use these flags to generate other
per-design outputs:

  - `--dump_pdb`: pdb file of the generated backbone
  - `--dump_npz`: npz file with 6D predictions for pyrosetta folding or relax.
  - `--dump_fas`: fasta file with designed sequence, with '/' delimiting chains.
  - `--dump_all`: output all of the above

When the output has more than one chain, an additional \<prefix\>_two_chain.npz
file is output that contains permuted dimensions for compatibility with the
`BFF/design/postprocess/fold_complex.sh` folding script. (See README in that
folder).

Post-processing and analysis can be done using the scripts in
`rfdesign/scripts/`. See README there for details.

## RoseTTAFold checkpoints

 - v00: RoseTTAFold trained on fixed-backbone sequence design task (i.e.
   backbone structure + masked sequence input, sequence output logits loss)
 - v01: RoseTTAFold trained on fixed-backbone sequence design and seq/struc inpainting (mask and recover both sequence and structure)
 - v02: RoseTTAFold trained on fixed-backbone sequence design, seq/struc inpainting, and seq/struc inpainting with masked flanking residues.

In general, v02 has the best performance, but for some problems v00 may be
better. v01 usually is inferior to v00 and v02.


## Specifying masked versus kept residues

The `--window` option specifies regions to mask (remove and replace). e.g.
`--window A,1,5:B,24,36` will rebuild residues 1-5 (inclusive) on chain A and
residues 24-36 on chain B.

The `--contigs` option specifies regions to keep (and to build the rest of the
protein). The kept regions ("contigs") are also placed randomly into a *new*
protein of a certain length.

  - `--contigs A5-10,A25-40 --len 100` will put the contiguous regions A5-10
    and A25-40 randomly into a new protein of length 100 (and build the rest of
    that protein).

    By default, the order of contigs is randomized in the new protein. Set
    `--keep_order` to avoid this. By default, the minimum number of residues
    between contigs is 3.  Set `--min_gap` to change this.

  - `--contigs 5,A5-10,7-9,A25-40,0-4` will put a masked region of length 5,
    then the region A5-10 from the input pdb, then a masked region of length
    7-9 (inclusive, sampled uniformly randomly for each design), then A25-40
    from the input, then a masked region of 0-4.  This does not require a
    `--len` argument because the total length is the sum of all the sampled
    masked regions and the contigs. The order of the contigs is maintained. 

    To rebuild a loop in residues 5-10 of the an input protein of length 100,
    but have it vary in length in the output between 4-8 AAs:

        ./inpaint.py --contigs A1-4,4-8,A11-100 ...

If using `--contigs` and you want to design in the context of a receptor from
the input, specify the chain letter of the receptor with `--receptor_chain`.


## Prediction modes

`--inf_method` controls how the script builds the masked regions:

  - `--one_shot` predicts all masked positions in 1 forward pass. Output
    sequence is the argmax of the predicted sequence logits.

  - `--multi_shot` predicts all masked positions in `--n_iters` forward passes.
    After the 1st pass, the entire predicted structure is fed as the template
    for the subsequent pass, and 1d template features are the residue-wise
    predicted lDDTs. Use `--clamp_1d` to limit how high these can get, and
    `--local_clamp` to only clamp at masked positions. Output sequence is the
    argmax of the predicted sequence logits at the end. 

  - `--autoregressive` predicts all masked positions, sets the N-term-most
    masked residue to the AA sampled from the logits with temperature
    `--autoreg_T`, predicts all remaining positions, and so on until all masked
    positions have been assigned a sequence. Returns the structure and sequence
    predicted on the last forward pass. Structure of masked region is masked
    the entire time.


## Command-line options

    --task      : which task are you performing in the window? Options are 'seq' (sequence design), 'str' (structure prediction), 'hal' (hallucination, aka structure and sequence inpainting), or 'den' (denoising). 

    --window    : residue indices to remove sequence and structure information from. A colon-separated list of start,end of window.
                    E.g., `--window 1,5:10,20:45,47` tells the algorithm to remove information from residues (indexed in the PDB file) 1 through 5, 
                            10 through 20, and 45 through 47. 
    --res_translate : physically move individual residues by up to a maximum distance, in any direction. 
                    E.g., `--res_translate A35,2:B22,4` translates residue A35 up to 2A in a random direction, and B22 up to 4A.
                            If no distance is provided, default distance is 2A. Translation is different for each of --num_designs
    --pdb       : path to pdb file you want to use 

    --n_iters   : How many iterations do you want to do with multi-shot prediction 

    --dump_pdb  : If this flag appears, will dump pdb of resulting structure + ORIGINAL sequence (need to fix this, will do it soon)

    --outf     : Folder to dump output 
    
    --checkpoint : which checkpoint to use (options are v00, v01 or v02). Some evidence that v02 (Joe's model) performs better than previous models. 
