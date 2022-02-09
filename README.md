# RFDesign: Protein hallucination and inpainting with RoseTTAFold
Jue Wang (juewang@post.harvard.edu)    
Doug Tischer (dtischer@uw.edu)    
Sidney Lisanza (lisanza@uw.edu)    
David Juergens (davidcj@uw.edu)    
Joe Watson (jwatson3@uw.edu)    

This repository contains code for protein hallucination or inpainting, as
described in [our
preprint](https://www.biorxiv.org/content/10.1101/2021.11.10.468128v2). Code
for postprocessing and analysis scripts included in `scripts/`.


## License

All code is released under the MIT license.

All weights for neural networks are released for non-commercial use only under the [Rosetta-DL license](https://files.ipd.uw.edu/pub/RoseTTAFold/Rosetta-DL_LICENSE.txt).

## Installation

1. Clone the repository:
```
    git clone https://github.com/RosettaCommons/RFDesign.git
    cd rfdesign
```

2. Create environment and install dependencies:

```
    cd envs
    conda env create -f SE3.yml
```

3. Download model weights (see license info above).

```
    wget https://files.ipd.uw.edu/pub/rfdesign/weights.tar.gz
    tar xzf weights.tar.gz
```

4. Configure path to weights. Put a file called config.json in `hallucination/` and
`inpainting/` with the path to the weights directory. An example file is in each
folder to copy from.

### Dependencies
If you want/need to configure your environment manually, here are the packages in our environment:

 - python 3.8
 - pytorch 1.10.1
 - cudatoolkit 11.3.1
 - numpy
 - scipy
 - requests
 - packaging
 - pytorch-geometric ([installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html))
 - dgl ([installation instructions](https://www.dgl.ai/pages/start.html))
 - se3-transformer (install from [github](https://github.com/FabianFuchsML/se3-transformer-public))
 - lie_learn
 - icecream (for `inpainting.py`)

### Notes
 - If you are running this on digs at the IPD, you don't need to do steps 3-4.
 - If you are getting output pdbs that are a ball of disconnected segments (as viewed in pymol), this may be due to a problem with the spherical harmonics cached by SE3-transformer. A workaround is to copy the `hallucination/cache/` folder (a correct, clean copy of the cache) to your working directory before running `hallucinate.py` or `inpaint.py`.


## Usage

See READMEs in `hallucination/` and `inpainting/` subfolders.

## References

J. Wang, S. Lisanza, D. Juergens, D. Tischer, et al. Deep learning methods for designing proteins scaffolding functional sites. bioRxiv (2021). [link](https://www.biorxiv.org/content/10.1101/2021.11.10.468128v2)

M. Baek, et al., Accurate prediction of protein structures and interactions using a three-track neural network, Science (2021). [link](https://www.science.org/doi/10.1126/science.abj8754)

An earlier version of our hallucination method can be found at the [trdesign-motif repo](https://github.com/dtischer/trdesign-motif) and published at:

D. Tischer, S. Lisanza, J. Wang, R. Dong, I. Anishchenko, L. F. Milles, S. Ovchinnikov, D. Baker. Design of proteins presenting discontinuous functional sites using deep learning. (2020) bioRxiv [link](https://www.biorxiv.org/content/10.1101/2020.11.29.402743v1)

Our work is based on previous hallucination methods for unconstrained protein generation and fixed-backbone sequence design ([trDesign repo](https://github.com/gjoni/trDesign)):

I Anishchenko, SJ Pellock, TM Chidyausiku, ..., S Ovchinnikov, D Baker. De novo protein design by deep network hallucination. (2021) Nature [link](https://www.nature.com/articles/s41586-021-04184-w)

C Norn, B Wicky, D Juergens, S Liu, D Kim, B Koepnick, I Anishchenko, Foldit Players, D Baker, S Ovchinnikov. Protein sequence design by conformational landscape optimization. (2021) PNAS [link](https://www.pnas.org/content/118/11/e2017228118)
