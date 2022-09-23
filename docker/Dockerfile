FROM --platform=linux/x86_64 nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
WORKDIR /root

RUN apt-get update \
    && apt-get install -y wget git \
    && rm -rf /var/lib/apt/lists/* \
    && wget \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh \
    && conda --version

RUN git clone --depth 1 --branch main https://github.com/RosettaCommons/RFDesign.git \
    && cd RFDesign \
    && wget -nv -P hallucination/weights/rf_Nov05 http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_last.pt \
    && wget -nv -P inpainting/weights/ http://files.ipd.uw.edu/pub/rfdesign/weights/BFF_mix_epoch25.pt

# Download AlphaFold2 parameters.
# AlphaFold is used to evaluate the rfdesign results.
# The /software/mlfold stuff is a hack because the rfdesign af2_metrics.py
# script has that path hardcoded.
RUN mkdir -p alphafold-params \
    && wget -nv https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar -O params.tar \
    && tar --extract --verbose --file=params.tar --directory=alphafold-params --preserve-permissions \
    && rm -f params.tar \
    && mkdir -p /software/mlfold/alphafold-data \
    && ln -s /root/alphafold-params /software/mlfold/alphafold-data/params

RUN conda update -n base -c defaults conda \
  && conda config --set ssl_verify no \
  && conda init bash \
  && conda clean -afy

# Note that we are using a different version of pytorch than recommended
# in RFDesign readme (suggested version is pytorch=1.10.1)
RUN conda create -n rfdesign-cuda \
    python=3.8 \
    pytorch=1.11 \
    dgl-cuda11.3 \
    pyg \
    numpy==1.21.5 scipy==1.7.3 requests==2.28.1 packaging pip \
    openmm==7.5.1 \
    -c pytorch -c dglteam -c pyg -c conda-forge

RUN /root/miniconda3/envs/rfdesign-cuda/bin/pip install \
        https://github.com/openmm/pdbfixer/archive/refs/tags/v1.7.tar.gz \
        icecream==2.1.3 \
        lie_learn==0.0.1.post1 \
        opt_einsum==3.3.0 \
        e3nn==0.3.4 \
    && /root/miniconda3/envs/rfdesign-cuda/bin/pip install \
        "jaxlib[cuda]==0.1.69" \
        "jax[cuda]==0.2.14" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && /root/miniconda3/envs/rfdesign-cuda/bin/pip install \
        dm-tree==0.1.6 \
        dm-haiku==0.0.4 \
        absl-py==1.0.0 \
        ml-collections==0.1.0 \
        tensorflow-gpu==2.9.0 \
        biopython==1.79 \
    && conda clean -afy
