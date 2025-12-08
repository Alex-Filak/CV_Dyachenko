FROM jupyter/datascience-notebook

USER root

RUN mkdir -p /home/jovyan/work && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/work

RUN conda install --quiet --yes -c conda-forge \
    plotly \
    plyfile \
    scipy \
    h5py \
    tqdm \
    seaborn \
    scikit-learn \
    && conda clean --all -f -y

RUN pip install --no-cache-dir \
    torch \
    torchvision \
    open3d \
    ipywidgets

USER ${NB_UID}
