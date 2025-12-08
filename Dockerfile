FROM jupyter/datascience-notebook

USER root

RUN mkdir -p /home/jovyan/work && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/work

# Use python -m pip to guarantee correct environment
RUN conda install --quiet --yes -c conda-forge \
    plotly \
    plyfile \
    scipy \
    && conda clean --all -f -y

USER ${NB_UID}
