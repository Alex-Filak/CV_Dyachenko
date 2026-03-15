FROM jupyter/datascience-notebook

USER root

# Create work directory if it doesn't exist
RUN mkdir -p /home/jovyan/work && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/work

# Install plotly
RUN mamba install --yes -c conda-forge \
    plotly \
    plyfile \
    && mamba clean --all -f -y

# Switch back to jovyan user
USER ${NB_UID}
