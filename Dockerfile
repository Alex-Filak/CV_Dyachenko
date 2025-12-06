FROM jupyter/datascience-notebook

USER root

# Create work directory if it doesn't exist
RUN mkdir -p /home/jovyan/work && \
    chown -R ${NB_UID}:${NB_GID} /home/jovyan/work

# Install plotly
RUN pip install plotly

# Switch back to jovyan user
USER ${NB_UID}
