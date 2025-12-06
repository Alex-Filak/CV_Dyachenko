FROM jupyter/datascience-notebook

USER root

RUN pip install plotly

USER $(NB_UID)
