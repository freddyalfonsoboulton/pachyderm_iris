FROM continuumio/miniconda3
LABEL maintainer='Freddy Boulton freddy.boulton@gmail.com'
WORKDIR /code
RUN conda install jupyter -y && \
	conda install scikit-learn -y && \
	conda install pandas -y && \
	conda install numpy -y && \
	conda install matplotlib && \	
	conda clean -y -all
ADD reporting.py /code/reporting.py
