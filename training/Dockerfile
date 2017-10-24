FROM continuumio/miniconda3
LABEL maintainer='Freddy Boulton freddy.boulton@gmail.com'
RUN conda install jupyter -y && \
	conda install scikit-learn -y && \
	conda install pandas -y && \
	conda install numpy -y && \
	conda install matplotlib && \	
	conda clean -y -all
WORKDIR /code
ADD  training_model.py /code/training_model.py
