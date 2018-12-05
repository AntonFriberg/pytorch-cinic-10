FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime
# Download CINIC-10 dataset
RUN mkdir -p data/cinic-10 && curl -L \
 https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz | \
 tar xz -C data/cinic-10
# Copy over source code
COPY . ./
CMD python main.py