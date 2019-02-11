FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
WORKDIR /code
# Download CINIC-10 dataset
RUN mkdir -p data/cinic-10 && curl -L \
 https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz | \
 tar xz -C data/cinic-10
# Copy over source code
COPY . ./
ENTRYPOINT ["python", "main.py", "data/cinic-10"]