FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
# Copy over source code
COPY . ./
ENTRYPOINT ["python", "main.py"]