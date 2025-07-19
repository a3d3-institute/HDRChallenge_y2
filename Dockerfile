# Use a smaller base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Combine all apt operations in a single RUN command
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    pkg-config \
    hdf5-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Combine all pip installations in a single RUN command
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    numpy \
    pandas \
    pyarrow \
    matplotlib \
    xgboost \
    scikit-learn \
    nflows \
    lightgbm \
    seaborn \
    iminuit \
    keras \
    transformers \
    netcdf4 \
    h5netcdf \
    scipy \
    xarray \
    tensorflow[and-cuda] && \
    pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
# Set the default command to Python3
CMD ["python3"]

# To build image from docker file
# docker build -t fair_universe --platform linux/amd64 .

# To run this image
# docker run -it fair_universe

# get container id from running image
# then exec with the following command
# docker exec -it container_id /bin/bash

# copy bundle to docker container
# docker cp /your/local/directory your_container_name:/app


# tag your image
# docker tag fair_universe ihsaanullah/fair_universe:latest

# push image
# docker push ihsaanullah/fair_universe:latest
