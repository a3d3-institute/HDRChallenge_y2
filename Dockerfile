# Use a smaller base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Combine all apt operations in a single RUN command
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    pkg-config \
    hdf5-tools \
    wget gnupg ca-certificates \
    r-base r-base-dev \
    libcurl4-openssl-dev libssl-dev libxml2-dev libgit2-dev \
    build-essential \
    libnetcdf-dev netcdf-bin gfortran libhdf5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc \
    | gpg --dearmor -o /usr/share/keyrings/r-project.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/r-project.gpg] \
         https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" \
         > /etc/apt/sources.list.d/r-project.list
         
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
    huggingface_hub \
    tensorflow[and-cuda] && \
    pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN R -e "install.packages('remotes', repos='https://cloud.r-project.org', Ncpus = parallel::detectCores())" && \
    R -e "remotes::install_github('eco4cast/score4cast', dependencies=TRUE, upgrade='never', Ncpus = parallel::detectCores())"

CMD ["python3"]

# To build image from docker file
# docker build -t hdr_images --platform linux/amd64 .

# To run this image
# docker run -it hdr_images

# get container id from running image
# then exec with the following command
# docker exec -it container_id /bin/bash

# copy bundle to docker container
# docker cp /your/local/directory your_container_name:/app


# tag your image
# docker tag hdr_images ytchou97/hdr_images:latest

# push image
# docker push ytchou97/hdr_images:latest
