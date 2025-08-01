

```

## update apt

sudo apt update 

## Install curl

sudo apt install curl  

## Install vim 

sudo apt install vim 

# Install NVIDIA drivers (this will install the latest compatible version)
sudo apt install -y nvidia-driver-535

# Reboot the VM to load the drivers
sudo reboot

# Check if drivers are loaded
nvidia-smi

## Install docker 

curl https://get.docker.com | sudo sh
sudo usermod -aG docker $USER

# Configure the production repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update the packages list from the repository:
sudo apt-get update


# Install the NVIDIA Container Toolkit packages
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure the runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker


## Create folder under root directory

# Create a folder under root "/"
sudo mkdir /codabench

# Set proper permissions (optional - for Codabench worker)
sudo chown $USER:$USER /codabench
sudo chmod 755 /codabench

# Verify the folder was created
ls -la /codabench


# Create .env and docker_compose

# .env 
# Queue URL
BROKER_URL=<desired broker URL>

# Location to store submissions/cache -- absolute path!
HOST_DIRECTORY=/codabench

# If SSL isn't enabled, then comment or remove the following line
BROKER_USE_SSL=True




```

