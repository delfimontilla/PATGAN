FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# use an older system (18.04) to avoid opencv incompatibility (issue#3524)

#ENV DEBIAN_FRONTEND noninteractive

#Change of NVIDIA public keys
RUN sh -c 'echo "APT { Get { AllowUnauthenticated \"1\"; }; };" > /etc/apt/apt.conf.d/99allow_unauth'
RUN apt -o Acquire::AllowInsecureRepositories=true -o Acquire::AllowDowngradeToInsecureRepositories=true update
RUN apt-get install -y curl wget
RUN apt-key del 7fa2af80
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/apt.conf.d/99allow_unauth cuda-keyring_1.0-1_all.deb
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC F60F4B3D7FA2AF80

#fix GPU not recognized
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
RUN curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-opencv ca-certificates python3-dev git wget sudo ninja-build nvidia-container-toolkit
RUN ln -sv /usr/bin/python3 /usr/bin/python
 
WORKDIR /home/
ENV PATH="/home/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

#for GUI
#RUN apt-get install -qqy x11-apps

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard cmake   # cmake from apt-get is too old
RUN pip install --user torch==1.10 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html


RUN mkdir PAT_GAN/
COPY . /home/PAT_GAN/ 

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN cd /home/PAT_GAN/ && pip install -r requirements.txt
RUN cd /home/PAT_GAN/ && python setup.py develop

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
