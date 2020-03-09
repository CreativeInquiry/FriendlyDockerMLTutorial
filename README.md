![FriendlyDockerMLTutorial](https://user-images.githubusercontent.com/480224/76093011-9bc59c00-5fc0-11ea-9fe7-f60ebd3e1242.png)

# Get ML Stuff Running, Using Docker – A Friendly Tutorial for Artist and Designers

By [@b-g](https://github.com/b-g) Benedikt Groß

We find ourselves in an exciting technological moment. In the last few years, it seems magic started to happen in Artificial Intelligence “AI”. After a long AI winter, machine learning “ML” methods and techniques started to work. 

If you follow ML news outlets like [Two Minute Papers](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg) (ML papers nicely exlpained in 2 min videos) or the recent development of [Runway ML App](https://runwayml.com/) (think of "ready made" ML effect app) there seems to be popping up one interesting ML model after another. And often these ML models come with example code on Github! Yay!

But the excitment often fades away quickly, as even though the example code or the demo of the ML model doesn't look crazy complicated ... it can become quickly pure hell to actually get it running on your computer. Most often simply because of ultra specfic software and GPU driver dependencies, which propably don't go well with what you already have installed on your machine. It is often a total mess :(

This tutorial tries to jump in by showing you an approach to handle the ML software dependency complexity better! We are going to use a [Docker](https://en.wikipedia.org/wiki/Docker_(software)) Container (think of an entire operating system nicely bundled in a container) which has access to the GPU and the files of a host machine running it. So instead of installing everything direclty in the operating system of your local machine, we create a layer of abstraction for every model. You could also say every ML model gets its own (virtual) operation system, as if you had for every ML model a dedicated computer.

Let's assume you stumbled upon the "DeepFill" paper [Generative Image Inpainting with Contextual Attention](https://arxiv.org/abs/1801.07892) and the corresponding Github repository [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting). Here is a quick illustration on what DeepFill does:

![deepfill-illustration](assets/deepfill-illustration.png)

Fancy! You provide a mask and Deepfill is hallucinating the content which matches the context.

The following sections are step by step instructions how to get DeepFill running in a Docker container. The process should be fairly similar for other models and hence can be seen as a general approach to encapsulate the setup complexity which comes with state of the art ML models. 

My hope is furthermore that dedicated Docker containers will make ML models a lot more shareable and accessible for a wider audience in Art & Design, to facilitate the very needed debate of wider implications of AI/ML.



## 0. TOC

* [1. Prerequisite 🐧](#1-prerequisite-)

* [2. Install Party: CUDA, Docker and nvidia-container-toolkit 💻](#2-install-party-cuda-docker-and-nvidia-container-toolkit-)
  
  + [Install Docker](#install-docker)
  + [Install Nvidia CUDA driver](#install-nvidia-cuda-driver)
  + [Install nvidia-container-toolkit](#install-nvidia-container-toolkit)
  
* [3. Example: Getting DeepFill running in Docker 📦](#3-example-getting-deepfill-running-in-docker-)
  + [Requirements spotting](#requirements-spotting)
  + [Fork the DeepFill repository](#fork-the-deepfill-repository)
  + [Create a Dockerfile](#create-a-dockerfile)
  + [Build the DeepFill container](#build-the-deepfill-container)
  + [Run the DeepFill container](#run-the-deepfill-container)
  + [Download pretrained DeepFill models](#download-pretrained-deepfill-models)
  + [Run the DeepFill demo in the container](#run-the-deepfill-demo-in-the-container)
  
* [4. Strategies for finding the requirements 🤯](#4-strategies-for-finding-the-requirements-)

* [5. Acknowledgments 🙏](#5-acknowledgments-)

  


## 1. Prerequisite 🐧

You will need the following hard- and software setup to be able to run Docker with GPU support:

- An Ubuntu computer/server with a Nvidia CUDA GPU
- Docker with version >= 1.4
- Nvidia drivers with version >= 361


## 2. Install Party: CUDA, Docker and nvidia-container-toolkit 💻


### Install Docker

Follow the official documentation: [https://docs.docker.com/install/linux/docker-ce/ubuntu/](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

Verify Docker version:

```bash
docker version
```

The output should be a long list with infos like "API version: 1.4" etc.

### Install nVidia CUDA driver

Install CUDA along with latest nVidia driver for your graphics card.

- Go to: https://developer.nvidia.com/cuda-downloads
- Select Linux > x86_64 > Ubuntu
- Select your ubuntu version
- Select Installer type (we tested with deb local or deb network)
- Follow instructions
- After install, reboot your machine
- Test if nvidia driver are installed with: `nvidia-smi`


Verify Nvidia drivers version:

```bash
nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.67       Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 1050    Off  | 00000000:02:00.0 Off |                  N/A |
| N/A   40C    P0     8W /  N/A |      0MiB /  2002MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
```

### Install nvidia-container-toolkit

- Follow the official [quickstart documentation](https://github.com/NVIDIA/nvidia-docker#quickstart) e.g. for Ubuntu 16.04/18.04:

```bash
# Add the package repositories
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install and reload Docker
$ sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```

- Verify installation by running a dummy docker image

```bash
$ sudo docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

# Should output something like
+----------------------------------------------------------+
| NVIDIA-SMI 418.87   Driver Version: 418.87 CUDA Version: 10.1
|-------------------------------+----------------------
```

Yay 🎉🎉🎉 ! You can go on to finally run ML models in Docker on your machine! Good news is that you just have to do the horrible installation part once.

_⚠️ Currently (December 2019) the nvidia-docker projects seems to be in an odd transition phase of supporting two slightly different ways of leveraging NVIDIA GPUs in docker containers. At the moment best practice seems to install the nvidia-container-toolkit and if needed the deprecated nvidia-docker. You can install both without running into conflicts. The install order doesn't matter as well. _

## 3. Example: Getting DeepFill running in Docker 📦

So let's have a look in DeepFill Github repository [JiahuiYu/generative_inpainting](https://github.com/JiahuiYu/generative_inpainting). The README.md is nice in the sense that it explains with a few images what DeepFill does, has references and even sections on requirements and how to run the demo. But you still won't be able to run it out of the box. Like other ML models DeepFill relies on very specific software dependencies. And as ML researchers are busy with their research, documenting software setups for a wider audience seem currently not to be a priority in those circles. The dream situation would be that there is already a `Dockerfile` (e.g. [Detectron2](https://github.com/facebookresearch/detectron2) is a notable exception), or at least a`requirements.txt` (used in Python to define dependencies). 

### Requirements spotting

We have to create the Docker container on our own. This is where we have to start to play detective! :)

Stroll around in the repository and try to find clues of what we are going to need. I found the following:
- there is a badge saying Tensorflow v1.7.0
- under requirements the author states: python3, tensorflow, neuralgym
- OpenCV (cv2) is mentioned
- and to download the pretrained models and copy them to the folder `model_logs/`

### Fork the DeepFill repository

to your own Github account. Press the "Fork" button at JiahuiYu/generative_inpainting. Result should be github.com/{your-username}/generative_inpainting

![fork_button](assets/fork_button.jpg)

Checkout your fork of generative_inpainting to your own computer.

### Create a Dockerfile

Now we will create a Docker container which reflects all the requirements we have spotted. Add an empty file named `Dockerfile`  to your DeepFill repository.

We will base everything on an official Docker container from Nvidia. That way we get a clean Ubuntu with correctly installed CUDA / CUDNN / GPU / drivers.  The syntax to write a docker container is quite understandable:

- **FROM**: new container should be based on "cuda" with tag "9.0-cudnn7-runtime-ubuntu16.04" published by "nvidia"
- **RUN**: run a single or many terminal commands to install something.
- **apt-get**: is the command line package manager of Ubuntu. Think of an app store for command line apps.
- **pip3**: is the command line package manager of Python 3. 

 Here is how the `Dockerfile` looks:

```dockerfile
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Install a few basic command line tools e.g. zip, git, cmake
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    curl 

# Install Python 3, numpy and pip3
RUN apt-get install -y \
    python3-dev \
    python3-numpy \
    python3-pip 

# Install OpenCV
RUN apt-get install -y \
    libopencv-dev \
    python-opencv

# Cleanup apt-get installs
RUN rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 --no-cache-dir install \
    opencv-python \
    Pillow \
    pyyaml \
    tqdm

# Install project specific dependencies
RUN pip3 --no-cache-dir install \
    tensorflow-gpu==1.8.0 \
git+git://github.com/JiahuiYu/neuralgym.git@88292adb524186693a32404c0cfdc790426ea441
```

You can also think of the `Dockerfile` as a very long list of installation instructions. There are additional docker keywords for config setting e.g. which network ports should be available etc.

We will later also talk about strategies for finding which versions / packages / cuda ... match your ML models and go well together.

### Build the DeepFill container

```bash
docker build -t deepfill:v0 .
```

**deepfill** is the name of our container and **v0** is our version tag

During the installation process Docker prints out what is going on. Keep your eyes open for red lines (errors). If everything goes well you can run the container now.

### Run the DeepFill container

```bash
docker run -it --runtime=nvidia --volume $(pwd)/:/shared --workdir /shared deepfill:v0 bash
```

- **-it** and **bash** run the container interactive, container should provide a bash terminal prompt

- **--runtime=nvidia** container can access to GPU

- **--volume** mount the current folder (the DeepFill repo) to folder /shared in the docker container filesystem. The folder is shared between the host and the docker container

- **deepfill:v0** run a docker container named deepfill with version v0

### Download pretrained DeepFill models

Download the [pretrained models](https://github.com/JiahuiYu/generative_inpainting#pretrained-models) e.g. Places2 (places background) or CelebA-HQ (faces) and copy it to folder `model_logs`. The demo relies on it.

### Run the DeepFill demo in the container

Copy these two images the DeepFill repo folder:

| input.png           | mask.png          |
| ------------------- | ----------------- |
| ![input](input.png) | ![mask](mask.png) |

Paste the command below in the terminal of our running DeepFill container:

```bash
python3 test.py --image input.png --mask mask.png --output output.png --checkpoint_dir model_logs/release_places2_256_deepfill_v2/
```

The terminal will return a lot of debugging infos ... don't bother. After a few seconds you should get this result:

| output.png                          |
| ----------------------------------- |
| ![0001](assets/deepfill-output.png) |

Yay 🎉🎉🎉 ! 

## 4. Strategies for finding the requirements 🤯

To be honest it can take quite a while to figure out the requirements. Yes it is a mess. These strategies helped me figuring it out a bit less painful:

- Try to define the dependecy as specific as possible, as there is a steady release of updates which not always go well together. You want to be able to run your container a few weeks later:
  - Bad: pip3 install tensorflow-gpu
  - Good: pip3 install tensorflow-gpu==1.8.0
  - Bad: FROM nvidia/cuda
  - Good: FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
- Guess a cuda version which matches the age of the core ML libaray used e.g:
  - Tensorflow 2.0 → cuda:10.2-cudnn7
  - Tensorflow 1.8.0 → cuda:9.0-cudnn7
  - If it is unclear, start with the new ones and gradually move back in time
  - All available docker containers published by Nvidia can be found here: https://hub.docker.com/r/nvidia/cuda/
- Google error message in combination with specfic tensorflow / cuda versions e.g. `FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04` and then running the demo gave me `ImportError: libcublas.so.9.0: cannot open shared object file: No such file or directory`. After reading a few posts it turned out that this is a typical error which can be avoided using cuda 9.0.
- If there is a requirements state e.g. tensorflow-gpu==1.7.0 but you still have problems e.g. `E tensorflow/stream_executor/cuda/cuda_dnn.cc:396] Loaded runtime CuDNN library: 7603 (compatibility version 7600) but source was compiled with 7005 (compatibility version 7000)` . Try to gently bump up or down the version. In the case of DeppFill using tensorflow-gpu==1.8.0 solved the issue.

## 5. Acknowledgments 🙏

- Article section around how to install Docker is based on the [Install nvidia-docker](https://github.com/opendatacam/opendatacam/blob/master/documentation/nvidia-docker/INSTALL_NVIDIADOCKER.md) guide from the [OpenDataCam](https://github.com/opendatacam/opendatacam) project
- Cover image based on [OpenMojis](https://openmoji.org/) – the open-source emoji and icon project. License: CC BY-SA 4.0
- DeepFill demo images are from the movie "Fast and Furious" (2019) ... and my silly experiments [Fast and the Furious without guys and cars](https://twitter.com/bndktgrs/status/1204425598510227463)

