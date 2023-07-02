# Photoacoustic Tomography & GAN

📖 Paper abstract - The goal of this work is to study a preprocessing method for the data measured by a two-dimensional optoacoustic tomograph in order to reduce or eliminate artifacts introduced by the low number of detectors in the experimental setup and their limited bandwidth. A generative adversarial deep neural network was used to accomplish this task and its performance was compared with a reference U-Net neural network. In most of the test cases carried out, a slight improvement was found by applying the proposed network when measuring the Pearson correlation and the peak signal noise ratio between the reconstructed image product of the data processed by the model and the high-resolution reference image. Keywords: optoacustic tomography; machine learning; GAN.

📖 [[Full article](http://elektron.fi.uba.ar/index.php/elektron/article/view/180)] 

Super resolution, noise removal and BW enhancement for photoacoustic tomography sinograms using the [Real Enhanced Super Resolution GAN](https://github.com/xinntao/Real-ESRGAN) architecture and the training pipeline from [Basic Super Resolution](https://github.com/XPixelGroup/BasicSR). The original code was adapted to satisfy this project requirements. The neural network used as benchmark is the U-Net implemented on [Deep Neural-Network Based Sinogram Super-resolution and Bandwidth Enhancement for Limited Data Photoacoustic Tomography](https://ieeexplore.ieee.org/document/9018129), [code](https://sites.google.com/site/sercmig/home/dnnpat) available.


## Prerequisites:

 - Python 3.8.10 or higher with specified packages in pyproject.toml.

Optionally:

- [Pyenv](https://github.com/pyenv/pyenv): tool that lets you easily switch between multiple versions of Python.

- [Poetry](https://python-poetry.org/docs/): tool for dependency management and packaging in Python.
    
- [Docker](https://www.docker.com/): an open platform for developing, shipping, and running applications. 

Installation instructiones for Ubuntu at the end.


### Directory structure

    .
    ├───Real-ESRGAN
    │   ├───datasets
    │   │   ├───breast_phantoms
    │   │   │   ├───meta_info
    │   │   │   ├───test
    │   │   │   ├───train
    │   │   │   └───val
    │   │   ├───original_dataset -> DICOM MRIs
    │   │   └───testing_images
    │   ├───docs
    │   ├───experiments
    │   │   └───pretrained_models
    │   ├───inputs
    │   ├───options
    │   ├───realesrgan
    │   │   ├───archs
    │   │   ├───data
    │   │   ├───models
    │   │   └───weights
    │   ├───results
    │   ├───scripts
    |   ├── tb_logger
    │   └───tests
    ├───results
    ├───scripts
    │   ├───data_analysis
    │   ├───data_curation
    │   └───data_generation_python
    └───UNET_benchmark
        └───util
 
## Training Real-ESRGAN

The training has been divided into two stages:

1. First pre-train the Real-ESRNet architecture with L1 and perceptual loss.



     poetry run python train.py -opt options/train_realesrnet_x2_patches.yml



2. Use the pre-trained Real-ESRNet model as an initialization of the generator, and train the Real-ESRGAN with a combination of L1 loss, perceptual loss and GAN loss.


    poetry run python train.py -opt options/train_realesrgan_x2_patches.yml



Training parameters configuration:

- options/*.yml

In case of training from scratch, use the train_*.yml files; otherwise, to finetune a trained model use the finetune_*.yml files. Inside train_realesrnet_2x_half.yml, most parameters are followed by a short comment as a way to help future users understand what they mean.

[RRDBNet](https://github.com/XPixelGroup/BasicSR/blob/f6b3790537e8e5da225cd3c018bf407e8e4519b4/basicsr/archs/rrdbnet_arch.py#L67): networks consisting of Residual in Residual Dense Block, which is used as a generator, where it's first employed the process of pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size and enlarge the channel size before feeding inputs into the main Real-ESRGAN architecture).

[UNetDiscriminatorSN](https://github.com/xinntao/Real-ESRGAN/blob/35ee6f781e9a5a80d5f2f1efb9102c9899a81ae1/realesrgan/archs/discriminator_arch.py#L8): Defines a U-Net discriminator with spectral normalization (SN). 


### Dataset Preparation

For this project the RealESRGANPairedDataset class was used (defined in ``realesrgan/data/realesrgan_paired_dataset.py``). It reads LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs. This class was modified to be able to load .mat as input files instead of image (.png, .jpeg, etc.) files.


For the paired dataset, you have to prepare a txt file containing the image paths. You can use the ``scripts/generate_meta_info.py`` script to generate the txt file. Example output from ``metadata_train.txt``:

     train_gt/data_1000_01.mat, train_lq/data_1000_01.mat
     train_gt/data_1000_010.mat, train_lq/data_1000_010.mat
     train_gt/data_1000_02.mat, train_lq/data_1000_02.mat
     train_gt/data_1000_03.mat, train_lq/data_1000_03.mat
     
     
For example:

    poetry run python generate_meta_info_pairdata.py --input ../datasets/breast_phantoms/train/train_gt/ ../datasets/breast_phantoms/train/train_lq/ --meta_info ../datasets/breast_phantoms/meta_info/metadata_train_patches.txt
 

This script also filters all the null examples (images where the mean value is equal to zero).



## Outputs

Results will be placed in the ``experiments`` directory, under the name specified on the yaml that was used for training/finetuning. 

- models: *.pth pytorch models
- training states: *.state files containing all training parameters related to that iteration.
- visualization: if validation was done and save_img is True, here will be placed the processed lq validation images in each vallidation iteration. So the training progress can be visually tracked.

Training metrics are logged in a tensorboard event file found in the ``tb_logger`` directory, it can be displayed running

    poetry run tensorboard --logdir=tb_logger/train_RealESRGANx2_patches


# Inference on sinograms

For example, to run inference on sinograms with the trained model at the 1M epoch, 32bit precision (default is 16bit) and output extension .mat:


        poetry run python inference_realesrgan.py -i datasets/test/ -o results/ -n sinogram --epoch 1000000 -s 2 --fp32 --ext mat --sinogram --patches



### Docker container 

    docker build -t PAT_GAN:v0 .
    docker run --gpus all -it --shm-size=8gb --env="DISPLAY" PAT_GAN:v0


### Results


# Reconstructing images

First, patches have to reconstructed to a full sized sinogram of 256 x 512 px.


    PAT_GAN/scripts/data_generation_python$ poetry run python patches2sino.py


Then, those sinograms have to be multiplied by the reconstruction matrix.

    PAT_GAN/scripts/data_generation_python$ poetry run python main_matrix_recon.py


# Evaluating performance

Compare ground truth sinograms and images to inferences using four performance metrics: Pearson covariance, structural similarity index measure (SSIM), root mean-square error (RMSE) and peak signal noise ratio (PSNR).


    PAT_GAN/scripts/data_analysis$ poetry run python evaluate_performance.py 


And plot results changing the fname variable to the path correspoing to the resulting csv of the previous analysis:


     PAT_GAN/scripts/data_analysis$ poetry run python plot_results.py 



## Optional Installations

### Poetry installation in Ubuntu

Run the following command on the terminal to download and install poetry:

    curl -sSL https://install.python-poetry.org | python3 -

In the ./bashrc file found at /home/, add the following line at the end:
  
    export PATH="/home/$USER/.local/bin:$PATH"


### Pyenv installation 

Run the following command on the terminal to download and install pyenv with its dependecies:

    curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
    sudo apt update; sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev wget curl llvm libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev

In the ./bashrc file found at /home/, add the following lines at the end:

    export PYENV_ROOT="$HOME/.pyenv"
    command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"


Run the following command on the terminal to install the requiered Python version: 

    pyenv install 3.10 

In project directory, activate the Python version and associated to the poetry environment, then install all the required packages specified in the pyproject:
    
    pyenv local 3.10
    poetry env use python3.10
    poetry install


[Docker cheatsheet](https://dockerlabs.collabnix.com/docker/cheatsheet/)
