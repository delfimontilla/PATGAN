from patchify import unpatchify
import numpy as np
from scipy.io import savemat, loadmat
import os
from natsort import natsorted
from pathlib import Path
import matplotlib.pyplot as plt


SINO_HEIGHT = 256
SINO_WIDTH = 512
PATCH_HEIGHT = 64
PATCH_WIDTH = 64
STEP = 32
output_height = SINO_HEIGHT - (SINO_HEIGHT - PATCH_HEIGHT) % STEP
output_width = SINO_WIDTH - (SINO_WIDTH - PATCH_WIDTH) % STEP
output_shape = (output_height, output_width)

LOAD_PATH="/PAT_GAN/results/res_nopretraining_patches/"
SAVE_PATH ="/PAT_GAN/results/res_nopretraining/"


if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    print('\nFolder Created', SAVE_PATH)
else:
    print('\nAlready exists', SAVE_PATH)

files = list(os.listdir(LOAD_PATH))
files = natsorted(files)
for k in range(0, len(files), 105):
    patches_files = files[k:k+105]
    filename = Path(patches_files[0]).stem
    filepath = SAVE_PATH + filename[:-2] + ".mat"

    patches_list = []
    for file in patches_files:
        file = LOAD_PATH + file
        print(file)
        data = loadmat(file)
        patches_list.append(data['realesrgan'])
    
    output_patches = np.empty((7, 15, 64, 64))
    q = 0
    for i in range(7):
        for j in range(15):
            output_patches[i, j] = patches_list[q]
            q += 1

    full_sinogram = unpatchify(output_patches, output_shape)
    f={"full_sino":full_sinogram}
    savemat(filepath, f)


