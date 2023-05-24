import os
from pathlib import Path
from pydicom import dcmread
from patchify import patchify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.sparse import csc_matrix
from scipy.io import savemat
from TOA import createForwMatdotdetLBW, CreateFilterMatrix
from downgrade_sinograms import addnoise


TYPE = "LQ"

if TYPE == "GT":
    Ns = 256
    dx = 50e-6
    nx = 401
    DIR = False
elif TYPE =="LQ":
    Ns = 128
    dx = 50e-6
    nx = 401
    DIR = False #This was implemented later directly on the sinogram
elif TYPE =="RECON":
    Ns = 256
    dx = 100e-6
    nx = 201
    DIR = False

Nt = 512
dsa = 22.5e-3
arco = 360
vs = 1500
to = 6.2e-6
tf = to + 17e-6

A_b = createForwMatdotdetLBW(Ns,Nt,dx,nx,dsa,arco,vs,to,tf,DIR)

#%%
snr_list = [10, 20, 30, 40, 50, 60, 70]
#TRAIN VAL TEST DATASET GENERATION

DATASETS = ["train", "val", "test"]


for DATASET in DATASETS:
    images_paths_txt = f"/PAT_GAN/scripts/data_curation/curated_{DATASET}_relpath.txt"
    with open(images_paths_txt, 'r') as fp:
        file_paths = fp.readlines()
        #file_paths.sort()

    if TYPE == "GT" or TYPE == "LQ":
        save_dir =f"/PAT_GAN/datasets/breast_phantoms/{DATASET}/{DATASET}_{TYPE.lower()}/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        print('\nFolder Created', save_dir)
    else:
        print('\nAlready exists', save_dir)

    print(f"{DATASET} and {TYPE}")

    for file in file_paths:
        file = file[:-1]
        file = "/PAT_GAT/datasets/original_dataset/" + file
        orig_path = Path(file)
        filename = orig_path.stem

        orig_img = dcmread(file)
        orig_img = orig_img.pixel_array
        #Downsize image to fit the system's visual field
        orig_img = np.array(Image.fromarray(orig_img).resize((401,401)))
        #Normalize min to 0 and max a 1
        orig_img = (orig_img-np.min(orig_img))/(np.max(orig_img)-np.min(orig_img))
        #Convert image to column sparse matrix
        img = csc_matrix(orig_img.flatten())
        sinogram_array = A_b @ img.transpose()
        sinogram_array = sinogram_array.toarray()

        if TYPE == "GT":
            sino = sinogram_array.reshape((256,512))
            sino = sino/np.amax(sino)

        elif TYPE == "LQ":
            sino = sinogram_array.reshape((128,512))
            MDIR = CreateFilterMatrix(Nt,to,tf)
            MDIR = MDIR.astype(np.float32)
            sino =  sino @ MDIR
            sino = Image.fromarray(sino)
            sino = sino.resize((512, 256))
            sino = np.array(sino)
            snr_index = randint(0,6)
            sino, _ = addnoise(sino, snr_list[snr_index], 'peak')
            sino = sino/np.amax(sino) 

        sino = sino.astype(np.float32)
        patches = patchify(sino, (64, 64), step=32)

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                num = i * patches.shape[1] + j
                patch = patches[i, j]
                d = {"sino":patch}
                save_path = f"{save_dir}{filename}_{num}.mat"
                savemat(save_path, d)

#%%

#TESTING
test_dir = "PAT_GAN/datasets/testing_images/images_gt/"
save_dir = f"PAT_GAT/datasets/testing_images/sino_{TYPE.lower()}/"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    print('\nFolder Created', save_dir)
else:
    print('\nAlready exists', save_dir)

test_path = Path(test_dir)
test_files = list(test_path.iterdir())

for file in test_files:
    filename = Path(file).stem
    image = Image.open(file)
    image = image.convert("L")
    orig_img = image.resize((401,401))
    img = np.array(orig_img)
    img = img.astype(np.float32)

    #normalize
    img = img/255
    # change background to 0 and object to 1
    img = 1 - img

    print(np.amin(img))
    print(np.amax(img))
    plt.imshow(img)
    plt.show()

    img = csc_matrix(img.flatten())
    sinogram_array = A_b @ img.transpose()
    sinogram_array = sinogram_array.toarray()

    if TYPE == "GT":
        sino = sinogram_array.reshape((256,512))
        sino = sino/np.amax(sino) #Normalize max to 1
        plt.imshow(sino)
        plt.show()

        d_sino={"sino":sino}
        save_path = f"{save_dir}{filename}_sino.mat"
        savemat(save_path, d_sino)


    elif TYPE == "LQ":
        sino = sinogram_array.reshape((128,512))
        MDIR = CreateFilterMatrix(Nt,to,tf)
        MDIR = MDIR.astype(np.float32)
        sino = sino @ MDIR
        sino = Image.fromarray(sino)
        sino = sino.resize((512, 256))
        sino = np.array(sino)
        sino = sino/np.amax(sino)

        sino = sino.astype(np.float32)
        patches = patchify(sino, (64, 64), step=32)

        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                num = i * patches.shape[1] + j
                patch = patches[i, j]
                d = {"sino":patch}
                save_path = f"{save_dir}{filename}_{num}.mat"
                savemat(save_path, d)