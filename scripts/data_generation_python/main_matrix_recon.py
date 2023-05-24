from TOA import createForwMatdotdetLBW, CreateFilterMatrix
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix # Column sparse
from scipy.io import savemat, loadmat
import os

def imfrommat(img_bytes):
    f = loadmat(img_bytes)
    if "sino" in f:
        sino = f["sino"]
    elif "pred" in f:
        sino = f["pred"]
    elif "full_sino" in f:
        sino = f["full_sino"]
    else:
        raise Exception("Error opening mat file, incorrect field name") 
    sino = np.array([sino])
    return sino


TYPE = "RECON"

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


LOAD_PATH = "/PAT_GAN/results/res_pretrained/"
SAVE_PATH = "/PAT_GAN/results/res_pretrained_recon/"

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    print('\nFolder Created', SAVE_PATH)
else:
    print('\nAlready exists', SAVE_PATH)

load_path = Path(LOAD_PATH)
files = list(load_path.glob("*.mat"))

for file in files: 
    sino = imfrommat(file)
    orig_path = Path(file)
    filename = orig_path.stem

    #plt.imshow(sino[0])
    #plt.show()

    sino_array = csc_matrix(sino.flatten())
    img_array = sino_array @ A_b
    img_array = img_array.toarray()
    img = img_array.reshape((201,201))
    if np.amax(img) != 0:
        img = img/np.amax(img)

    #plt.imshow(img)
    #plt.show()

    d_sino = {"img": img}
    save_path = f"{SAVE_PATH}{filename}.mat"
    savemat(save_path, d_sino)