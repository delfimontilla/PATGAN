import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn as nn
from scipy.io import loadmat, savemat
import os
from loss import *
from unet import *
from Load_Data import extract_patches, reconstruct_from_patches
import matplotlib.pyplot as plt


# Testing the model over numerical and experimental data stored in 'Test_data/' 
print('#'*11)
print('# Testing #')
print('#'*11)

IMG_CHANNELS = 1
IMG_HEIGHT= 64                                                                                 
IMG_WIDTH = 64
START_FILTERS = 32 # Starting with these many filters in u-net 
WEIGHTS_PATH = '../experiments/benchmark/model_benchmark_test.pth' 
TEST_PATH = '/home/delfina/github-repos/gpu_test/datasets/breast_phantoms/test/test_lq/'
SAVE_PATH = "../experiments/benchmark/res_model_test/"
lst = os.listdir(TEST_PATH)
n_samples = len(lst)//2

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
    print('\nFolder Created', SAVE_PATH)
else:
    print('\nAlready exists', SAVE_PATH)

model = UNet(f = START_FILTERS)# start_filter = START_FILTERS
model.load_state_dict(torch.load(WEIGHTS_PATH))

# Load Model
device = torch.device('cpu')
model.eval()
files = os.listdir(TEST_PATH)

def imfrommat(img_bytes):
    f = loadmat(img_bytes)
    if "sino" in f:
        sino = f["sino"]
    elif "limited_noise_interpolated" in f:
        sino = f["limited_noise_interpolated"]
    else:
        raise Exception("Error opening mat file, incorrect field name")
    sino = np.array(sino)
    return sino



for file in files:
    if (file.startswith('original')):
        continue
    load_path = TEST_PATH + file
    print(file)
    test_image = imfrommat(load_path)
    #plt.imshow(test_image, "gray")
    #plt.show()
    test_image = Image.fromarray(test_image)
    test_image = test_image.resize((512, 200), resample = Image.Resampling.NEAREST)
    test_image = np.array(test_image)
    patches = extract_patches(test_image)
    test_images = np.expand_dims(patches, axis=3)

    outputs = []
    for test_img in test_images:
        test_img = test_img.reshape(1, 1, IMG_HEIGHT, IMG_WIDTH)
        test_img = torch.Tensor(test_img)
        with torch.no_grad():
            model_out = Variable(test_img) 
            model_out = model(model_out)
        model_out = np.asarray(model_out.cpu().detach().numpy(), dtype = 'float32').reshape(64,64)
        outputs.append(model_out)

    recon = reconstruct_from_patches(np.array(outputs))
    #plt.imshow(recon, "gray")
    #plt.show()
    save_name = file.split('.')[0] + '_pred.mat'
    savemat(SAVE_PATH + '/' + save_name, {"unet": recon})
    print('Saving mat file...', save_name)
