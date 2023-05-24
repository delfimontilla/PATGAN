import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from loss import *
from unet import *

START_FILTER = 32
IMG_CHANNELS = 1
IMG_HEIGHT= 512                                                                                    
IMG_WIDTH = 512
WEIGHTS_PATH = '../results/benchmark/model_benchmark.pth'
TEST_PATH = '/PAT_GAN/datasets/testing_images/sino_lq/'
SAVE_PATH = '/PAT_GAN/datasets/testing_images/sino_unet/'
lst = os.listdir(TEST_PATH)
n_samples = len(lst)//2

if not os.path.exists(SAVE_PATH):
	os.mkdir(SAVE_PATH)
	print('\nFolder Created', SAVE_PATH)
else:
	print('\nAlready exists', SAVE_PATH)


# Load Model
device = torch.device('cpu')
model = UNet(START_FILTER)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model = torch.nn.DataParallel(model, device_ids=list(
	range(torch.cuda.device_count()))).cuda()
model.eval()

limited = 'full_sino'
files = os.listdir(TEST_PATH)

for file in files:
	load_path = TEST_PATH + file
	# Load image and add reflective padding
	print(file)
	mat = sio.loadmat(load_path)
	test_img = np.asarray(mat[limited], dtype='float32')
	#plt.imsave('in_sino.png', test_img, cmap='Greys')
	test_img = np.pad(test_img, [(128,128),(0,0) ], mode = 'reflect')
	#plt.imsave('in.png', test_img, cmap='Greys')
	test_img = test_img.reshape(1, IMG_HEIGHT, IMG_WIDTH)
	# print('test_img np: ',test_img)
	test_img = torch.FloatTensor(test_img)
	# print('test_tensor',test_img.shape)
	print('\n')


	with torch.no_grad():
		# print('test image: ',test_img)
		model_out = Variable(test_img.unsqueeze(0).cuda())
		# print('model_out: ',model_out)
		model_output = model(model_out)
		# print('model_out: ',model_output)

	pred_out = np.asarray(model_output.cpu().detach().numpy(), dtype = 'float32').reshape(512, 512)
	#plt.imsave('pred.png', pred_out, cmap='Greys')
	pred_out = pred_out[128:384, :]
	#plt.imsave('pred_crop.png', pred_out, cmap='Greys')
	save_name = file.split('.')[0] + '_pred.mat'
	sio.savemat(SAVE_PATH + '/' + save_name, {"pred": pred_out})
	print('Saving mat file...', save_name)
