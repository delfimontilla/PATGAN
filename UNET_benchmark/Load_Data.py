# make sure pickle is imported
import pickle
from pathlib import Path
from scipy.io import loadmat
import numpy as np
from PIL import Image


class Dataset(object):
    """An abstract class representing a Dataset.
    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class Dataset_sino(Dataset):
    
    def __init__(self, file_path_x, file_path_y, transform=None):
        print('Loading Data....')
        with open(file_path_x, 'rb') as x:
            X = pickle.load(x)
        print('X loaded.')
        with open(file_path_y, 'rb') as y:
            Y = pickle.load(y)
        print('Y loaded.\n')

        self.data_x = X
        self.data_y = Y
        self.transform = transform
        
    def __len__(self):
        return len(self.data_x)
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)
        train_image = self.data_x[index, :, : ,:]
        test_image = self.data_y[index, :, :, :]
        
        if self.transform is not None:
            train_image = self.transform(train_image)
            test_image = self.transform(test_image)
            
        return train_image, test_image


class Dataset_sino_bp(Dataset):
    
    def __init__(self, file_path_x, file_path_y, transform=None):
        print('Loading Data....')
        files_lq = Path(file_path_x)
        files_lq = list(files_lq.glob("*.mat"))
        files_lq.sort()
        print('X loaded.')
        if len(files_lq)<1:
            raise Exception('Not enough files')
        self.files_lq = files_lq
        
        print('Loading Data....')
        files_gt = Path(file_path_y)
        files_gt = list(files_gt.glob("*.mat"))
        files_gt.sort()
        print('Y loaded.')
        if len(files_gt)<1:
            raise Exception('Not enough files')
        self.files_gt = files_gt

        if len(files_gt) != len(self.files_lq):
            raise Exception('Different amount of files')

        self.transform = transform

        test_image = imfrommat(self.files_gt[0])
        patches = extract_patches(test_image)
        test_images = np.expand_dims(patches, axis=3)

        train_image = imfrommat(self.files_lq[0])
        patches = extract_patches(train_image)
        train_images = np.expand_dims(patches, axis=3)

        for index in range(1,len(self.files_gt)):
            test_image = imfrommat(self.files_gt[index])
            patches = extract_patches(test_image)
            patches = np.expand_dims(patches, axis=3)
            test_images = np.vstack((test_images, patches))

            train_image = imfrommat(self.files_lq[index])
            train_image = np.array(train_image)
            patches = extract_patches(train_image)
            patches = np.expand_dims(patches, axis=3)
            train_images = np.vstack((train_images, patches))
        
        self.data_x = train_images
        self.data_y = test_images
        
    def __len__(self):
        return len(self.files_gt)*32
    
    def __getitem__(self, index):
        # load image as ndarray type (Height * Width * Channels)
        # be carefull for converting dtype to np.uint8 [Unsigned integer (0 to 255)]
        # in this example, i don't use ToTensor() method of torchvision.transforms
        # so you can convert numpy ndarray shape to tensor in PyTorch (H, W, C) --> (C, H, W)

        train_image = self.data_x[index, :, : ,:]
        test_image = self.data_y[index, :, :, :]

        if self.transform is not None:
            train_image = self.transform(train_image)
            test_image = self.transform(test_image)
            
        return train_image, test_image

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

def extract_patches(img):
    patches
    return patches

def reconstruct_from_patches(patches):
    img 
    return img