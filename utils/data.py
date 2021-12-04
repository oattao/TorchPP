import os
import torch
import cv2 as cv
import numpy as np
from .signal import trim_signal


class CNN1DImprovementDataset(torch.utils.data.Dataset):
    def __init__(self, data, signal_length, transforms=None):
        self.signal_length = signal_length
        self.transforms = transforms
        self.data = {}
        self.midx = [0]
        self.num_samples = 0

        for label, signal in data.items():
            trimed_signal, max_num_samples = trim_signal(signal, signal_length)
            self.data[label] = trimed_signal
            self.num_samples += max_num_samples
            self.midx.append(self.num_samples)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Find range of the idx
        label = 0
        for i in range(len(self.midx) - 1):
            if self.midx[i] <= idx < self.midx[i+1]:
                label = i
                break
        smallidx = idx - self.midx[label]
        signal = self.data[label][smallidx*self.signal_length: (smallidx+1)*self.signal_length]
        signal = np.expand_dims(signal, axis=0)
        signal = torch.Tensor(signal)

        return signal, label

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, label_class_dict, channel=cv.IMREAD_GRAYSCALE, transforms=None):
        super(ImageDataset, self).__init__()
        self.image_list = image_list
        self.label_class_dict = label_class_dict
        self.channel = channel
        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = cv.imread(image_path, self.channel)
        label = image_path.split(os.path.sep)[-3]
        label = self.label_class_dict[label]

        if self.transforms is not None:
            image = self.transforms(image)

        image = image / 255
        return image, label

    def __len__(self):
        return len(self.image_list)

class DoubleImageDataset(torch.utils.data.Dataset):
    def __init__(self, double_image_list, label_class_dict, transforms=None):
        super(DoubleImageDataset, self).__init__()
        self.double_image_list = double_image_list
        self.label_class_dict = label_class_dict
        self.transforms = transforms

    def __getitem__(self, idx):
        gray_path, scalo_path = self.double_image_list[idx]
        gray_image = cv.imread(gray_path, cv.IMREAD_GRAYSCALE)
        scalo_image = cv.imread(scalo_path, cv.IMREAD_GRAYSCALE)
        label = gray_path.split(os.path.sep)[-2]
        label = self.label_class_dict[label]

        if self.transforms:
            gray_image = self.transforms(gray_image)
            scalo_image = self.transforms(scalo_image)

        return scalo_image, gray_image, label

    def __len__(self):
        return len(self.double_image_list)

