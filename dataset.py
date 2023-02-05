import os 
import glob
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset 

import cv2
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, img_list, img_size):
        self.img_list = img_list
        self.img_size = img_size
        self.transforms = T.Compose([
            T.Resize((self.img_size, self.img_size)), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(f'{self.img_list[idx]}')
        print(f"=="*10)
        print(f"numpy shape is {img_bgr.shape}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = Image.fromarray(np.uint8(img_rgb))
        img = self.transforms(img_rgb)
        print(f"tenser type is {type(img)}")
        print(f"tenser shape is {img.shape}")
        print(f"=="*10)
        return {'img_name': self.img_list[idx], 'label' : self.img_list[idx].split('/')[-2], 'img': img}

#if __name__=="__main__":
if __name__=="__main__":
    img_list = sorted(glob.glob('./Food_dataset/train/*/*.jpg'))
    # #print(img_list)
    custom_dataset = CustomDataset(img_list, 224)
    print(custom_dataset[0])
    # img = cv2.imread('./tiger.jpg')
    # print(f"img shape is {img.shape}\n")
    # print(f"img is {img}\n")
    # print(f"img height is {len(img)}\n")
    # img_sliced = img[:,:,:]
    # cv2.imwrite('./tiger_after.jpg', img_sliced)

    