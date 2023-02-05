# -*- coding: utf-8 -*-
import os
import glob
import argparse
import torch 
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet50

#import models

import dataset # ImageFolder가 있어서 생략가능 둘중 하나만 사용하면 됨 


def load_data():
    # case 1. Custom dataset 인스턴스 생성(이미지 1장 단위)
    # train_dataset = CustomDataset(img_list, img_size) #1. dataset의 인자가 뭐뭐 들어갔어야 했지?
    # val_dataset = CustomDataset(img_list, img_size) #1. dataset의 인자가 뭐뭐 들어갔어야 했지?
    
    # case 2. Custom dataset 인스턴스 생성(이미지 1장 단위)
    transform = T.Compose([#T.Resize((self.img_size, self.img_size)), \
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder('./Food_dataset/train/', transform=transform)
    val_dataset = ImageFolder('./Food_dataset/val/', transform=transform)
    label_map_dict = train_dataset.class_to_idx #{'간자장': 0, '가자미식혜': 1}
    label_list = train_dataset.classes # ['간자장', '가자미식혜']

    #Custom dataset으로 dataloader 만들기(배치 단위)
    # %JYP num_workers --> how many subprocesses to use for data loading.
    # %JYP shuffle for val_loader is not necessary
    # 100 images. batch_size = 10. -> 1 epoch 10 iterations
    #   if no shuffle, 1st, 2nd, 3rd, ... 10th batch is always same for every epoch
    #   else (shuffled), 1st, 2nd, 3rd, ... 10th batch is different for every epoch
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0) 
    return train_loader, val_loader

def train(train_loader):
    epochs = 10 
    model = resnet50(pretrained=True)
    #model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for i in range(epochs):
        print(f"ith epoch is {i}")
        for batch_idx, (X_train, y_train) in enumerate(train_loader):
            print(X_train.shape)
            pred = model(X_train)
            loss = criterion(pred, y_train)
            optimizer.zero_grad() #이게 왜 필요하지
            loss.backward()
            optimizer.step()
            print(f"batch_idx is {batch_idx}\t\tloss is {loss}")
            
            
def eval():
    pass

def main(data_dir):
    # 모든 함수 여기서 실행 
    # 1. dataset, dataloader
    # 2. train 
    # 3. loss, optimizer 
    # 4. validation
    # 5. result save(.pth, .txt)  
    #train_dataloader, val_dataloader = data_load()
    #train(train_dataloader, val_dataloader)
    
    pass

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='USAGE')
    parser.add_argument('--data_dir', type=str, help='/home/')
    args = parser.parse_args()

    data_dir = '/home/dacon_ocr/torch_prac/Food_dataset'
    train_loader, val_loader = load_data()
    train(train_loader)    
    #main(data_dir)
