from random import shuffle
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class DataGenerator(data.Dataset):
    def __init__(self,annotation_lines,mode="Train",input_shape_US=[650,650],input_shape_TH=[240,240]):
        self.annotation_lines = annotation_lines
        self.input_shape_US = input_shape_US
        self.input_shape_TH = input_shape_TH
        if mode == "Train":
            self.trans = transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        else:
            self.trans = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
    
    def __len__(self):
        return len(self.annotation_lines)
    
    def __getitem__(self,index):
   
        annotation_path_US = self.annotation_lines[index].split(';')[1].strip()
        
        annotation_path_TH = annotation_path_US.replace('_US','_TH')
        annotation_path_TH = annotation_path_TH.replace('.jpg','.png')
       

        image_US = Image.open(annotation_path_US)
        image_TH = Image.open(annotation_path_TH)

        image_US = self.trans(image_US)
        image_TH = self.trans(image_TH)

        y = int(self.annotation_lines[index].split(';')[0])

        return image_US,image_TH,y


    
        
        