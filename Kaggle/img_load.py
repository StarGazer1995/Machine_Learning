import pandas as pd
import numpy as np 
import os
import cv2 as cv
from skimage import io


def load():
    img_path="G:\\Kaggle\\Data Set\\all\\train"
    label_path="G:\\Kaggle\\Data Set\\all\\train_labels.csv"
    set_size=30000
    train_set,train_label=img_load(img_path,label_path,set_size)
    return train_set,train_label

def img_load(img_path,target_path,batch_size):
    data=pd.read_csv(target_path)
    target_label=data.label
    filename=img_path+"\\"+data.id+".tif"
    img=[]
    for i in range(0,batch_size-1):
        img.append(np.array(cv.imread(filename[i])))
    return img, target_label[0:batch_size-1]

