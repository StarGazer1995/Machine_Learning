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
    dict1=dict(zip(data.id,data.label))
    files=file_name(img_path)
    labels=[];img=[]
    for i in files[:batch_size]:
        labels.append(dict1.get(i))
        del dict1[i]
        filename=img_path+"\\"+i+".tif"
        img.append(np.array(cv.imread(filename)))
    return np.asarray(img), np.asarray(labels)

def file_name(path):
    filename=[]
    for root,dire,files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1]=='.tif':
                filename.append(str(os.path.splitext(file)[0]))
    return filename

