import pandas as pd
import numpy as np 
import os
import cv2 as cv
import time

def load():
    img_path="G:\\Kaggle\\Data Set\\all\\train"
    label_path="G:\\Kaggle\\Data Set\\all\\train_labels.csv"
    set_size=10
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
        filename=img_path+"/"+i+".tif"
        pic=contrast_brightness_change(cv.imread(filename)[16:80,16:80])
        img.append(np.array(pic))
    return np.asarray(img), np.asarray(labels)

def file_name(path):
    filename=[]
    for root,dire,files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1]=='.tif':
                filename.append(str(os.path.splitext(file)[0]))
    return filename

def contrast_brightness_change(picture,contrast=1,brightness=1):
    pic=np.zeros(picture.shape,picture.dtype)
    pic=cv.addWeighted(picture,contrast,pic,brightness,0)
    return pic

def upsampling(times,inputsize):
    size=inputsize.shape
    template=np.ones([times*size[0],times*size[1],size[2]],inputsize.dtype)
    for i in range(0,size[0]-1):
        for j in range(0,size[1]-1):
            template[times*i:times*i-1,times*j:times*j-1]*=inputsize[i,j]
    print(template[0,0]);print(template[0,1])
    print(template[1,0]);print(template[1,1])
    print(inputsize[0,0])

    
start=time.time()
x,y=load()
z=np.array([1,2,3])
dst=np.zeros([128,128,3])
print(cv.pyrUp(x[1],dst=dst)[3,1],x[1,1,1])
#print("It cost "+str(int(time.time()-start))+"s to finish whole process")
#print(np.shape(np.asarray(x)),np.shape(np.asarray(y)))
