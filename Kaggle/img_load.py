import pandas as pd 
import os
import scipy as misc
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
    img_index=list(get_tif(img_path))
    img=[]
    for i in range(0,batch_size-1):
        img.append(io.imread(img_index[i]))
    return img, target_label[0:batch_size-1]

def get_tif(path):
    return(os.path.join(path,f) for f in os.listdir(path) if f.endswith('.tif'))