import os
import numpy as np
import pandas as pd
import cv2 as cv
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.applications.densenet import DenseNet201,preprocess_input
#img_load and preprocessing
def load():
    img_path="/home/popzq/dataset/train/"
    label_path="/home/popzq/train_labels.csv"
    set_size=220000
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
def contrast_brightness_change(picture,contrast=1.2,brightness=1):
    pic=np.zeros(picture.shape,picture.dtype)
    pic=cv.addWeighted(picture,contrast,pic,brightness,0)
    return pic
#data load
img,labels=img.load()
img=np.asarray(img)
labels=np.asarray(labels)
#some parameters
ratio=0.8
s=np.int(labels.shape[0]*ratio);s1=labels.shape[0]-s
img_train=img[:s];labels_train=labels[:s]
img_val=img[s:];labels_val=labels[s:]
number_of_class=2
epoch=200
batch_size=8
#image generate and agumention
train_generator=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_generator=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
#model define
    ##model input
train_gene=train_generator.flow(
    img_train,
    y=labels_train,
    batch_size=batch_size
)
validation_gene=val_generator.flow(
    img_val,
    y=labels_val,
    batch_size=batch_size
)
def toplayers(model,classes):
    x=model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(classes*2,activation='relu')(x)
    predictions=Dense(classes,activation='sigmoid')(x)
    model_with_toplayer=Model(input=model.input,output=predictions)
    return model_with_toplayer

model = DenseNet201(include_top=False)    
model=toplayers(model,classes=1)
model.load_weights('/home/popzq/kaggle/checkpoint/checkpoint--14e-val_acc0.94.hdf5')
model.compile(
    optimizer=SGD(lr=0.001,momentum=0.9,decay=0.0001,nesterov=True),
    loss='binary_crossentropy',
    metrics=['accuracy'])

output_model_path='/home/popzq/kaggle/checkpoint/checkpoint--{epoch:02d}e-val_acc{val_acc:.2f}.hdf5'
Checkpoint=ModelCheckpoint(output_model_path,monitor='val_acc',verbose=1,save_best_only=True,mode=max)
RUN=RUN+1 if 'RUN' in locals() else 1
LOG_DIR='/home/popzq/kaggle/log'
tensorboard=TensorBoard(log_dir=LOG_DIR,write_images=True)

history=model.fit_generator(
    train_gene,
    samples_per_epoch=s,
    nb_epoch=epoch,
    callbacks=[tensorboard,Checkpoint],
    validation_data=validation_gene,
    validation_steps=s1   
)
