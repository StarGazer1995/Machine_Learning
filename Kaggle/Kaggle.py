import os
import cv2
import numpy as np
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import img_load as img
from densenet121 import DenseNet
#data load
img,labels=img.load()
#data augment

#data train
img = np.expand_dims(img, axis=0)

model = DenseNet(reduction=0.5, classes=2, weights_path=None)
filepath='checkpoint.h5'
if os.path.isfile(filepath)==False:
    os.mknod(str(filepath))
    
Checkpoint=ModelCheckpoint(filepath,monitor='val-loss',verbose=1,save_best_only=True,mode=min)
train_history=model.fit(img,labels,epoch=30,callbacks=[Checkpoint],batch_size=16)
sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
