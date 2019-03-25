import pandas as pd
import numpy as np
from fastai.vision import *
from sklearn.metrics import roc_auc_score
from torchvision.models import *

model_parh='.'
path='/home/popzq/dataset/'
train_folder=f'{path}train'
test_folder=f'{path}test'
train_label=f'{path}train_labels.csv'
ORG_SIZE=96
batch_size=64
num_workers=None
size=96

labels=pd.read_csv(train_label)
tfms=get_transforms(do_flip=True,
                    flip_vert=True,
                    max_rotate=.0,
                    max_zoom=0.1,
                    max_lighting=0.05,
                    max_warp=0.0
                               )

data=ImageDataBunch.from_csv(
                             path,
                             csv_labels=train_label,
                             folder='train',
                             ds_tfms=tfms,
                             size=size,
                             suffix='.tif',
                             test=test_folder,
                             bs=batch_size)
stats=data.batch_stats()
data.normalize(stats)

def auc_score(y_pred,y_true,tens=True):
    score=roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])
    if tens:
        score=tensor(score)
    else:
        score=score
    return score


learn=cnn_learner(data,
                  densenet201,
                  path='.',
                  metrics=[auc_score],
                  ps=0.5
                  )

learn.fit_one_cycle(40,slice(1.1e-6,1e-3))

pred_test,y=learn.get_preds()
pred_test_tta,y_tta=learn.TTA()
pred_score=auc_score(pred_test,y)
pred_score_tta=auc_score(pred_test_tta,y_tta)

preds_test,y_test=learn.get_preds(ds_type=DatasetType.Test)
preds_test_tta,y_test_tta=learn.TTA(ds_type=DatasetType.Test)


sub=pd.read_csv(f'{path}sample_submission.csv').set_index('id')
clean_fname=np.vectorize(lambda fname:str(fname).split('/')[-1].split('.')[0])
fname_cleaned=clean_fname(data.test_ds.items)
fname_cleaned=fname_cleaned.astype(str)
sub.loc[fname_cleaned,'label']=to_np(preds_test[:,1])
sub.to_csv(f'submission_{pred_score}.csv')
sub.loc[fname_cleaned,'label']=to_np(preds_test_tta[:,1])
sub.to_csv(f'submission_{pred_score_tta}.csv')
