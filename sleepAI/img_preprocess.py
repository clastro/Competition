#이미지 벡터화

#!/usr/bin/python3.6

import cv2
import numpy as np
import pandas as pd
from itertools import chain

CSV_PATH = '/DATA/'

TRAIN_CSV = pd.read_csv(CSV_PATH + 'trainset-for_user.csv', encoding='utf-8')
TEST_CSV = pd.read_csv(CSV_PATH + 'testset-for_user.csv', encoding='utf-8')

TRAIN_CSV.columns = ['folder','file','label']
TEST_CSV.columns = ['folder','file']

df_train = pd.DataFrame(columns=['name'])
df_test = pd.DataFrame(columns=['file_name'])

df_train['name'] = CSV_PATH + TRAIN_CSV['folder']+ "/" + TRAIN_CSV['file']
df_test['file_name'] = CSV_PATH + TEST_CSV['folder'] + "/" + TEST_CSV['file']

test_files = np.array(df_test['file_name'])

train_label = TRAIN_CSV['label'].map({'Wake':0,'REM':1,'N1':2,'N2':3,'N3':4})
df_train['label'] = train_label

train_label = np.array(df_train['label'])
img_height = 270
img_width = 480
img_channels = 1

n_fold = 5
img_size = (img_height, img_width)

train_batch = []
test_batch = []

img_dim = (img_height, img_width, img_channels)

batch_size = 100
epochs = 3

def batch_file(name):
    img = cv2.imread(name,cv2.IMREAD_GRAYSCALE) #이미지 흑백만 사용하기
    img = cv2.resize(img, img_size) #이미지 리사이즈
    img = img[:,32:107] #이미지 부분 CROP
    img = list(chain.from_iterable(img)) #이미지를 리스트로 만들기
    return img

df_train.name.apply(lambda x : train_batch.append(batch_file(x)))
df_test.file_name.apply(lambda x : test_batch.append(batch_file(x)))

#train_batch = np.array(train_batch, np.float32) / 255.
#test_batch = np.array(test_batch, np.float32) / 255.

df_train_batch = pd.DataFrame(train_batch)
df_train_batch['label'] = df_train.label

df_train_batch.to_csv('train_batch.csv',encoding='utf-8')
