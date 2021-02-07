#!/usr/bin/python3.6

#파이썬 패키지가 설치되어 있는 폴더를 수동으로 설정

import sys
SYS_PATH = '/USER/backup/usr/'
sys.path.append('/USER/backup/challenger/.local/lib/python3.6/site-packages')
sys.path.append(SYS_PATH+'lib/python36.zip')
sys.path.append(SYS_PATH+'lib/python3.6')
sys.path.append(SYS_PATH+'lib/python3.6/lib-dynload')
sys.path.append(SYS_PATH+'lib/python3/dist-packages')
sys.path.append(SYS_PATH+'local/lib/python3.6/dist-packages')
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
import keras
from keras.models import Sequential
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import f1_score
from keras import backend as K
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import glob

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

#Label별로 70000 데이터씩 추출하기

df_0 = df_train[df_train['label']==0].sample(n = 70000,random_state=1)
df_1 = df_train[df_train['label']==1].sample(n = 70000,random_state=1)
df_2 = df_train[df_train['label']==2].sample(n = 70000,random_state=1)
df_3 = df_train[df_train['label']==3].sample(n = 70000,random_state=1)
df_4 = df_train[df_train['label']==4].sample(n = 70000,random_state=1)

df_t = pd.concat([df_0,df_1,df_2,df_3,df_4])
df_t = df_t.sample(frac = 1, random_state = 1) #데이터 shuffle의 기능
df_t = df_t.reset_index()
df_t.drop('index',1,inplace=True)
train_label = np.array(df_t['label'])

img_height = 270
img_width = 480
img_channels = 3

img_dim = (img_height, img_width, img_channels)

base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=img_dim)
base_model.trainable = False #사전학습 하지 않기

input_tensor = Input(shape=img_dim)
bn = BatchNormalization()(input_tensor)
x = base_model(bn)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(5, activation='softmax')(x)
model = Model(input_tensor, output)
test_set = pd.DataFrame()
batch_size = 32
epochs = 4

img_size = (img_height, img_width)
kf = KFold(n_splits=n_fold, shuffle=True)

#recall,precision,f1score 함수

def recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    count_true_positive_false_negative = K.sum(y_target_yn)
    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    return recall


def precision(y_target, y_pred):
    y_pred_yn = K.round(K.clip(y_pred, 0, 1))
    y_target_yn = K.round(K.clip(y_target, 0, 1))
    count_true_positive = K.sum(y_target_yn * y_pred_yn)
    count_true_positive_false_positive = K.sum(y_pred_yn)
    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    return precision


def f1score(y_target, y_pred):
    _recall = recall(y_target, y_pred)
    _precision = precision(y_target, y_pred)
    _f1score = ( 2 * _recall * _precision) / (_recall + _precision+ K.epsilon())
    return _f1score

# 학습 모델

def train_model(model, batch_size, epochs, img_size, x, y, test, n_fold, kf):

    preds_train = np.zeros(len(x), dtype = np.float)
    preds_test = np.zeros(len(test), dtype = np.float)
    class_test = np.zeros((len(test),5), dtype = np.float)
    class_output = []
    df_class = pd.DataFrame(columns=['label'])
    i = 1

    for train_index, test_index in kf.split(x):
        x_train = x.iloc[train_index]
        x_valid = x.iloc[test_index]
        y_train = y[train_index]
        y_valid = y[test_index]

        def train_generator():
            while True:
                for start in range(0, len(x_train), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_train))
                    train_batch = x_train[start:end]
                    for filepath, tag in train_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = img[:,32:107] #이미지를 부분 설정
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch
                    
        def valid_generator():
            while True:
                for start in range(0, len(x_valid), batch_size):
                    x_batch = []
                    y_batch = []
                    end = min(start + batch_size, len(x_valid))
                    valid_batch = x_valid[start:end]
                    for filepath, tag in valid_batch.values:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = img[:,32:107]
                        x_batch.append(img)
                        y_batch.append(tag)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    y_batch = np.array(y_batch, np.uint8)
                    yield x_batch, y_batch

        def test_generator():
            while True:
                for start in range(0, len(test), batch_size):
                    x_batch = []
                    end = min(start + batch_size, len(test))
                    test_batch = test[start:end]
                    for filepath in test_batch:
                        img = cv2.imread(filepath)
                        img = cv2.resize(img, img_size)
                        img = img[:,32:107]
                        x_batch.append(img)
                    x_batch = np.array(x_batch, np.float32) / 255.
                    yield x_batch

        callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ModelCheckpoint(filepath='4epoch_inception.fold_' + str(i) + '.hdf5', verbose=1,save_best_only=True, mode='auto')]

        train_steps = len(x_train) / batch_size
        valid_steps = len(x_valid) / batch_size
        test_steps = len(test) / batch_size

        model = model
        
        model.compile(optimizer=Adam(lr=1e-4), loss='sparse_categorical_crossentropy',
                      metrics = ['accuracy',precision,recall,f1score])

        model.fit_generator(train_generator(), train_steps, epochs=epochs, verbose=1,
                            callbacks=callbacks, validation_data=valid_generator(),
                            validation_steps=valid_steps)

        filepath='4epoch_inception.fold_' + str(i) + '.hdf5'
        model.load_weights(filepath)


        preds_test_fold = model.predict_generator(generator=test_generator(),
                steps=test_steps, verbose=1)[:]

        class_test += preds_test_fold


        print('\n\n')

        i += 1

        if i <= n_fold:
            print('Now beginning training for fold {}\n\n'.format(i))
        else:
            print('Finished training!')


    class_test /= n_fold


    return class_test



test_pred = train_model(model, batch_size, epochs, img_size,df_t,train_label, test_files,n_fold,kf)

test_class = pd.DataFrame(test_pred.argmax(axis=1),columns=['label'])
test_class = test_class['label'].map({0:'Wake',1:'REM',2:'N1',3:'N2',4:'N3'})

test_class.to_csv('4epoch_files_inception.csv',index = None)
