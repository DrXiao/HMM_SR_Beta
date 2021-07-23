from pathlib import Path
import os
from numpy.core.fromnumeric import shape
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from hmmlearn import hmm
from scipy.fftpack import dct
from python_speech_features import mfcc
import numpy as np
import librosa
import matplotlib.pyplot as plt
import math
import pickle
from tensorflow.python.keras.backend import dropout

from tensorflow.python.util.tf_export import TENSORFLOW_API_NAME
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPool2D
from keras.utils import to_categorical
import pickle
from keras.models import load_model

dataset_path = Path('dataset/')
model_params = Path('model_param')
test_size = 0.4

"""

Utility:
            將語者 Dataset 切成 training set / testing set
Input:
            speaker_corpus_path : .wav file path of a spekaer.
Output:
        
            train_dataset   : training dataset
            test_dataset    : testing dataset
            All of above will be returned by train_test_split function

"""


def preprocessing(signal, sample_rate):
    
    return mfcc(signal, samplerate=sample_rate, nfft=1103)


def prepareSpeakerDataset(speaker_corpus_path):
    global test_size
    dataset = []
    print('Reading dataset...')
    all_corpus_list = os.listdir(speaker_corpus_path)
    for corpus_name in all_corpus_list:
        signal, sample_rate = librosa.load(
            speaker_corpus_path / Path(corpus_name), mono=True, sr=None)
        dataset.append(preprocessing(signal, sample_rate))
    return train_test_split(dataset, test_size=test_size, random_state=42)


def prepareAllSpeakersDataset():
    global dataset_path
    label_dirs = os.listdir(dataset_path)
    speakers_train_dataset, speakers_test_dataset = {}, {}
    for speaker_corpus_path in label_dirs:
        speaker_label = int(speaker_corpus_path)
        speakers_train_dataset[speaker_label], speakers_test_dataset[speaker_label] = prepareSpeakerDataset(
            dataset_path / speaker_corpus_path)

    return speakers_train_dataset, speakers_test_dataset


def kmeansCenter(train_dataset):
    train_data = []
    for label in train_dataset:
        train_data += train_dataset[label]
    train_data = np.vstack(train_data)
    clusters = len(train_dataset.keys()) + 1
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(train_data)
    centers = kmeans.cluster_centers_
    return centers


"""
createCNN_Models
"""
def createCNN_Model(x_train,x_test,y_train_hot,y_test_hot,x_train_shape):
    cnn_model = Sequential()
    cnn_model.add(Conv2D(32,kernel_size=(2,2),activation='relu', input_shape=(299, 13, 1)))
    cnn_model.add(MaxPool2D(pool_size=(2,2)))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))
    cnn_model.add(Dropout(0.25))
    cnn_model.add(Dense(7,activation='softmax'))
    cnn_model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    
    cnn_model.fit(x_train,y_train_hot,batch_size=100,epochs=100,verbose=2,validation_data=(x_test,y_test_hot))
    score = cnn_model.evaluate(x_test,y_test_hot,verbose=1)
    print(score)
    cnn_model.save('model.h5')
    return

def dataset_spliting(train_dataset,test_dataset):
    x_train = np.asarray(train_dataset.get(0))
    x_test = np.asarray(test_dataset.get(0))
    #for i in range(1,6):
        #x_train = np.vstack([x_train,train_dataset.get(i)])
        #x_test = np.vstack([x_test,test_dataset.get(i)])
    y_train = np.zeros(x_train.shape[0])
    y_test = np.zeros(x_test.shape[0])

    for i in range(1,7):
        if i != 4:
            print(i)
            x_train_input = np.asarray(train_dataset.get(i))
            x_test_input = np.asarray(test_dataset.get(i))
            print(x_train_input.shape)
            print(x_test_input.shape)
            x_train = np.vstack([x_train,x_train_input])
            x_test = np.vstack([x_test,x_test_input])
            y_train = np.concatenate((y_train,np.full(x_train_input.shape[0],fill_value=i)), axis=0)
            y_test = np.concatenate((y_test,np.full(x_test_input.shape[0],fill_value=i)), axis=0)
            print("-----------------")
    x_train = x_train.reshape(x_train.shape[0],299,13,1)
    x_test = x_test.reshape(x_test.shape[0],299,13,1)
    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)
    #x_train = np.expand_dims(x_train,axis=3)
    #x_test = np.expand_dims(x_test,axis=3)
    #y_train = np.expand_dims(y_train,axis=3)
    #y_test = np.expand_dims(y_test,axis=3)
    
    
    #print("x_train = ",x_train)
    #print("x_test = " ,x_test)
    print("x_train shape = ",x_train.shape)
    print("x_test shape = ",x_test.shape)
    print("y_train = ",y_train.shape)
    print("y_test = ",y_test.shape)
    print("y_train_hot.shape = " ,y_train_hot.shape)
    print("y_test_hot.shape = " ,y_test_hot.shape)
    return x_train,x_test,y_train_hot,y_test_hot,x_train.shape



def main():
    speakers_train_dataset, speakers_test_dataset = prepareAllSpeakersDataset()
    #speakers_train_dataset = {0:open('00.pkl','wb'),1:open('01.pkl','wb'),2:open('02.pkl','wb'),3:open('03.pkl','wb'),4:open('04.pkl','wb'),5:open('05.pkl','wb'),6:open('06.pkl','wb')}
    #kmeans_centers = kmeansCenter(speakers_train_dataset)
    x_train,x_test,y_train_hot,y_test_hot,x_train_shape = dataset_spliting(speakers_train_dataset,speakers_test_dataset)
    cnn_models = createCNN_Model(x_train,x_test,y_train_hot,y_test_hot,x_train_shape)
    


if __name__ == '__main__':
    #for size in range(40, 100, 20):
        size = 90
        test_size = size / 100
        main()