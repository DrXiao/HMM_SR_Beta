from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from hmmlearn import hmm
from python_speech_features import mfcc
import numpy as np
import librosa
import math
import pickle

dataset_path = Path('dataset/matdb_Wav')
model_params = Path('model_param/')
test_size = 100


def preprocessing(signal, sample_rate):

    return mfcc(signal, samplerate=sample_rate, nfft=2048)


"""

Utility:
            將一個語者 Dataset 切成 training set / testing set
Input:
            speaker_corpus_path : .wav file path of a spekaer.
Output:
        
            train_dataset   : training dataset
            test_dataset    : testing dataset
            All of above will be returned by train_test_split function

"""


def prepareSpeakerDataset(speaker_corpus_path):
    global test_size
    dataset = []
    print('Reading dataset...')
    all_corpus_list = os.listdir(speaker_corpus_path)
    for corpus_name in all_corpus_list:
        signal, sample_rate = librosa.load(
            speaker_corpus_path / Path(corpus_name),  sr=None)
        dataset.append(preprocessing(signal, sample_rate))
    return train_test_split(dataset, test_size=test_size, random_state=42)


"""
Utility:
        讀取所有語者的 dataset
Input:  
        None
Output:
        所有語者的 train_dataset, test_dataset
"""


def prepareAllSpeakersDataset():
    global dataset_path
    label_dirs = os.listdir(dataset_path)
    speakers_train_dataset, speakers_test_dataset = {}, {}
    for speaker_corpus_path in label_dirs:
        speaker_label = int(speaker_corpus_path)
        speakers_train_dataset[speaker_label], speakers_test_dataset[speaker_label] = prepareSpeakerDataset(
            dataset_path / speaker_corpus_path)

    return speakers_train_dataset, speakers_test_dataset


"""
Utility:
        利用 train_dataset，訓練出 kmeans centers
Input:
        train_dataset : all speakers' dataset (dictionary)
Output:
        kmeans centers (2D list)
"""


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


def createHMM_Model(train_dataset, kmeans_centers):
    hmm_model = {}
    states_num = 6
    print('Training models...')
    for train_label in train_dataset:
        model = hmm.MultinomialHMM(
            n_components=states_num, n_iter=20, algorithm='viterbi', tol=0.01)
        train_data = train_dataset[train_label]
        train_data = np.vstack(train_data)
        train_data_label = []
        for corpus_idx in range(len(train_data)):
            dic_min = np.linalg.norm(
                train_data[corpus_idx] - kmeans_centers[0])
            label = 0
            for centers_idx in range(len(kmeans_centers)):
                if np.linalg.norm(train_data[corpus_idx] - kmeans_centers[centers_idx]) < dic_min:
                    dic_min = np.linalg.norm(
                        train_data[corpus_idx] - kmeans_centers[centers_idx])
                    label = centers_idx
            train_data_label.append(label)

        train_data_label.append(len(kmeans_centers) - 1)
        train_data_label = np.array([train_data_label])
        # print(train_data_label)
        model.fit(train_data_label)
        hmm_model[train_label] = model
    print('Train finished.')
    return hmm_model


"""
Utility:
        測試模型的準確度，並儲存模型參數
Input:
        hmm_models : hmm 模型 (dictionary)
        kmeans_centers : kmeans center points (list)
        test_dataset : 要測試的 dataset (dictionary)
Output:
        print the recognition rate
        Save the parameters of models and kmeans centers
"""


def testing(hmm_models, kmeans_centers, test_dataset):
    print('Testing...')
    true = []
    pred = []
    score_cnt = 0
    corpus_num = 0
    for test_label in test_dataset:
        feature = test_dataset[test_label]
        corpus_num += len(feature)
        for corpus_idx in range(len(feature)):
            test_data_label = []
            for j in range(len(feature[corpus_idx])):
                dic_min = np.linalg.norm(
                    feature[corpus_idx][j] - kmeans_centers[0])
                predict_label = 0
                for centers_idx in range(len(kmeans_centers)):
                    if np.linalg.norm(feature[corpus_idx][j] - kmeans_centers[centers_idx]) < dic_min:
                        dic_min = np.linalg.norm(
                            feature[corpus_idx][j] - kmeans_centers[centers_idx])
                        predict_label = centers_idx
                test_data_label.append(predict_label)
            test_data_label = np.array([test_data_label])
            # print(test_data_label)
            score_list = {}
            for model_label in hmm_models:
                model = hmm_models[model_label]

                score = model.score(test_data_label)
                score_list[model_label] = math.exp(score)
            predict_label = max(score_list, key=score_list.get)
            # print(score_list)
            # print("Test on true label ", test_label, ": predict result label is ", predict_label)
            if test_label == predict_label:
                score_cnt += 1
            true.append(test_label)
            pred.append(predict_label)
    #print("true:", true, "pred:", pred, sep='\n')
    rate = 100.0 * score_cnt/corpus_num
    print("Final recognition rate is %.2f%%" %
          (rate))

    global test_size
    with open('%d_%d.txt' % (100 - test_size * 100, test_size * 100), 'a') as result:
        result.write("Final recognition rate is %.2f%%\n" %
                     (rate))
    for speaker_label in hmm_models:
        with open(model_params / Path('%02d.pkl' % (speaker_label)), 'wb') as file:
            pickle.dump(hmm_models[speaker_label], file)
    kmeans_centers_np = np.array(kmeans_centers)
    np.save(model_params / 'kmeans_param.npy', kmeans_centers_np)


def createGMMHMM(train_dataset):
    hmm_model = {}
    states_num = 6
    print('Training models...')
    for train_label in train_dataset:
        model = hmm.GaussianHMM(
            n_components=states_num, n_iter=20, algorithm='viterbi', tol=0.01)
        # print(train_data_label)
        train_data = train_dataset[train_label]
        train_data = np.vstack(train_data)
        model.fit(train_data)
        hmm_model[train_label] = model
    print('Train finished.')
    return hmm_model


def testingGMMHMM(hmm_models, test_dataset):
    print('Testing...')
    global test_size
    true = []
    pred = []
    score_cnt = 0
    corpus_num = 0
    for test_label in test_dataset:
        feature = test_dataset[test_label]
        corpus_num += len(feature)
        for corpus_idx in range(len(feature)):

            # print(test_data_label)
            score_list = {}
            for model_label in hmm_models:
                model = hmm_models[model_label]
                score_list[model_label] = model.score(feature[corpus_idx])
            predict_label = max(score_list, key=score_list.get)
            if test_label == predict_label:
                score_cnt += 1
            else:
                print(score_list)
                print("Test on true label ", test_label,
                        ": predict result label is ", predict_label)
            true.append(test_label)
            pred.append(predict_label)
    #print("true:", true, "pred:", pred, sep='\n')
    rate = 100.0 * score_cnt/corpus_num
    print("Final recognition rate is %.2f%%" %
            (rate))
    



# def main():
    
    #kmeans_centers = kmeansCenter(speakers_train_dataset)
    #hmm_models = createHMM_Model(speakers_train_dataset, kmeans_centers)
    #testing(hmm_models, kmeans_centers, speakers_test_dataset)
    


def main():
    pass

if __name__ == "__main__":
    main()