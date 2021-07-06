from pathlib import Path
import os
from scipy.fftpack import dct
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
import numpy as np
import librosa
import wave
import matplotlib.pyplot as plt
import math
import pickle

dataset_path = Path('dataset/')

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
    return dataset


def prepareAllSpeakersDataset():
    global dataset_path
    label_dirs = os.listdir(dataset_path)
    speakers_test_dataset = {}
    for speaker_corpus_path in label_dirs:
        speaker_label = int(speaker_corpus_path)
        speakers_test_dataset[speaker_label] = prepareSpeakerDataset(
            dataset_path / speaker_corpus_path)

    return speakers_test_dataset

class SpeechRecognizer:
    def __init__(self):
        self.model_param = Path("model_param")
        self.createModels()
    def createModels(self):
        self.hmm_models = {}
        self.kmeans_centers = []
        models_pkl = [ model_file for model_file in os.listdir(self.model_param) if '.pkl' in model_file]
        for model_file in models_pkl:
           label = int(model_file.split(".pkl")[0])
           with open(self.model_param / model_file, 'rb') as file:
               self.hmm_models[label] = pickle.load(file)
        self.kmeans_centers = np.load(self.model_param / 'kmeans_param.npy').tolist()
    def validateTestDataset(self, test_dataset):
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
                        feature[corpus_idx][j] - self.kmeans_centers[0])
                    predict_label = 0
                    for centers_idx in range(len(self.kmeans_centers)):
                        if np.linalg.norm(feature[corpus_idx][j] - self.kmeans_centers[centers_idx]) < dic_min:
                            dic_min = np.linalg.norm(
                                feature[corpus_idx][j] - self.kmeans_centers[centers_idx])
                            predict_label = centers_idx
                    test_data_label.append(predict_label)
                test_data_label = np.array([test_data_label])
                # print(test_data_label)
                score_list = {}
                for model_label in self.hmm_models:
                    model = self.hmm_models[model_label]

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



if __name__ == '__main__':
    obj = SpeechRecognizer()
    speakers_test_dataset = prepareAllSpeakersDataset()
    obj.validateTestDataset(speakers_test_dataset)