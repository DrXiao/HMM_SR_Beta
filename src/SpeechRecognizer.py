from pathlib import Path
import os
from typing import Any
from scipy.sparse import data
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
import numpy as np
import librosa
import pickle
from hmmlearn import hmm
import random


def preprocessing(signal: np.ndarray, sample_rate: int) -> Any:
    """
    Utility:
            Converting audio signal to MFCC.
    Input:
            signal: the audio signal.
            sample_rate: the rate of sampling.
    Output:
            MFCC array.
    """
    return mfcc(signal, samplerate=sample_rate, nfft=2048)


class SpeechRecognizer:
    def __init__(self, model_name: str, train_dataset: dict):
        self.__model_name = model_name
        self.__hmm_models = {}
        self.__train(train_dataset)

    def __train(self, train_dataset):
        states_num = 6
        print('Training models...')
        for train_label in train_dataset:
            model = hmm.GMMHMM(
                n_components=states_num, n_iter=20, algorithm='viterbi', tol=0.01)
            # print(train_data_label)
            train_data = train_dataset[train_label]
            train_data = np.vstack(train_data)
            model.fit(train_data)
            self.__hmm_models[train_label] = model
        print('Train finished.')

    def recoginze(self, corpus_features: np.ndarray):
        score_list = {}
        for model_label in self.__hmm_models:
            model = self.__hmm_models[model_label]
            score_list[model_label] = model.score(corpus_features)
        recoginzed_label = max(score_list, key=score_list.get)
        return recoginzed_label

    def validate(self, test_dataset: dict) -> float:
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
                for model_label in self.__hmm_models:
                    model = self.__hmm_models[model_label]
                    score_list[model_label] = model.score(feature[corpus_idx])
                predict_label = max(score_list, key=score_list.get)
                if test_label == predict_label:
                    score_cnt += 1
                else:
                    #print(score_list)
                    #print("Test on true label ", test_label,
                    #      ": predict result label is ", predict_label)
                    pass
                true.append(test_label)
                pred.append(predict_label)
        #print("true:", true, "pred:", pred, sep='\n')
        rate = 100.0 * score_cnt/corpus_num
        return rate

    def getModelName(self):
        return self.__model_name


def main():
    
    dataset_path = Path("matdb_Wav/")
    print(os.listdir(dataset_path))
    for db in os.listdir(dataset_path):
        speaker_dataset, speaker_datasetV = {}, {}
        for lab in os.listdir(dataset_path / db):
            print("Read dataset...")
            for speaker_label in os.listdir(dataset_path / db / lab):
                speaker_dataset[lab + speaker_label] = []
                speaker_datasetV[lab + speaker_label] = []
                speaker_dir = os.listdir(dataset_path / db / lab / speaker_label)
                random.shuffle(speaker_dir)
                flag = 0
                for corpus in speaker_dir:
                    signal, sample_rate = librosa.load(dataset_path / db / lab / speaker_label / corpus)
                    corpus_feature =  preprocessing(signal, sample_rate)
                    if flag == 3:
                        speaker_datasetV[lab + speaker_label].append(corpus_feature)
                    else:
                        speaker_dataset[lab + speaker_label].append(corpus_feature)
                        flag += 1
            print("Reading finish!")
        print(len(speaker_dataset.keys()))
        print(speaker_dataset.keys())
        speaker_recoginzer = SpeechRecognizer("SpeakerRecognizer", speaker_dataset)
        with open("result2.txt", "a") as file:
            print(db, speaker_recoginzer.validate(speaker_datasetV), "%", file=file)

if __name__ == "__main__":
    main()
