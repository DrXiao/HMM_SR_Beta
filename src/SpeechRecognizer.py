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
    speaker_dataset, speaker_cmd_dataset = {}, {}
    speaker_datasetV, speaker_cmd_datasetV = {}, {}


    dataset_path = Path("features")

    print(os.listdir(dataset_path))

    for speaker_label in os.listdir(dataset_path):
        speaker_dataset[int(speaker_label)] = []
        speaker_datasetV[int(speaker_label)] = []
        speaker_cmd_dataset[int(speaker_label)] = {}
        speaker_cmd_datasetV[int(speaker_label)] = {}
        for cmd_label in os.listdir(dataset_path / speaker_label):
            speaker_cmd_dataset[int(speaker_label)][int(cmd_label)] = []
            speaker_cmd_datasetV[int(speaker_label)][int(cmd_label)] = []
            count = 0
            speaker_corpus_dir = os.listdir(dataset_path / speaker_label / cmd_label)
            random.shuffle(speaker_corpus_dir)
            for corpus in speaker_corpus_dir:
                
                with open(dataset_path / speaker_label / cmd_label / corpus, "rb") as corpus_file:
                    corpus_feature = pickle.load(corpus_file)
                if count == 3:
                    speaker_cmd_datasetV[int(speaker_label)][int(
                        cmd_label)].append(corpus_feature)
                    speaker_datasetV[int(speaker_label)].append(corpus_feature)
                else:
                    speaker_cmd_dataset[int(speaker_label)][int(
                        cmd_label)].append(corpus_feature)
                    count += 1
            speaker_dataset[int(speaker_label)].append(corpus_feature)

    speaker_recoginzer, speaker_cmd_recoginzer = None, {}
    speaker_recoginzer = SpeechRecognizer("SpeakerRecoginzer", speaker_dataset)
    for speaker_label in speaker_cmd_dataset:
        speaker_cmd_recoginzer[speaker_label] = SpeechRecognizer(
            "%d_Cmd_Recoginzer" % (speaker_label), speaker_cmd_dataset[speaker_label])
        print(speaker_cmd_recoginzer[speaker_label].getModelName())
    print()
    
    print("Speaker Validation")
    print(speaker_recoginzer.validate(speaker_datasetV))
    print("Cmd Validation")
    for speaker_label in speaker_cmd_recoginzer:
        print(speaker_label ,speaker_cmd_recoginzer[speaker_label].validate(speaker_cmd_datasetV[speaker_label]))
    
    print()

    total_count = 0
    correct_count = 0
    for speaker_label in speaker_cmd_datasetV:
        for cmd_label in speaker_cmd_datasetV:
            for corpus in speaker_cmd_datasetV[speaker_label][cmd_label]:
                speaker = speaker_recoginzer.recoginze(corpus)
                cmd = speaker_cmd_recoginzer[speaker].recoginze(corpus)
                # print("Prediction is (%d, %d), Truth is (%d, %d)"%(speaker, cmd, speaker_label, cmd_label))
                if speaker == speaker_label and cmd == cmd_label:
                    correct_count += 1
                total_count += 1
    print("Final recoginze rate :", 100 * correct_count / total_count)

if __name__ == "__main__":
    main()
