from pathlib import Path
import os
from typing import Any
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
from sklearn.cluster import KMeans
import numpy as np
import librosa
import math
import pickle
from hmmlearn import hmm
import pyaudio
import wave

"""
Suggested arrangement of dataset directory.

Explanation of file name.

ex: 00_0_0.wav
        00: the number of a speaker.
        0 (first): the number of a vocabulary.
        0 (second): the number of record that the speaker speaks the vocabulary.

The structure of dataset directory.

dataset (a parent directory of all speakers' directories.)
            |
            |
            |----  00 (a directory containing a speaker's dataset)
            |       |
            |       --------- 00_0_0.wav
            |       --------- 00_0_1.wav
                    .
                    .
                    .
            |----  01
                    |
                    --------- 01_0_0.wav
                    --------- 01_0_1.wav
                    .
                    .
                    .
and so on.
"""



class SpeakerRecognizer:
    
    def __init__(self, train_dataset_path: str = "dataset",  model_param_path: str = ""):
        """
        Utility:
                Constructor of SpeakerRecognizer
        Input:
                model_name: the name of the object of SpeakerRecognizer.
                train_dataset_path: the path of train dataset.
                model_param_path: the path of pkl files.

                If model_param_path is given, it will read the pkl files directly.
                otherwise, it will train the models by reading dataset from the path train_dataset_path/
        Output:
                An object of SpeakerRecognizer.
        """
        self.__hmm_models = {}
        self.__kmeans_centers = []
        if model_param_path:
            print("get model_param")
            self.__readHMM_Models(Path(model_param_path))
        else:
            train_dataset = self.__prepareAllSpeakersDataset(
                Path(train_dataset_path))
            self.__kmeansCenter(train_dataset)
            self.__createHMM_Model(
                train_dataset, self.__kmeans_centers)

    def __preprocessing(self, signal: np.ndarray, sample_rate: int) -> Any:
        """
        Utility:
                Converting audio signal to MFCC.
        Input:
                signal: the audio signal.
                sample_rate: the rate of sampling.
        Output:
                MFCC array.
        """
        return mfcc(signal, samplerate=sample_rate, nfft=1103)
  
    def __prepareCorpusData(self, speaker_corpus_path_name: Path) -> Any:
        """
        Utility:
                Loading one .wav file, getting its signal and sample rate, then
                creating MFCC array by calling __preprocessing.
        Input:
                speaker_corpus_path_name: the path of a certrain .wav file.
        Output:
                MFCC array.
        """
        signal, sample_rate = librosa.load(
            speaker_corpus_path_name, mono=True, sr=None)
        mfcc_data = self.__preprocessing(signal, sample_rate)
        return mfcc_data
   
    def __prepareSpeakerDataset(self, speaker_corpus_path: Path) -> list:
        """
        Utility:
                Preparing dataset of a speaker by calling __prepareCorpusData. 
        Input:
                speaker_corpus_path: the directory of a speaker's .wav files.
        Output:
                dataset of a speaker.
        """
        dataset = []
        print('Reading dataset...')
        all_corpus_list = os.listdir(speaker_corpus_path)
        for corpus_name in all_corpus_list:
            dataset.append(self.__prepareCorpusData(
                speaker_corpus_path / corpus_name))
        return dataset
    
    def __prepareAllSpeakersDataset(self, dataset_path: Path) -> dict:
        """
        Utility:
                Preparing dataset of all speaker.
                Calling __prepareSpeakerDataset iteratively. 
        Input:
                dataset_path: the directory of all speaker's directories.
        Output:
                A dictionary of dataset of all speakers.
        """
        label_dirs = os.listdir(dataset_path)
        speakers_dataset = {}
        for speaker_corpus_path in label_dirs:
            speaker_label = int(speaker_corpus_path)
            speakers_dataset[speaker_label] = self.__prepareSpeakerDataset(
                dataset_path / speaker_corpus_path)

        return speakers_dataset
    
    def __kmeansCenter(self, train_dataset: dict):
        """
        Utility:
                Getting the points of KMeans from dataset.
        Input:
                train_dataset: the dictionary of train dataset.
        Output:
                The points of KMeans.(an array)
                (be an attribute of object.)
        """
        train_data = []
        for label in train_dataset:
            train_data += train_dataset[label]
        train_data = np.vstack(train_data)
        clusters = len(train_dataset.keys()) + 1
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(train_data)
        self.__kmeans_centers = kmeans.cluster_centers_

    def __createHMM_Model(self, train_dataset: dict, kmeans_centers: list) -> None:
        """
        Utility:
                Create hmm models by train_dataset and kmeans_centers.
        Input:
                train_dataset: train dataset
                kmeans_centers: points of kmeans.
        Output:
                hmm models (containing all speakers' models)
                (be an attribute of object.)
        """
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
            self.__hmm_models[train_label] = model
        print('Train finished.')

    def __readHMM_Models(self, model_param_path: Path) -> None:
        """
        Utility:
                Reading pkl files, then creating hmm models.
        Input:
                model_param_path: path of pkl files.
        Output:
                hmm models (containing all speakers' models.)
                points of kmeans.
                (Above of them become an attribute of object, respectively.)
        """
        print("Reading models...")
        models_pkl = [model_file for model_file in os.listdir(
            model_param_path) if '.pkl' in model_file]
        for model_file in models_pkl:
            label = int(model_file.split(".pkl")[0])
            with open(model_param_path / model_file, 'rb') as file:
                self.__hmm_models[label] = pickle.load(file)
        self.__kmeans_centers = np.load(
            model_param_path / 'kmeans_param.npy').tolist()
            
    def __validateTestDataset(self, test_dataset: dict):
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
                        feature[corpus_idx][j] - self.__kmeans_centers[0])
                    predict_label = 0
                    for centers_idx in range(len(self.__kmeans_centers)):
                        if np.linalg.norm(feature[corpus_idx][j] - self.__kmeans_centers[centers_idx]) < dic_min:
                            dic_min = np.linalg.norm(
                                feature[corpus_idx][j] - self.__kmeans_centers[centers_idx])
                            predict_label = centers_idx
                    test_data_label.append(predict_label)
                test_data_label = np.array([test_data_label])
                # print(test_data_label)
                score_list = {}
                for model_label in self.__hmm_models:
                    model = self.__hmm_models[model_label]

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

    def __recognize(self, corpus_mfcc_data: np.ndarray) -> int:
        """
        Utility:
                Given MFCC series, recognizing the label of the corpus.
        Input:
                corpus_mfcc_data: MFCC series. (the signal converted to MFCC series.)
        Output:
                The label of the corpus.
        """
        print('Recognizing...')
        recognition_label = -1
        test_data_label = []
        for idx in range(len(corpus_mfcc_data)):
            dic_min = np.linalg.norm(
                        corpus_mfcc_data[idx] - self.__kmeans_centers[0])
            predict_label = 0
            for centers_idx in range(len(self.__kmeans_centers)):
                if np.linalg.norm(corpus_mfcc_data[idx] - self.__kmeans_centers[centers_idx]) < dic_min:
                    dic_min = np.linalg.norm(
                        corpus_mfcc_data[idx] - self.__kmeans_centers[centers_idx])
                    predict_label = centers_idx
            test_data_label.append(predict_label)
        test_data_label = np.array([test_data_label])
        print(test_data_label)
        score_list = {}
        for model_label in self.__hmm_models:
            model = self.__hmm_models[model_label]

            score = model.score(test_data_label)
            score_list[model_label] = math.exp(score)
        print(score_list)
        recognition_label = max(score_list, key=score_list.get)
        return recognition_label

    def record(self) -> int:
        """
        Utility:
                Recording and recognizing the label of the speaker.
        Input:
                None
        Output:
                The label of the speaker.
        """
        chunk = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 3
        WAVE_OUTPUT_FILENAME = "tmp/tmp.wav"

        microphone_reader = pyaudio.PyAudio()
        stream = microphone_reader.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        frames_per_buffer=chunk)
        
        speech_corpus = []
        print('開始說話...')
        for i in range(0, (RATE//chunk * RECORD_SECONDS)):
            # By read method, it gets audio data from microphone.
            data = stream.read(chunk)   # data will be a segment of audio.
            speech_corpus.append(data)         # Let the segment be put into a list.
        stream.close()
        microphone_reader.terminate()
        
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(microphone_reader.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(speech_corpus))
        wf.close()
        mfcc_data = self.__prepareCorpusData(Path(WAVE_OUTPUT_FILENAME))
        print(mfcc_data.shape)
        return self.__recognize(mfcc_data)



if __name__ == '__main__':
    obj = SpeakerRecognizer(model_param_path="model_param")
    