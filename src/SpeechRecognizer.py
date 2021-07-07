from pathlib import Path
import os
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


class SpeechRecognizer:

    def __init__(self, model_name: str, train_dataset_path: str = "dataset", test_dataset_path: str = "", model_param_path: str = "", test_size=0, testing: bool = False):
        self.__hmm_models = {}
        self.__kmeans_centers = []
        self.__model_name = model_name
        if model_param_path:
            print("get model_param")
            self.__readHMM_Models(Path(model_param_path))
        else:
            dataset = self.__prepareAllSpeakersDataset(
                Path(train_dataset_path))
            if test_dataset_path == "" and test_size:
                train_dataset, test_dataset = {}, {}
                for label in dataset:
                    train_dataset[label], test_dataset[label] = train_test_split(
                        dataset[label], test_size=test_size, random_state=42)
            else:
                train_dataset = dataset
            self.__kmeansCenter(train_dataset)
            self.__createHMM_Model(
                train_dataset, self.__kmeans_centers)
        if (test_dataset_path or test_size) and testing:
            self.__validateTestDataset(test_dataset)
        elif testing:
            print("Error: No any test dataset.")

    def __preprocessing(self, signal: np.ndarray, sample_rate: int):

        return mfcc(signal, samplerate=sample_rate, nfft=1103)

    def __prepareCorpusData(self, speaker_corpus_path_name: Path):
        signal, sample_rate = librosa.load(
            speaker_corpus_path_name, mono=True, sr=None)
        mfcc_data = self.__preprocessing(signal, sample_rate)
        return mfcc_data

    def __prepareSpeakerDataset(self, speaker_corpus_path: Path):
        dataset = []
        print('Reading dataset...')
        all_corpus_list = os.listdir(speaker_corpus_path)
        for corpus_name in all_corpus_list:
            dataset.append(self.__prepareCorpusData(
                speaker_corpus_path / corpus_name))
        return dataset

    def __prepareAllSpeakersDataset(self, dataset_path: Path):
        label_dirs = os.listdir(dataset_path)
        speakers_dataset = {}
        for speaker_corpus_path in label_dirs:
            speaker_label = int(speaker_corpus_path)
            speakers_dataset[speaker_label] = self.__prepareSpeakerDataset(
                dataset_path / speaker_corpus_path)

        return speakers_dataset

    def __kmeansCenter(self, train_dataset: dict):
        train_data = []
        for label in train_dataset:
            train_data += train_dataset[label]
        train_data = np.vstack(train_data)
        clusters = len(train_dataset.keys()) + 1
        kmeans = KMeans(n_clusters=clusters)
        kmeans.fit(train_data)
        self.__kmeans_centers = kmeans.cluster_centers_

    def __createHMM_Model(self, train_dataset: dict, kmeans_centers: list):
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

    def __readHMM_Models(self, model_param_path: Path):
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

    def __recognize(self, corpus_mfcc_data: np.ndarray):
        print('Testing...')
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
        score_list = {}
        for model_label in self.__hmm_models:
            model = self.__hmm_models[model_label]

            score = model.score(test_data_label)
            score_list[model_label] = math.exp(score)
        recognition_label = max(score_list, key=score_list.get)
        return recognition_label
        
    def record(self):
        chunk = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 2
        RATE = 44100
        RECORD_SECONDS = 3
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
        WAVE_OUTPUT_FILENAME = "tmp/tmp.wav"
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
       
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(microphone_reader.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(speech_corpus))
        wf.close()
        mfcc_data = self.__prepareCorpusData(Path(WAVE_OUTPUT_FILENAME))
        return self.__recognize(mfcc_data)

    def getRecognizerName(self):
        return self.__model_name


if __name__ == '__main__':
    obj = SpeechRecognizer("speaker", model_param_path="model_param")
    print(obj.record())