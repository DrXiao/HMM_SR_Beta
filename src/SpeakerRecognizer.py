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
    return mfcc(signal, samplerate=sample_rate, nfft=1103)
  
def prepareCorpusData(speaker_corpus_path_name: Path) -> Any:
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
    mfcc_data = preprocessing(signal, sample_rate)
    return mfcc_data

def prepareSpeakerDataset(speaker_corpus_path: Path) -> list:
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
        dataset.append(prepareCorpusData(
            speaker_corpus_path / corpus_name))
    return dataset

def prepareAllSpeakersDataset(dataset_path: Path) -> dict:
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
        speakers_dataset[speaker_label] = prepareSpeakerDataset(
            dataset_path / speaker_corpus_path)

    return speakers_dataset


class SpeakerRecognizer:
    
    def __init__(self):
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
        pass

            
    def __validateTestDataset(self, test_dataset: dict):
        pass

    def __recognize(self, corpus_mfcc_data: np.ndarray) -> int:
        """
        Utility:
                Given MFCC series, recognizing the label of the corpus.
        Input:
                corpus_mfcc_data: MFCC series. (the signal converted to MFCC series.)
        Output:
                The label of the corpus.
        """
        return 0

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
    obj = SpeakerRecognizer()
    