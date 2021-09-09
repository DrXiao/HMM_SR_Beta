import os
from sklearn.model_selection import train_test_split
import librosa
from python_speech_features import mfcc
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

dataset_path = Path("dataset/")
# testPath06 = os.getcwd() + '/06/'
test_size = 0.98


def buildDataset():
    # 切割檔名
    global test_size
    trainData, testData = [], []
    trainDataset, testDataset = {}, {}
    for speaker in os.listdir(dataset_path):
        for label in range(8):
            if not trainDataset.get(label):
                trainDataset[label] = []
                testDataset[label] = []
            mfcc_data = [librosa.load(dataset_path / speaker / file)
                         for file in os.listdir(dataset_path / speaker) if file.find("_%d_" % (label)) != -1]
            mfcc_data = [mfcc(corpus[0], samplerate=corpus[1], nfft=1103) for corpus in mfcc_data]
            
            trainData, testData = train_test_split(
                mfcc_data, test_size=test_size)
            trainDataset[label] += trainData
            testDataset[label] += testData
    
    return trainDataset, testDataset


'''
utility
  訓練模型
input
  dataset
output
  訓練完的hmm模型
'''


def HMMTrain(dataset):

    hmmModels = {}

    for label in dataset.keys():  # 每個label都去train一個model

        # Prepare dataset
        trainData = np.array(dataset[label])
        dataSetLength = [i.shape[0] for i in trainData]  # 取時間的長度
        # print([i.shape for i in trainData])
        trainData = np.concatenate(trainData, axis=0)  # 用維度0(時間)去接
        # print(trainData)
        hmmModel = hmm.GaussianHMM(n_components=8)  # 建立model的架構
        # hmmModel = hmm.MultinomialHMM(n_components=12) # 建立model的架構

        hmmModel.fit(trainData, lengths=dataSetLength)  # 訓練model(猜他應該是只能切維度0)

        hmmModels[label] = hmmModel

    return hmmModels


'''
# utility
#   測試準確率
# input
#   testDataset, hmmModels
# output
#   準確率
'''


def dataTest(testDataset, hmmModels):
    correctCnt = 0  # 記錄對的次數
    totalCnt = 0  # 記錄總比數
    for trueLabel in testDataset.keys():
        singleTestDataset = testDataset[trueLabel]
        for testData in singleTestDataset:
            print("True label: {}; Predict Score: ".format(trueLabel), end='')

            prob_list = {}
            for predLabel in hmmModels.keys():
                model = hmmModels[predLabel]
                score = model.score(testData)

                # Output the result
                print("{:.02f} ".format(score), end='')
                prob_list[int(predLabel)] = score
            idx = 0
            for i in (prob_list):
                if prob_list[i] > prob_list[idx]:
                    idx = i
            if(idx) == int(trueLabel):
                correctCnt = correctCnt+1

            totalCnt = totalCnt+1

            print("predict result :", int(idx) == int(trueLabel))

    print(round(correctCnt*100/totalCnt, 3), '%')
    with open("02_train02_test98.txt", "a") as file:
        file.write("%f%%\n" % (round(correctCnt*100/totalCnt, 3)))


def main():
    for i in range(5):
        trainDataset, testDataset = buildDataset()
        print("Finish prepare the training data\n")
        print("Model Training...\n")
        hmmModels = HMMTrain(trainDataset)
        # 測試準確率
        print("\nStart testing....\n")
        dataTest(testDataset, hmmModels)


if __name__ == '__main__':
    main()
