import os
# from librosa.feature import mfcc
from python_speech_features import mfcc
import pickle
import librosa 
from pathlib import Path


'''
utility
  提取mfcc特徵值(目前未調整參，直接採用librosa的mfcc套件)
input
  要做mfcc的檔案路徑
output
  特徵值
'''
def mfcc_process(path):
    
    audio, sample_freq = librosa.load(path, mono=True, sr=None)
    mfcc_features = mfcc(audio, samplerate=sample_freq, nfft=2048) # 提取MFCC特徵
    # print(f"Shape: {mfcc_features.shape}")
    #mfcc_features = mfcc_features.transpose() #維度0(mfcc coefisient係數) 和1(time)交換
    
    '''
    繪製特徵圖
    
    mfcc_features = mfcc_features.T 
    plt.matshow(mfcc_features) 
    plt.title('MFCC')
    '''
    return mfcc_features

def main():
    

    for label in [0, 1, 2, 3, 6]:
      # os.mkdir('features/%02d/'%(label))
      voicePath = Path('dataset/%02d/'%(label))
      featurePath = Path('features/%02d/'%(label))
      fileList = [fileName for fileName in os.listdir(voicePath) if os.path.splitext(fileName)[1] == '.wav']

      for fileName in fileList:
          print(f"Processing {fileName}")
          label = fileName.split("_")[1]
          feature = mfcc_process(voicePath / fileName)

          outFileName = featurePath / str(label) / (os.path.splitext(fileName)[0]+'.pickle')
          with open(outFileName, 'wb') as f:
            pickle.dump(feature, f)

if __name__=="__main__":
    main()