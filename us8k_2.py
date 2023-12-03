import os
import pickle
import librosa
import numpy as np
import pandas as pd
import librosa.display
from keras.utils import to_categorical

# In[ ]:
    
#mode = 'mfcc'
mode = 'log-mel'
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 96
N_MFCC = 14
deltas = True

# In[ ]:
    
# read metadata
with open('dataset_df.pickle','rb') as f:
     dataset_df = pickle.load(f)

# In[ ]:
# extract feature
i = 0

features = []
labels = []
for filepath in dataset_df['filepath']:
    y, sr = librosa.load(filepath, res_type='kaiser_fast')
    
    if mode == 'stft':
       feature = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length)
    elif mode == 'mel':
       feature = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    elif mode == 'log-mel':
       feature = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)   
       feature = librosa.power_to_db(feature, ref=np.max)
    elif mode == 'mfcc':
       feature = librosa.feature.mfcc(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mfcc=N_MFCC)
       if deltas:
         delta1 = librosa.feature.delta(feature, mode='nearest')
         delta2 = librosa.feature.delta(feature, order=2, mode='nearest')
         feature = np.concatenate((feature, delta1, delta2))            
    features.append(feature)
    labels.append(to_categorical(dataset_df['classID'].iloc[i], 10))    
    i += 1
    if i % 100 == 0:
       print('{} files processed'.format(i))
    
dataset_df['feature'] = features
dataset_df['label'] = labels

# In[ ]:
    
with open('feature_log-mel_96_df.pickle','wb') as f:
     pickle.dump(dataset_df, f)



