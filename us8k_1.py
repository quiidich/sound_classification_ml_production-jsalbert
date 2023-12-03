import os
import numpy as np
import pandas as pd
import mutagen
import mutagen.wave
import pickle

# In[ ]:
    
LENGTH = 4
BITS = 8

# In[ ]:
    
# read metadata
dataset_df = pd.read_csv('metadata/UrbanSound8K.csv')

# In[ ]:
    
# add more columns
filepaths = []
lengths = []
bits = []
for i, row in dataset_df.iterrows():
    filepath = os.path.join('audio','fold'+str(row['fold']),row['slice_file_name'])
    f = mutagen.wave.WAVE(filepath)
    length =  f.info.length 
    bit = f.info.bits_per_sample
    filepaths.append(filepath)
    lengths.append(length)
    bits.append(bit)            
dataset_df['filepath'] = filepaths
dataset_df['length'] = lengths    
dataset_df['bit'] = bits    

# In[ ]:
    
# keep files of >= LENGTH & > BITS        
cond = dataset_df['length'] >= LENGTH
dataset_df = dataset_df[cond]   
cond = dataset_df['bit'] > BITS
dataset_df = dataset_df[cond]           
dataset_df.head()

# In[ ]:
    
# save metadata
with open('dataset_df.pickle','wb') as f:
     pickle.dump(dataset_df, f)
