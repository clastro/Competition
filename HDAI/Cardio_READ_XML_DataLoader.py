import pandas as pd
import numpy as np
from ast import literal_eval
import glob
import pandas_read_xml as pdx
from tqdm import tqdm
import base64

xml_normal = glob.glob('./data/train/normal/*.xml')
xml_arr = glob.glob('./data/train/arrhythmia/*.xml')
xml_train = xml_normal + xml_arr

train_array = np.empty((0,8,1500))

#train_array
for xml in tqdm(xml_train[24369:]):
    
    Waveform = pdx.read_xml(xml, ['RestingECG', 'Waveform'])
    Rhythm = Waveform[1]
    df_rhythm = pd.json_normalize(Rhythm)    
    rhythm_Lead = pd.json_normalize(df_rhythm['LeadData'])
    df_RL = pd.DataFrame()
    
    for i in range(len(rhythm_Lead.columns)):
        df_normal = pd.json_normalize(rhythm_Lead[i])
        df_RL = df_RL.append(df_normal)  
    #rhythm_Lead = rhythm_Lead.drop([2,3,4,5])
        
    df_rhythm_sum = pd.concat([df_rhythm.iloc[:,:-1],df_RL],1)
    df_rhythm_sum = df_RL
    df_rhythm_sum.reset_index(drop=True,inplace=True)
    x = np.array(0)
    rhythm_dataset = np.empty((0,1500))
    
    for i in range(len(df_rhythm_sum['WaveFormData'])):
        wave_row = df_rhythm_sum.iloc[i,-1].replace('\n','').replace('\\n', '').replace('\\','').replace("b'",'')
        wave_row_bytes = base64.b64decode(wave_row)
        encoded_vec = np.fromstring(wave_row_bytes, dtype = np.int16)
        rhythm_dataset = np.vstack((rhythm_dataset,encoded_vec[2000:3500]))
    rhythm_dataset = rhythm_dataset.reshape(-1,8,1500)
    
    train_array = np.vstack((train_array,rhythm_dataset))


#다른 포맷

#train_array
for xml in tqdm(xml_train[24369:]):
    
    Waveform = pdx.read_xml(xml, ['RestingECG', 'Waveform'])
    #Rhythm = Waveform#[1]
    #df_rhythm = pd.json_normalize(Rhythm)
    rhythm_Lead = pd.json_normalize(Waveform['LeadData'][0])
    #df_RL = pd.DataFrame()
    
    #for i in range(len(rhythm_Lead.columns)):
    #    df_normal = pd.json_normalize(rhythm_Lead[i])
    #    df_RL = df_RL.append(df_normal)  
    rhythm_Lead = rhythm_Lead.drop([2,3,4,5])
    df_RL = rhythm_Lead
        
    #df_rhythm_sum = pd.concat([df_rhythm.iloc[:,:-1],df_RL],1)
    df_rhythm_sum = df_RL
    df_rhythm_sum.reset_index(drop=True,inplace=True)
    x = np.array(0)
    rhythm_dataset = np.empty((0,1500))
    
    for i in range(len(df_rhythm_sum['WaveFormData'])):
        wave_row = df_rhythm_sum.iloc[i,-1].replace('\n','').replace('\\n', '').replace('\\','').replace("b'",'')
        wave_row_bytes = base64.b64decode(wave_row)
        encoded_vec = np.fromstring(wave_row_bytes, dtype = np.int16)
        
        if(len(encoded_vec)<1500):
            print(xml)
            continue
        rhythm_dataset = np.vstack((rhythm_dataset,encoded_vec[2000:3500]))
    if(len(rhythm_dataset)<8):
        print(xml)
        continue
    rhythm_dataset = rhythm_dataset.reshape(-1,8,1500)
    
    
    train_array = np.vstack((train_array,rhythm_dataset))

arr_reshaped = train_array.reshape(train_array.shape[0], -1)
np.save('./x_save_6784', arr_reshaped)
#save_file = np.load('./x_save.npy')
