import os
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from xwt import xwt

def DataPreprocessing(data_root):
    # 1. 데이터 읽어오기
    sensor1 = pd.read_csv(os.path.join(data_root,'g1_sensor1.csv'), names=['time', 'normal', 'type1', 'type2', 'type3'])
    sensor2 = pd.read_csv(os.path.join(data_root,'g1_sensor2.csv'), names=['time', 'normal', 'type1', 'type2', 'type3'])
    sensor3 = pd.read_csv(os.path.join(data_root,'g1_sensor3.csv'), names=['time', 'normal', 'type1', 'type2', 'type3'])
    sensor4 = pd.read_csv(os.path.join(data_root,'g1_sensor4.csv'), names=['time', 'normal', 'type1', 'type2', 'type3'])

    #2. Re-Sampling
    x_new = np.arange(0,140,0.001)
    y_new1 = []
    y_new2 = []
    y_new3 = []
    y_new4 = []

    for item in ['normal', 'type1', 'type2', 'type3']:
        f_linear1 = interpolate.interp1d(sensor1['time'], sensor1[item], kind ='linear')
        y_new1.append(f_linear1(x_new))
        f_linear2 = interpolate.interp1d(sensor2['time'], sensor2[item], kind ='linear')
        y_new2.append(f_linear2(x_new))
        f_linear3 = interpolate.interp1d(sensor3['time'], sensor3[item], kind ='linear')
        y_new3.append(f_linear3(x_new))
        f_linear4 = interpolate.interp1d(sensor4['time'], sensor4[item], kind ='linear')
        y_new4.append(f_linear4(x_new))
        
    sensor1 = pd.DataFrame(np.array(y_new1).T, columns=['normal', 'type1', 'type2', 'type3'])
    sensor2 = pd.DataFrame(np.array(y_new2).T, columns=['normal', 'type1', 'type2', 'type3'])
    sensor3 = pd.DataFrame(np.array(y_new3).T, columns=['normal', 'type1', 'type2', 'type3'])
    sensor4 = pd.DataFrame(np.array(y_new4).T, columns=['normal', 'type1', 'type2', 'type3'])

    normal_ = pd.concat([sensor1['normal'], sensor2['normal'], sensor3['normal'], sensor4['normal']], axis=1)
    type1_ = pd.concat([sensor1["type1"], sensor2["type1"], sensor3["type1"], sensor4["type1"]], axis=1)
    type2_= pd.concat([sensor1["type2"], sensor2["type2"], sensor3["type2"], sensor4["type2"]], axis=1)
    type3_= pd.concat([sensor1["type3"], sensor2["type3"], sensor3["type3"], sensor4["type3"]], axis=1)

    normal_.columns = ['s1','s2','s3','s4']
    type1_.columns  = ['s1','s2','s3','s4']
    type2_.columns  = ['s1','s2','s3','s4']
    type3_.columns  = ['s1','s2','s3','s4']


    #3. De-Noising(Gaussian Filtering)
    M = 15
    normal_s1 = np.convolve(normal_['s1'], np.ones(M), 'valid') / M
    normal_s1 = normal_s1.reshape(len(normal_s1),1)
    normal_s2 = np.convolve(normal_['s2'], np.ones(M), 'valid') / M
    normal_s2 = normal_s2.reshape(len(normal_s2),1)
    normal_s3 = np.convolve(normal_['s3'], np.ones(M), 'valid') / M
    normal_s3 = normal_s3.reshape(len(normal_s3),1)
    normal_s4 = np.convolve(normal_['s4'], np.ones(M), 'valid') / M
    normal_s4 = normal_s4.reshape(len(normal_s4),1)
    type1_s1 = np.convolve(type1_['s1'], np.ones(M), 'valid') / M
    type1_s1 = type1_s1.reshape(len(type1_s1),1)
    type1_s2 = np.convolve(type1_['s2'], np.ones(M), 'valid') / M
    type1_s2 = type1_s2.reshape(len(type1_s2),1)
    type1_s3 = np.convolve(type1_['s3'], np.ones(M), 'valid') / M
    type1_s3 = type1_s3.reshape(len(type1_s3),1)
    type1_s4 = np.convolve(type1_['s4'], np.ones(M), 'valid') / M
    type1_s4 = type1_s4.reshape(len(type1_s4),1)
    type2_s1 = np.convolve(type2_['s1'], np.ones(M), 'valid') / M
    type2_s1 = type2_s1.reshape(len(type2_s1),1)
    type2_s2 = np.convolve(type2_['s2'], np.ones(M), 'valid') / M
    type2_s2 = type2_s2.reshape(len(type2_s2),1)
    type2_s3 = np.convolve(type2_['s3'], np.ones(M), 'valid') / M
    type2_s3 = type2_s3.reshape(len(type2_s3),1)
    type2_s4 = np.convolve(type2_['s4'], np.ones(M), 'valid') / M
    type2_s4 = type2_s4.reshape(len(type2_s4),1)
    type3_s1 = np.convolve(type3_['s1'], np.ones(M), 'valid') / M
    type3_s1 = type3_s1.reshape(len(type3_s1),1)
    type3_s2 = np.convolve(type3_['s2'], np.ones(M), 'valid') / M
    type3_s2 = type3_s2.reshape(len(type3_s2),1)
    type3_s3 = np.convolve(type3_['s3'], np.ones(M), 'valid') / M
    type3_s3 = type3_s3.reshape(len(type3_s3),1)
    type3_s4 = np.convolve(type3_['s4'], np.ones(M), 'valid') / M
    type3_s4 = type3_s4.reshape(len(type3_s4),1)

    normal_temp = np.concatenate((normal_s1,normal_s2,normal_s3,normal_s4), axis =1)
    type1_temp = np.concatenate((type1_s1,type1_s2,type1_s3,type1_s4), axis =1)
    type2_temp = np.concatenate((type2_s1,type2_s2,type2_s3,type2_s4), axis =1)
    type3_temp = np.concatenate((type3_s1,type3_s2,type3_s3,type3_s4), axis =1)

    #4. Scaling (Min-Max)
    scaler = MinMaxScaler()
    scaler.fit(normal_)
    normal = scaler.transform(normal_temp)
    type1  = scaler.transform(type1_temp)
    type2  = scaler.transform(type2_temp)
    type3  = scaler.transform(type3_temp)

    #5. Saving to csv
    normal_pd= pd.DataFrame(normal)
    type1_pd = pd.DataFrame(type1)
    type2_pd = pd.DataFrame(type2)
    type3_pd = pd.DataFrame(type3)

    normal_pd.to_csv(os.path.join(data_root, 'normal.csv'), index=False)
    type1_pd.to_csv(os.path.join(data_root, 'type1.csv'), index=False)
    type2_pd.to_csv(os.path.join(data_root, 'type2.csv'), index=False)
    type3_pd.to_csv(os.path.join(data_root, 'type3.csv'), index=False)

    #6. Split to Train Test Val
    normal = normal[30000:130000][:]
    type1  = type1[30000:130000][:]
    type2  = type2[30000:130000][:]
    type3  = type3[30000:130000][:]

    normal_train = normal[:][:60000]
    normal_valid = normal[:][60000:80000] 
    normal_test = normal[:][80000:]
    
    type1_train  = type1[:][:60000]
    type1_valid  = type1[:][60000:80000] 
    type1_test  = type1[:][80000:]

    type2_train  = type2[:][:60000]
    type2_valid  = type2[:][60000:80000]
    type2_test  = type2[:][80000:]

    type3_train  = type3[:][:60000]
    type3_valid  = type3[:][60000:80000]
    type3_test  = type3[:][80000:]

    train = np.concatenate((normal_train,type1_train,type2_train,type3_train))
    valid = np.concatenate((normal_valid,type1_valid,type2_valid,type3_valid))
    test = np.concatenate((normal_test,type1_test,type2_test,type3_test))

    train_pd = pd.DataFrame(train)
    test_pd = pd.DataFrame(test)
    valid_pd = pd.DataFrame(valid)

    train_label = [0 for x in range(60000)] + [1 for x in range(60000)] + [2 for x in range(60000)] + [3 for x in range(60000)]
    test_label = [0 for x in range(20000)] + [1 for x in range(20000)] + [2 for x in range(20000)] + [3 for x in range(20000)]
    valid_label = [0 for x in range(20000)] + [1 for x in range(20000)] + [2 for x in range(20000)] + [3 for x in range(20000)]

    train_pd['class'] = train_label
    test_pd['class'] = test_label
    valid_pd['class'] = valid_label

    train_pd.to_csv(os.path.join(data_root,'train.csv'), index=False)
    valid_pd.to_csv(os.path.join(data_root, 'valid.csv'), index=False)
    test_pd.to_csv(os.path.join(data_root, 'test.csv'), index=False)

class KAMPdataset(Dataset):
    def __init__(self, data_root, window_size, stride, is_uni=False):
        
        data_pd = pd.read_csv(data_root)

        self.dataset = []
        self.is_uni = is_uni

        print(f'building dataset...')
        # make Slice by strid and window_size
        for idx in tqdm(range(0, len(data_pd), stride)):
            start_idx = idx
            end_idx = idx+window_size

            if len(data_pd) < end_idx:
                break
            if data_pd['class'][start_idx] != data_pd['class'][end_idx]:
                continue
                
            sensor1_np = data_pd['0'][start_idx:end_idx].to_numpy()
            sensor2_np = data_pd['1'][start_idx:end_idx].to_numpy()
            sensor3_np = data_pd['2'][start_idx:end_idx].to_numpy()
            sensor4_np = data_pd['3'][start_idx:end_idx].to_numpy()
            class_np = data_pd['class'][start_idx]

            if self.is_uni:
                self.dataset.append([sensor1_np, class_np])
                self.dataset.append([sensor2_np, class_np])
                self.dataset.append([sensor3_np, class_np])
                self.dataset.append([sensor4_np, class_np])
            else:
                self.dataset.append([sensor1_np, sensor2_np ,class_np])
                self.dataset.append([sensor1_np, sensor3_np ,class_np])
                self.dataset.append([sensor1_np, sensor4_np ,class_np])
                self.dataset.append([sensor2_np, sensor3_np ,class_np])
                self.dataset.append([sensor2_np, sensor4_np ,class_np])
                self.dataset.append([sensor3_np, sensor4_np ,class_np])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx, is_biwavelet=True):
        
        if self.is_uni:
            data = self.dataset[idx]
            vib = torch.tensor(data[0], dtype=torch.float64)
            state = torch.tensor(data[-1], dtype=torch.float64)

            return vib, state
        
        else:
            data = self.dataset[idx]
            s1 = data[0]
            s2 = data[1]

            state = data[-1]
            print(s2)

            x_tensor = torch.tensor(np.concatenate((s1,s2)),dtype=torch.float64)
            y_tensor = torch.tensor(state, dtype=torch.float64)

            if is_biwavelet:
                _, _, _, Wcoh, WXdt, freqs, coi = xwt(  trace_ref       = s1,
                                                        trace_current   = s2,
                                                        fs              = 10000,
                                                        ns              = 3,
                                                        nt              = 0.005,
                                                        vpo             = 20,
                                                        freqmin         = 1,
                                                        freqmax         = 25,
                                                        nptsfreq        = len(s1))
                
                Wcoh_tensor = torch.tensor(Wcoh, dtype=torch.float64)
                

                return Wcoh_tensor, x_tensor, y_tensor

            return x_tensor, y_tensor

if __name__ =='__main__':
    data_root = os.path.join(os.getcwd(), 'data')
    DataPreprocessing(data_root)

    # my_data = KAMPdataset(data_root='./data/test.csv', window_size=256, stride=100)
    # sample = my_data.__getitem__(0)