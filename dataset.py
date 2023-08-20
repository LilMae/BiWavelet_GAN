import os
import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import os
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    def __init__(self, data_path, window_size, stride, n_ch=2, is_biwavelet=True, im_size=512):
        """KAMP 데이터에 기반한 파이토치 데이터셋

        Args:
            data_path (str): 데이터셋 경로
            window_size (int): 진동 데이터 분석 윈도우 크기
            stride (int): 윈도우를 이동하는 stride 크기
            n_ch (int, optional): 입력으로 사용하고자 하는 채널 개수
        """
        
        data_pd = pd.read_csv(data_path)
        self.im_size = im_size
        self.dataset = []
        self.n_ch = n_ch
        self.is_biwavelet = is_biwavelet

        print(f'building dataset...')
        # make Slice by strid and window_size
        for idx in tqdm(range(0, len(data_pd), stride)):
            start_idx = idx
            end_idx = idx+window_size


            if (len(data_pd)-1) < end_idx:
                break
            
            try:
                if data_pd['class'][start_idx] != data_pd['class'][end_idx]:
                    continue
            except:
                print(f'start : {start_idx}')
                print(f'end : {end_idx}')
                exit()
            sensor1_np = data_pd['0'][start_idx:end_idx].to_numpy()
            sensor2_np = data_pd['1'][start_idx:end_idx].to_numpy()
            sensor3_np = data_pd['2'][start_idx:end_idx].to_numpy()
            sensor4_np = data_pd['3'][start_idx:end_idx].to_numpy()
            class_np = data_pd['class'][end_idx]


            if self.n_ch == 1:
                self.dataset.append([sensor1_np, class_np])
                self.dataset.append([sensor2_np, class_np])
                self.dataset.append([sensor3_np, class_np])
                self.dataset.append([sensor4_np, class_np])
            elif self.n_ch == 2:
                self.dataset.append([sensor1_np, sensor2_np ,class_np])
                self.dataset.append([sensor1_np, sensor3_np ,class_np])
                self.dataset.append([sensor1_np, sensor4_np ,class_np])
                self.dataset.append([sensor2_np, sensor3_np ,class_np])
                self.dataset.append([sensor2_np, sensor4_np ,class_np])
                self.dataset.append([sensor3_np, sensor4_np ,class_np])
            else:
                print(f'Not Implemented Yet... n_ch : {self.n_ch} ')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        if self.n_ch == 1:
            data = self.dataset[idx]
            sensor_tensor = torch.tensor(data[0], dtype=torch.float64)
            class_tensor = torch.tensor(data[-1], dtype=torch.float64)

            return {'sensor' : sensor_tensor, 'class' :class_tensor}
        
        elif self.n_ch ==2:
            data = self.dataset[idx]
            
            s1 = data[0]
            s2 = data[1]
            state = data[-1]

            sensor_tensor = torch.tensor(np.concatenate((s1,s2), axis=0),dtype=torch.float64)
            class_tensor = torch.tensor(state, dtype=torch.int64)

            if self.is_biwavelet:
                time = np.array([i for i in range(len(s1))])
                WXamp, WXspec, WXangle, Wcoh, WXdt, freqs, coi = xwt(  trace_ref       = s1,
                                                        trace_current   = s2,
                                                        fs              = 10000,
                                                        ns              = 3,
                                                        nt              = 0.005,
                                                        vpo             = 20,
                                                        freqmin         = 1,
                                                        freqmax         = 25,
                                                        nptsfreq        = len(s1))
                
                WXdt_tensor = XWT_tensor(time,freqs,WXdt,im_size=self.im_size)
                Wcoh_tensor = XWT_tensor(time,freqs,Wcoh,im_size=self.im_size)
                
                return {'sensor' : sensor_tensor, 'class' :class_tensor, 'Wxdt' : WXdt_tensor, 'Wcoh' : Wcoh_tensor}

            return {'sensor' : sensor_tensor, 'class' :class_tensor}

def load_vib(opt):
    
    train_dataset = KAMPdataset(data_path=os.path.join(os.getcwd(), 'data','train.csv'),
                                window_size=opt.window_size,
                                stride=opt.stride,
                                im_size=opt.im_size)
    test_dataset = KAMPdataset(data_path=os.path.join(os.getcwd(), 'data','test.csv'),
                               window_size=opt.window_size,
                                stride=opt.stride,
                                im_size=opt.im_size)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=opt.batchsize,
                                                shuffle=True,
                                                num_workers=int(opt.workers),
                                                drop_last = True,
                                                worker_init_fn=(None if opt.manualseed == -1
                                                else lambda x: np.random.seed(opt.manualseed)))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=opt.batchsize,
                                                shuffle=True,
                                                num_workers=int(opt.workers),
                                                drop_last = True,
                                                worker_init_fn=(None if opt.manualseed == -1
                                                else lambda x: np.random.seed(opt.manualseed)))
    
    return {'train' : train_loader, 'test' : test_loader}

def XWT_tensor(time, freqs, target, im_size=512):
    # DPI 값을 설정합니다.
    dpi = 500  # 원하는 DPI 값을 이곳에 설정하세요.

    # Figure를 생성할 때 DPI를 지정합니다.
    fig = plt.figure(dpi=dpi)

    # 주어진 코드
    plt.pcolormesh(time, freqs, target, cmap='jet_r', edgecolors='none')
    plt.clim([-0.02, 0.01])
    plt.ylim(freqs[-1], freqs[0])
    plt.axis('off')

    # 그림을 그리고 바로 캔버스에서 데이터를 가져와 넘파이 배열로 변환
    plt.draw()
    canvas = plt.gca().get_figure().canvas
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(canvas.get_width_height()[::-1] + (3,))

    # 흰색(255, 255, 255)을 제거
    non_white_pixels = (image_array < [255, 255, 255]).any(axis=2)
    non_white_rows = non_white_pixels.any(axis=1)
    non_white_columns = non_white_pixels.any(axis=0)
    image_array_cropped = image_array[non_white_rows][:, non_white_columns]
    
    # 넘파이 배열을 텐서로 변환
    image_tensor = torch.from_numpy(image_array_cropped).permute(2, 0, 1).float() / 255.0
    
    # 리쉐이핑: 원하는 형태로 변경 (예: (3, 224, 224) 등)
    reshaped_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(im_size, im_size), mode='bilinear', align_corners=False)
    
    return reshaped_tensor.squeeze(0)  # batch dimension 제거 후 반환

if __name__ =='__main__':
    print(f'Testing Dataset')
    test = KAMPdataset(os.path.join(os.getcwd(), 'data','test.csv'), 10000, 1000)
    samplw = test[0]