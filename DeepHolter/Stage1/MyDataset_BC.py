import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
from time import time 
import numpy as np 
import torchvision
from torchvision.models import resnet152
#MyDataset(file_paths,y_xlsx,statement_xlsx,train_path,patientinfo_xlsx,hrv_paths,chara_paths)
class MyDataset(Dataset):
    def __init__(self, file_path, y_xlsx, file_list, positional_list=None, patientinfo_xlsx=None, chara_paths=None, timeinfo_paths=None, 
                 cvd_file_path=None, cvd_chara_paths=None, cvd_timeinfo_path=None,train_mean=None, train_std=None, scalar=None):

        self.file_path = file_path
        self.file_list = file_list
        self.y_xlsx = y_xlsx
        self.positional_list = positional_list
        self.patientinfo_xlsx = patientinfo_xlsx.fillna(0)
        self.chara_paths = chara_paths
        self.timeinfo_paths = timeinfo_paths
        self.train_mean = train_mean
        self.train_std = train_std
        self.cvd_file_path = cvd_file_path
        self.cvd_chara_paths = cvd_chara_paths
        self.cvd_timeinfo_path = cvd_timeinfo_path
        self.scalar = scalar
    def __len__(self):

        return len(self.file_list)



    def __getitem__(self, idx):
        y = self.y_xlsx[self.y_xlsx["ID"] == self.file_list[idx][0][0].split('_')[0]].values[0][1:].astype(float)
        y = torch.tensor(y, dtype=torch.float32)
        file_path = self.file_path if y.sum() == 0 else self.cvd_file_path
        chara_paths = self.chara_paths if y.sum() == 0 else self.cvd_chara_paths
        timeinfo_paths = self.timeinfo_paths if y.sum() == 0 else self.cvd_timeinfo_path

        data_list = []
        data_mask_list = []
        for di in self.file_list[idx]:
            dl = [np.expand_dims(np.load(os.path.join(file_path, file_name.split('_')[0], file_name)), axis=0)   
                for file_name in di]  

            data = np.concatenate(dl, axis=0)   
            data = torch.from_numpy(data.astype(np.float32)).transpose(2, 1)  

            data_mask = torch.zeros(24)
            data_mask[:data.shape[0]] = 1

            data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, 24-data.shape[0]), mode='constant', value=0)
            # print(data.size())
            data_list.append(data)
            data_mask_list.append(data_mask)
        data_list = torch.stack(data_list)
        data_mask_list = torch.stack(data_mask_list)

        patientinfo = self.patientinfo_xlsx[self.patientinfo_xlsx["ID"] == self.file_list[idx][0][0].split('_')[0]].values[0][1:].astype(float)
        patientinfo = torch.tensor(patientinfo, dtype=torch.float32)
        patientinfo = patientinfo.unsqueeze(0).expand(24,-1)

        timeinfo_l = []
        chara_l = []
        for di in self.file_list[idx]:
            file_idx = [int(file_name.split('_')[1].split('.')[0]) for file_name in di]
            timeinfoFile = pd.read_excel(timeinfo_paths + di[0].split('_')[0] +"_timeinfo.xlsx")
            timeinfo = timeinfoFile.iloc[file_idx].copy()
            timeinfo.fillna(0,inplace=True)
            timeinfo = timeinfo["Hour"].dt.hour
            timeinfo_list = []
            for t in timeinfo:
                time_cur = [0]*24
                time_cur[int(t)] = 1
                timeinfo_list.append(time_cur)
            timeinfo = torch.tensor(timeinfo_list, dtype=torch.float32)
            timeinfo = torch.nn.functional.pad(timeinfo, (0, 0, 0, 24-timeinfo.shape[0]), mode='constant', value=0)
            timeinfo_l.append(timeinfo)
            # print(timeinfo.shape)
            chara = pd.read_csv(chara_paths + di[0].split('_')[0] + ".csv")
            chara = chara.iloc[file_idx]
            if self.train_mean is not None and self.train_std is not None:
                chara = chara.fillna(0)
                chara = (chara - self.train_mean) / self.train_std
            if self.scalar is not None:
                chara = self.scalar.transform(chara)
            if isinstance(chara, pd.DataFrame):
                chara = chara.fillna(0).values.astype(float)
            elif isinstance(chara, np.ndarray):
                chara = np.nan_to_num(chara, nan=0.0).astype(float)
            chara = torch.tensor(chara, dtype=torch.float32)
            chara = torch.nn.functional.pad(chara, (0, 0, 0, 24-chara.shape[0]), mode='constant', value=0)
            chara_l.append(chara)
        timeinfo_l = torch.stack(timeinfo_l)
        chara_l = torch.stack(chara_l)
        # print(chara.shape)
        positional_list = []
        for pi in self.positional_list[idx]:
            positional = torch.tensor(pi,dtype=torch.float32)
            # print(positional.shape)
            positional = torch.nn.functional.pad(positional, (0, 0, 0, 24-positional.shape[0]), mode='constant', value=0)
            positional_list.append(positional)
        positional_list = torch.stack(positional_list)
        # print(positional.shape)
        return data_list, positional_list, data_mask_list, patientinfo, timeinfo_l, chara_l, y
