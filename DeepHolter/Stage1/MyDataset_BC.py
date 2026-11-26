import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import os 
from time import time 
import numpy as np 
def segment_index(filename):
    return int(filename.split('_')[1].split('.')[0])


def read_ecg_features(path, columns=None):
    frame = pd.read_csv(path, dtype=np.float64)
    if columns is not None and not frame.columns.equals(columns):
        if not frame.columns.symmetric_difference(columns).empty:
            raise ValueError(f"ECG feature columns differ from the training fold: {path}")
        frame = frame.loc[:, columns]
    return frame.replace([np.inf, -np.inf], np.nan)


def fit_ecg_feature_means(training_bags, chara_paths, cvd_chara_paths, cvd_ids):
    """Pool observed segment values per named lead/measurement/statistic column."""
    sums = counts = None
    segment_count = 0
    for bags in training_bags:
        patient_id = bags[0][0].split('_')[0]
        root = cvd_chara_paths if patient_id in cvd_ids else chara_paths
        path = os.path.join(root, patient_id + ".csv")
        frame = read_ecg_features(path, None if sums is None else sums.index)
        rows = [segment_index(name) for bag in bags for name in bag]
        retained = frame.iloc[rows]
        segment_count += len(rows)
        observed = retained.count()
        totals = retained.sum()
        if sums is None:
            sums, counts = totals, observed
        else:
            sums += totals
            counts += observed
    if counts is None:
        raise ValueError("Cannot fit ECG feature means on an empty training fold.")
    missing = counts.index[counts == 0]
    if len(missing):
        raise ValueError(f"No observed training values for ECG features: {missing.tolist()}")
    missing_fraction = 1 - counts.sum() / (segment_count * len(counts))
    print(f"Training-fold ECG feature missingness before imputation: {missing_fraction:.4%}")
    return sums / counts


#MyDataset(file_paths,y_xlsx,statement_xlsx,train_path,patientinfo_xlsx,hrv_paths,chara_paths)
class MyDataset(Dataset):
    def __init__(self, file_path, y_xlsx, file_list, positional_list=None, patientinfo_xlsx=None, chara_paths=None, timeinfo_paths=None, 
                 cvd_file_path=None, cvd_chara_paths=None, cvd_timeinfo_path=None,bag_mask=None,train_mean=None, train_std=None, scalar=None):

        self.file_path = file_path
        self.file_list = file_list
        self.y_xlsx = y_xlsx
        self.positional_list = positional_list
        self.patientinfo_xlsx = patientinfo_xlsx
        self.chara_paths = chara_paths
        self.timeinfo_paths = timeinfo_paths
        self.bag_mask = torch.tensor(bag_mask,dtype=torch.float32)
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

            if dl:
                data = np.concatenate(dl, axis=0)
                data = torch.from_numpy(data.astype(np.float32)).transpose(2, 1)
            else:
                # A retained hour always contributes to the first bag.
                data = data_list[0][:0]

            data_mask = torch.zeros(24)
            data_mask[:data.shape[0]] = 1

            data = torch.nn.functional.pad(data, (0, 0, 0, 0, 0, 24-data.shape[0]), mode='constant', value=0)
            # print(data.size())
            data_list.append(data)
            data_mask_list.append(data_mask)
        data_list = torch.stack(data_list)
        data_mask_list = torch.stack(data_mask_list)

        patientinfo = self.patientinfo_xlsx[self.patientinfo_xlsx["ID"] == self.file_list[idx][0][0].split('_')[0]].values[0][1:].astype(float)
        if not np.isfinite(patientinfo).all():
            raise ValueError("Patient demographics must be finite after preprocessing.")
        patientinfo = torch.tensor(patientinfo, dtype=torch.float32)
        patientinfo = patientinfo.unsqueeze(0).expand(24,-1)

        timeinfo_l = []
        patient_id = self.file_list[idx][0][0].split('_')[0]
        time_table = pd.read_excel(os.path.join(timeinfo_paths, patient_id + "_timeinfo.xlsx"))
        hours = pd.to_datetime(time_table["Hour"]).dt.hour
        columns = self.train_mean.index if isinstance(self.train_mean, pd.Series) else None
        features = read_ecg_features(os.path.join(chara_paths, patient_id + ".csv"), columns)
        rows = [segment_index(name) for bag in self.file_list[idx] for name in bag]
        features = features.iloc[rows]
        if self.train_mean is not None:
            means = pd.Series(self.train_mean, index=features.columns).to_numpy()
            values = features.to_numpy(copy=True)
            np.copyto(values, means, where=np.isnan(values))
            features = pd.DataFrame(values, index=features.index, columns=features.columns)
        if features.isna().to_numpy().any():
            raise ValueError(f"Missing ECG features require fitted training means: {patient_id}")
        if self.train_mean is not None and self.train_std is not None:
            std = pd.Series(self.train_std, index=features.columns)
            features = (features - self.train_mean) / std.replace(0, 1)
        if self.scalar is not None:
            features = pd.DataFrame(self.scalar.transform(features), index=features.index)
        feature_tensor = torch.from_numpy(features.to_numpy(dtype=np.float32))
        feature_bags = feature_tensor.split([len(bag) for bag in self.file_list[idx]])
        chara_l = [torch.nn.functional.pad(bag, (0, 0, 0, 24 - len(bag))) for bag in feature_bags]
        for di in self.file_list[idx]:
            file_idx = [segment_index(name) for name in di]
            hour_indices = torch.from_numpy(hours.iloc[file_idx].to_numpy(dtype=np.int64))
            timeinfo = torch.nn.functional.one_hot(hour_indices, num_classes=24).float()
            timeinfo = torch.nn.functional.pad(timeinfo, (0, 0, 0, 24-timeinfo.shape[0]), mode='constant', value=0)
            timeinfo_l.append(timeinfo)
            # print(timeinfo.shape)
        timeinfo_l = torch.stack(timeinfo_l)
        chara_l = torch.stack(chara_l)
        # print(chara.shape)
        positional_list = []
        for pi in self.positional_list[idx]:
            positional = torch.tensor(pi,dtype=torch.float32).reshape(-1, 24)
            # print(positional.shape)
            positional = torch.nn.functional.pad(positional, (0, 0, 0, 24-positional.shape[0]), mode='constant', value=0)
            positional_list.append(positional)
        positional_list = torch.stack(positional_list)
        bag_mask = self.bag_mask[idx]
        # print(positional.shape)
        return data_list, positional_list, data_mask_list, patientinfo, timeinfo_l, chara_l, bag_mask, y
