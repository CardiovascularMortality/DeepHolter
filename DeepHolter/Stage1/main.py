import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd 
import os 
from time import time 
import numpy as np 
import torchvision
from MyDataset_BC import MyDataset
from model import *
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, auc, precision_recall_fscore_support, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
# from Focalloss import FocalLoss
import random
from torch.optim.lr_scheduler import LambdaLR
from torchvision.ops import sigmoid_focal_loss
import concurrent.futures
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import joblib
import gc

def tensorboard_calculation(y, score, mode, epoch, writer, leng, total_loss):
    AUROC = roc_auc_score(y_true=y, y_score=score, average='macro')
    pred = (score >= 0.5).astype(int)
    PRE, RE, F1, _ = precision_recall_fscore_support(y_true=y, y_pred=pred, pos_label=1,average='binary',zero_division=np.nan)
    ACC = accuracy_score(y_true=y, y_pred=pred, normalize=True)
    TPC = np.sum(np.logical_and(y==1,pred==1))
    FPC = np.sum(np.logical_and(y==0,pred==1))
    TNC = np.sum(np.logical_and(y==0,pred==0))
    FNC = np.sum(np.logical_and(y==1,pred==0))
    print(f'\n{mode} Avg:[Loss:{total_loss/leng:.6f}][Acc:{ACC:.6f}][Precision:{PRE:.6f}][Recall:{RE:.6f}][F1:{F1:.6f}][AUC:{AUROC:.6f}][TPC:{TPC:}][FPC:{FPC:}][TNC:{TNC:}][FNC:{FNC:}]')
    pred2 = (score >= 0.95).astype(int)
    PRE2, RE2, F12, _ = precision_recall_fscore_support(y_true=y, y_pred=pred2, pos_label=1,average='binary',zero_division=np.nan)
    ACC2 = accuracy_score(y_true=y, y_pred=pred2, normalize=True)
    TPC2 = np.sum(np.logical_and(y==1,pred2==1))
    FPC2 = np.sum(np.logical_and(y==0,pred2==1))
    TNC2 = np.sum(np.logical_and(y==0,pred2==0))
    FNC2 = np.sum(np.logical_and(y==1,pred2==0))
    print(f'{mode} 0.95:[Loss:{total_loss/leng:.6f}][Acc:{ACC2:.6f}][Precision:{PRE2:.6f}][Recall:{RE2:.6f}][F1:{F12:.6f}][AUC:{AUROC:.6f}][TPC:{TPC2:}][FPC:{FPC2:}][TNC:{TNC2:}][FNC:{FNC2:}]')
    writer.add_scalar(f'Loss/{mode}', total_loss/leng, epoch)
    writer.add_scalar(f'Acc/{mode}', ACC, epoch)
    writer.add_scalar(f'Precision/{mode}', PRE, epoch)
    writer.add_scalar(f'Recall/{mode}', RE, epoch)
    writer.add_scalar(f'F1/{mode}', F1, epoch)
    writer.add_scalar(f'Auroc/{mode}', AUROC, epoch)
    writer.add_scalar(f'TPC/{mode}', TPC, epoch)
    writer.add_scalar(f'FPC/{mode}', FPC, epoch)
    writer.add_scalar(f'TNC/{mode}', TNC, epoch)
    writer.add_scalar(f'FNC/{mode}', FNC, epoch)
    return {'loss':total_loss/leng,'acc':ACC,'precision':PRE,'recall':RE,'f1':F1,'auroc':AUROC,'tpc':TPC,'fpc':FPC,'tnc':TNC,'fnc':FNC}

random_seed = 0  
random.seed(random_seed)  
file_paths = ""
chara_paths = ""
timeinfo_paths = ""
cvd_file_paths = ""
cvd_chara_paths = ""
cvd_timeinfo_paths = ""

file_list = os.listdir(file_paths)# recordid_de.csv

sum_excel = pd.read_excel(".xlsx")

y_xlsx = sum_excel[["ID","Group"]].copy()
test_size=0.3
valid_size=0.1
train_size = 0.7
batch_size = 32

expname = "rs"
#
device = torch.device("cuda")


cvd_filefolder = []
live_filefolder = []
external_valid_filefolder = []

test_cvd_filefolder = []
test_live_filefolder = []
train_cvd_filefolder = []
train_live_filefolder = []

timeinfo_list = os.listdir(timeinfo_paths)

timeinfo_list = [i.split('_')[0] for i in timeinfo_list]

#Group=1
sum_cvd_excel = sum_excel[
    (sum_excel["Group"] == 1) &
    ((sum_excel["Hospital"] == 0) | (sum_excel["Hospital"] == 1))
]
#
sum_cvd_excel = sum_cvd_excel[sum_cvd_excel["ID"].isin(timeinfo_list)]

cvd_filefolder = sum_cvd_excel["ID"].tolist()
random.shuffle(cvd_filefolder)  
all_cvd_len = len(cvd_filefolder)
n_test = int(all_cvd_len * test_size)
#fold1
test_cvd_filefolder = cvd_filefolder[:n_test]
train_cvd_filefolder = cvd_filefolder[n_test:]
###############################################


# # Group=0
sum_live_excel = sum_excel[
    (sum_excel["Group"] == 0) &
    ((sum_excel["Hospital"] == 0) | (sum_excel["Hospital"] == 1))
]
sum_live_excel = sum_live_excel[sum_live_excel["ID"].isin(timeinfo_list)]


live_filefolder = sum_live_excel["ID"].tolist()
random.shuffle(live_filefolder)  
all_live_len = len(live_filefolder)
n_test = int(all_live_len * test_size)
#fold1
test_live_filefolder = live_filefolder[:n_test]
train_live_filefolder = live_filefolder[n_test:]
#############################################

## C-E
external_valid__filefolder = list(sum_excel[(sum_excel["Hospital"]!=0)&(sum_excel["Hospital"]!=1)]["ID"])

external_valid__filefolder = [i for i in external_valid__filefolder if i in timeinfo_list]


print(f"cvd number:{len(cvd_filefolder)}, live number:{len(live_filefolder)}, external valid number:{len(external_valid__filefolder)}")

random.shuffle(external_valid__filefolder)


train_path = []
test_path = []
external_valid_path = []

train_positional_list = []
test_positional_list = []
external_valid_positional_list = []
hour_onehot_len = 24
train_cvd_cnt = 0
train_live_cnt = 0
test_cvd_cnt = 0
test_live_cnt = 0
external_valid_cnt = 0
max_hour = 0
for folder, path_list, positional_list, cnt_name in [
    (train_cvd_filefolder, train_path, train_positional_list, "train_cvd_cnt"),
    (train_live_filefolder, train_path, train_positional_list, "train_live_cnt"),
    (test_cvd_filefolder, test_path, test_positional_list, "test_cvd_cnt"),
    (test_live_filefolder, test_path, test_positional_list, "test_live_cnt"),
    (external_valid__filefolder, external_valid_path, external_valid_positional_list, "external_valid_cnt")
]:
    for i in folder:
        timeinfo = pd.read_excel(timeinfo_paths + i + "_timeinfo.xlsx")
        #
        hour_max = timeinfo["Hour"].value_counts().max()
        if hour_max > 60:
            hour_max = 60
        tmp_list = [[] for _ in range(hour_max)]
        tmp_list2 = [[] for _ in range(hour_max)]
        th = timeinfo[timeinfo["Hour"].isin(timeinfo['Hour'].unique()[0:24])]
        first_hour = th["Hour"].min()
        for hour in th["Hour"].unique():
            tmp = th[th["Hour"] == hour]
            tmp_len = len(tmp)
            if tmp_len > 60:
                tmp_len = 60
            for j in range(tmp_len):
                tmp_list[j].append(f"{i}_{tmp.index[j]}.npy")
                a = [0] * hour_onehot_len
                hour_cur = ((hour - first_hour).total_seconds() // 3600)%24
                a[int(hour_cur)] = 1
                max_hour = max(max_hour, int(hour_cur))
                tmp_list2[j].append(a)
        path_list.append(tmp_list)
        positional_list.append(tmp_list2)
    locals()[cnt_name] = len(path_list)
print(f"train cvd cnt:{train_cvd_cnt}, train live cnt:{train_live_cnt-train_cvd_cnt}, test cvd cnt:{test_cvd_cnt}, test live cnt:{test_live_cnt-test_cvd_cnt}, external valid cnt:{external_valid_cnt}")

print(f"Total round:{(len(train_path)+len(test_path)+len(external_valid_path))//batch_size}")


train_means = None
train_stds = None
scalar = None
charas_len = len(pd.read_csv(chara_paths+file_list[0].split("_")[0]+".csv").columns)

#
patientinfo_xlsx = sum_excel[["ID","Age","Sex"]].copy()
age_max = patientinfo_xlsx["Age"].max()
age_min = patientinfo_xlsx["Age"].min()
age_mean = patientinfo_xlsx["Age"].mean()
age_std = patientinfo_xlsx["Age"].std()
patientinfo_xlsx["Age"] = patientinfo_xlsx["Age"].fillna(int(age_mean))
patientinfo_xlsx["Age"] = patientinfo_xlsx["Age"].astype(float)
#
patientinfo_xlsx["Age"] = (patientinfo_xlsx["Age"]-age_min)/(age_max-age_min)

#0 1 
patientinfo_xlsx = pd.get_dummies(patientinfo_xlsx,columns=["Sex"],dtype=float)


train_dataset = MyDataset(file_paths,y_xlsx,train_path,train_positional_list,patientinfo_xlsx,chara_paths,timeinfo_paths,cvd_file_paths,cvd_chara_paths,cvd_timeinfo_paths,train_mean=train_means,train_std=train_stds,scalar=scalar)
test_dataset = MyDataset(file_paths,y_xlsx,test_path,test_positional_list,patientinfo_xlsx,chara_paths,timeinfo_paths,cvd_file_paths,cvd_chara_paths,cvd_timeinfo_paths,train_mean=train_means,train_std=train_stds,scalar=scalar)
external_valid_dataset = MyDataset(file_paths,y_xlsx,external_valid_path,external_valid_positional_list,patientinfo_xlsx,chara_paths,timeinfo_paths,cvd_file_paths,cvd_chara_paths,cvd_timeinfo_paths,train_mean=train_means,train_std=train_stds,scalar=scalar)
#WeightedRandomSampler
#DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=32,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=32,pin_memory=True)
external_valid_loader = DataLoader(external_valid_dataset, batch_size=batch_size, shuffle=True,drop_last=False,num_workers=32,pin_memory=True)

print("loading model...")
model = Model_All()
# model = torch.nn.DataParallel(model)
model.to(device)

print("model loaded.")
#
# pos_weight = torch.tensor([5.00]).to(device)
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
criterion = nn.BCEWithLogitsLoss()

#
optimizer = optim.AdamW(model.parameters(), lr=1e-5,weight_decay=0)

os.makedirs('pth/'+expname,exist_ok=True)
writer = SummaryWriter(log_dir='runs/'+expname)
n_epoch = 150
train_len = len(train_loader)
test_len = len(test_loader)
external_valid_len = len(external_valid_loader)
pbar = tqdm(total=train_len, mininterval=0.3)
max_tst_loss = 1e6
max_tst_auc = 0.0
bc_alpha = 1#0.999
contra_alpha = 1-bc_alpha
valid_epoch = 1

for epoch in range(n_epoch):  # loop over the dataset multiple times
    model.train()
    total_loss = 0.0
    y_train_total = np.array([])
    pred_train_total = np.array([])

    for inputs, positional,data_mask, patientinfo, timeinfo, chara, y in train_loader:
        #nan
        if torch.isnan(inputs).any():
            print("nan exists.")
        if torch.isnan(patientinfo).any():
            print("nan exists1.")
        if torch.isnan(chara).any():
            print("nan exists2.")
        if torch.isnan(timeinfo).any():
            print("nan exists3.")
        if torch.isnan(y).any() :
            print("nan exists4.") 
        if torch.isnan(positional).any():
            print("nan exists5.")

        inputs = inputs.to(device)
        patientinfo = patientinfo.to(device)
        chara = chara.to(device)
        timeinfo = timeinfo.to(device)
        positional = positional.to(device)
        data_mask = data_mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        # forward + backward + optimize
        pred,contra_loss,_,_,_,out_embed = model(inputs,positional,data_mask,patientinfo,timeinfo,chara,bc_only=False)

        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        y_argmax = y.cpu().numpy()

        pred = F.sigmoid(pred)

        y_train_total = np.append(y_train_total,y_argmax)
        pred_train_total = np.append(pred_train_total,pred.cpu().detach().numpy())
        pbar.set_description(f'Train:{epoch}/{n_epoch} '
                                f'[Loss:{loss.item():.4f}]'
                                )
        pbar.update(1)
        # break
    _ = tensorboard_calculation(y_train_total, pred_train_total, "train", epoch, writer, train_len, total_loss)

    #
    pbar.reset(test_len)
    total_test_loss = 0.0
    y_test_total = np.array([])
    pred_test_total = np.array([])

    total_external_valid_loss = 0.0
    y_external_valid_total = np.array([])
    pred_external_valid_total = np.array([])
    model.eval()
    with torch.no_grad():
        for inputs, positional,data_mask, patientinfo, timeinfo, chara, y in test_loader:
            if torch.isnan(inputs).any() :
                print("nan exists.")
            if torch.isnan(patientinfo).any() :
                print("nan exists1.")
            if torch.isnan(chara).any() :
                print("nan exists2.")
            if torch.isnan(timeinfo).any() :
                print("nan exists3.")
            if torch.isnan(y).any() :
                print("nan exists4.") 
            if torch.isnan(positional).any() :
                print("nan exists5.")
            inputs = inputs.to(device)
            patientinfo = patientinfo.to(device)
            chara = chara.to(device)
            timeinfo = timeinfo.to(device)
            positional = positional.to(device)
            data_mask = data_mask.to(device)
            y = y.to(device)
            
            pred,contra_loss,_,_,_,out_embed = model(inputs,positional,data_mask,patientinfo,timeinfo,chara,bc_only=False)

            loss = criterion(pred, y)
            ##

            total_test_loss += loss.item()
            y_argmax = y.cpu().numpy()
            pred = F.sigmoid(pred)
            y_test_total = np.append(y_test_total,y_argmax)
            pred_test_total = np.append(pred_test_total,pred.cpu().detach().numpy())
            pbar.set_description(f'Test:{epoch}/{n_epoch} '
                                    f'[Loss:{loss.item():.4f}]'
                                    )
            pbar.update(1)
            # break
        tc_dir = tensorboard_calculation(y_test_total, pred_test_total, "test", epoch, writer, test_len, total_test_loss)
        tt_auc = tc_dir['auroc']
        if tc_dir['auroc'] >= max_tst_auc and tc_dir['auroc'] > 0.5:
            max_tst_auc = tc_dir['auroc']
            torch.save(model.state_dict(), f"pth/{expname}/{expname}_{epoch}.pth")
            print("Model saved at epoch {}.".format(epoch))   
        pbar.reset(external_valid_len)
        for inputs, positional,data_mask, patientinfo, timeinfo, chara, y in external_valid_loader:
            #nan
            if torch.isnan(inputs).any() :
                print("nan exists.")
            if torch.isnan(patientinfo).any() :
                print("nan exists1.")
            if torch.isnan(chara).any() :
                print("nan exists2.")
            if torch.isnan(timeinfo).any() :
                print("nan exists3.")
            if torch.isnan(y).any() :
                print("nan exists4.") 
            if torch.isnan(positional).any() :
                print("nan exists5.")
            inputs = inputs.to(device)
            patientinfo = patientinfo.to(device)
            chara = chara.to(device)
            timeinfo = timeinfo.to(device)
            positional = positional.to(device)
            data_mask = data_mask.to(device)
            y = y.to(device)
            
            pred,contra_loss,_,_,_,out_embed = model(inputs,positional,data_mask,patientinfo,timeinfo,chara,bc_only=False)

            loss = criterion(pred, y)

            total_external_valid_loss += loss.item()

            y_argmax = y.cpu().numpy()
            
            ##
            pred = F.sigmoid(pred)
            ##
            
            y_external_valid_total = np.append(y_external_valid_total,y_argmax)
            pred_external_valid_total = np.append(pred_external_valid_total,pred.cpu().detach().numpy())
            pbar.set_description(f'External Valid:{epoch}/{n_epoch} '
                                    f'[Loss:{loss.item():.4f}]'
                                    )
            pbar.update(1)
            # break
        _ = tensorboard_calculation(y_external_valid_total, pred_external_valid_total, "external_valid", epoch, writer, external_valid_len, total_external_valid_loss)
        exvlid_auc = _['auroc']
        with open(f"pth/{expname}/{expname}_log.txt", "a") as f:
            f.write("Random {} epoch{}: valid auc {}, test auc {}, external valid auc {}.\n".format(random_seed,epoch,max_tst_auc,tt_auc,exvlid_auc))
        print("Random {} epoch{}: valid auc {}, test auc {}, external valid auc {}.\n".format(random_seed,epoch,max_tst_auc,tt_auc,exvlid_auc))
    if (epoch+1) % 10 == 0:
        torch.save(model.state_dict(), f"pth/{expname}/{expname}_{epoch}.pth")
    pbar.reset(train_len)
    gc.collect()
    # break
    