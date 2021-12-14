from math import fabs
import pandas as pd
import numpy as np
import torch
from torch import nn
from poutyne import Model
from torch.optim import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class Net(nn.Module):
    def __init__(self,
                 num_feature: int = 182,
                 num_class: int = 5) -> None:
        super(Net, self).__init__()

        self.fc1 = nn.Linear(num_feature, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x


def get_train_valid(batch_size):
    df = pd.read_csv('data_text_04.csv', skiprows=[1], sep=';')
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    df = df[df.columns.drop(['Chief_complain', 'KTAS_RN', 'Diagnosis in ED','Length of stay_min', 'KTAS duration_min', 'mistriage', 'Arrival mode'])]
    df = df.replace('??', np.nan)
    df = df.dropna(axis=1, how='all')
    df['NRS_pain'] = df['NRS_pain'].replace('#BOÞ!', 0)
    df['Saturation'] = df['Saturation'].replace(np.nan, 100)
    df.fillna(method='ffill', inplace=True)
    df = df.apply(pd.to_numeric)
    y = df['KTAS_expert']
    X = df.drop(['KTAS_expert'], axis=1)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=22)
    X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1, random_state=22)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    X_train, y_train = np.array(X_train), np.array(y_train) - 1
    X_val, y_val = np.array(X_val), np.array(y_val) - 1
    X_test, y_test = np.array(X_test), np.array(y_test) - 1
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader

def get_train_valid_reduced(batch_size, train, valid, test):
    train_dataset = ClassifierDataset(torch.from_numpy(train["data"].to_numpy()).float(), torch.from_numpy(train["target"].to_numpy() - 1).long())
    val_dataset = ClassifierDataset(torch.from_numpy(valid["data"].to_numpy()).float(), torch.from_numpy(valid["target"].to_numpy() - 1).long())
    test_dataset = ClassifierDataset(torch.from_numpy(test["data"].to_numpy()).float(), torch.from_numpy(test["target"].to_numpy() - 1).long())
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, test_loader