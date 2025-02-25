import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from configs.train_configs import TrainingConfig

from sklearn.preprocessing import StandardScaler, MinMaxScaler

class CustomDataset(Dataset):
    def __init__(self, df_input: pd.DataFrame, features: list, target: str, accelerator: str):
        self.features = torch.FloatTensor(df_input[features].to_numpy()).to(accelerator)
        self.target = torch.FloatTensor(df_input[target].to_numpy()).to(accelerator)

    def __len__(self):
        """
        Returns the total number of data
        """
        return len(self.features)
    
    def __getitem__(self, index):
        """
        Retrieve one sample at the given index

        Args:
            idx(int): index of the sample to retrieve

        Returns:
            tuple(feature, target): as tensor
        """
        features = self.features[index]
        target = self.target[index]
        return features, target
    
class DataModule(nn.Module):
    def __init__(self, df, config: TrainingConfig, mode):
        super().__init__()
        self.config = config
        if mode == 'train':
            # split_idx = math.floor(len(df)*config.train_test_split_size)
            # self.df_train = df[:split_idx]
            # self.df_test = df[split_idx:]

            self.df_train, self.df_test = self.custom_train_test_split(df, config=self.config)
            self.n_fold = config.n_fold
            self.cv_setup()
            
        elif mode == 'eval':
            # split_idx = math.floor(len(df)*config.train_test_split_size)
            # self.df_train = df[:split_idx]
            # self.df_test = df[split_idx:]
            self.df_train, self.df_test = self.custom_train_test_split(df, config=self.config)
            
        elif mode == 'predict':
            self.df_all = df
        
        self.batch_size = config.batch_size
        self.accelerator = torch.device("cuda" if (torch.cuda.is_available() and config.accelerator == "gpu") else "cpu")
        
        # initial the datasets as None
        self.tain_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.features = df.drop(config.data_config.target, axis=1).columns
        self.target = config.data_config.target
        self.num_features = df.select_dtypes(include=['float64', 'int64']).columns
        self.scaler = self.__get_scaler()

    def cv_setup(self, test_days=30):    
        self.index_dict = {}
        # use TimeSeriesSplit to separate train and valid datasets
        tss = TimeSeriesSplit(n_splits=self.n_fold, test_size=test_days)
        for i, (train_idx, val_idx) in enumerate(tss.split(self.df_train)):
            train_dates = self.df_train.index[train_idx].tolist()
            val_dates = self.df_train.index[val_idx].tolist()
            self.index_dict[i] = {
                "train_idx": train_dates,
                "val_idx": val_dates
            }

    def custom_train_test_split(self, df, config: TrainingConfig):
        split_idx = math.floor(len(df)*config.train_test_split_size)
        df_train = df[:split_idx]
        df_test = df[split_idx:]

        return df_train, df_test
    
    def __get_scaler(self):
        return StandardScaler() if self.config.data_config.use_standardization else MinMaxScaler()

    def get_fold_loader(self, fold, num_workers):

        # create train_loader
        train_idx = self.index_dict[fold]['train_idx']
        df_train_fold = self.df_train[self.df_train.index.isin(train_idx)]        
        df_train_fold.loc[:, self.num_features] = self.scaler.fit_transform(df_train_fold[self.num_features])
        self.train_dataset = CustomDataset(
            df_train_fold,
            features=self.features,
            target=self.target,
            accelerator=self.accelerator
        )
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        # create valid_loader
        val_idx  = self.index_dict[fold]['val_idx']
        df_val_fold = self.df_train[self.df_train.index.isin(val_idx)]
        df_val_fold.loc[:, self.num_features] = self.scaler.transform(df_val_fold[self.num_features])
        valid_dataset = CustomDataset(
            df_val_fold,
            features=self.features,
            target=self.target,
            accelerator=self.accelerator
        )
        valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        return train_loader, valid_loader

    def get_train_test_data_loader(self, num_workers):
        
        self.df_train.loc[:, self.num_features] = self.scaler.fit_transform(self.df_train[self.num_features])
        self.full_train_dataset = CustomDataset(
            self.df_train,
            features=self.features,
            target=self.target, 
            accelerator=self.accelerator
        )

        full_train_data_loader = DataLoader(self.full_train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.df_test.loc[:, self.num_features] = self.scaler.fit_transform(self.df_test[self.num_features])
        test_dataset = CustomDataset(
            self.df_test,
            features=self.features,
            target=self.target,
            accelerator=self.accelerator
        )

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        return full_train_data_loader, test_loader
    
    def input_data_loader_for_prediction(self, num_workers):

        self.df_all.loc[:, self.num_features] = self.scaler.fit_transform(self.df_all.loc[:, self.num_features])

        full_dataset = CustomDataset(
            self.df_all,
            features=self.features,
            target=self.target,
            accelerator=self.accelerator
        )
        return DataLoader(full_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        