import pandas as pd
from torch.utils.data import Dataset, Subset
import re
import logging
import numpy as np


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def filter_near_zero(x, y, threshold=1e-4):
    mask = (np.abs(x) > threshold) & (np.abs(y) > threshold)
    return x[mask], y[mask]


def collate_fn(batch):
    return batch


class DataItem(object):
    def __init__(self, sample_id, input_feat, target_feat, data_type, data_type_id):
        self.sample_id = sample_id
        self.feat_list = input_feat.index
        self.input_feat = input_feat.values
        self.target_feat = target_feat.values
        self.data_type = data_type
        self.data_type_id = data_type_id


def read_sample_dict(sample_dict, data_class):
    sample_dict_df = pd.read_csv(sample_dict, index_col=0)
    sample_type_dict = {}
    for sample_type in sample_dict_df.index:
        if sample_type.split("_")[1] != data_class:
            continue
        sample_list = sample_dict_df.loc[sample_type, :].dropna()
        for s in sample_list:
            sample_type_dict[s] = sample_type.split("_")[0]
    return sample_type_dict


def update_type_dict(data_type, data_type_dict):
    index = len(data_type_dict)
    if data_type not in data_type_dict:
        data_type_dict[data_type] = index
        index += 1


def stratified_split(dataset, test_ratio, type_num, random_state=42):
    np.random.seed(random_state)

    # 获取所有的标签
    labels = [dataset[i].data_type_id for i in range(len(dataset))]

    # 为每种类型创建索引列表
    indices_per_type = {i: [] for i in range(type_num)}
    for idx, label in enumerate(labels):
        indices_per_type[label].append(idx)

    train_indices = []
    test_indices = []

    # 对每种类型进行分割
    for type_indices in indices_per_type.values():
        np.random.shuffle(type_indices)
        n_test = int(len(type_indices) * test_ratio)
        test_indices.extend(type_indices[:n_test])
        train_indices.extend(type_indices[n_test:])

    # 再次打乱以确保随机性
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    # 创建 Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


class OmicDataset(Dataset):
    def __init__(self, input_data, target_data, sample_dict, data_class, data_type_dict=None, feat_dict=None):
        self.data_items = []
        self.input_df = pd.read_csv(input_data, index_col=0)
        self.target_df = pd.read_csv(target_data, index_col=0)
        self.sample_type_dict = read_sample_dict(sample_dict, data_class)
        if data_type_dict is None:
            self.data_type_dict = dict()
        else:
            self.data_type_dict = data_type_dict
        if feat_dict is None:
            self.feat_list = []
        else:
            self.feat_list = self.get_feat_list(feat_dict)
        self.read_data(data_class)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        return self.data_items[idx]

    def get_feat_list(self, feat_dict):
        feat_list = []
        feat_df = pd.read_csv(feat_dict, index_col=0)
        for feat_idx in self.input_df.index:
            if feat_idx in feat_df['gene_id'].values:
                feat_name = feat_df[feat_df['gene_id'] == feat_idx]['gene_name'].values[0]
            else:
                feat_name = feat_idx
            feat_list.append(feat_name)
        return feat_list

    def read_data(self, data_class):
        if data_class == "NORMAL":
            input_samples = [col for col in self.input_df.columns if col.endswith('.N')]
        else:
            input_samples = [col for col in self.input_df.columns if not col.endswith('.N')]
        update_dict = False
        if len(self.data_type_dict) == 0:
            update_dict = True
        for s in input_samples:
            inputs = self.input_df[s]
            target = self.target_df[s]
            small_value_indices = (inputs < 1e-4) | (target < 1e-4)
            inputs[small_value_indices] = 1e-5
            target[small_value_indices] = 1e-5
            # data_type: cancer type
            data_type = self.sample_type_dict[s]
            if update_dict:
                update_type_dict(data_type, self.data_type_dict)
            data_type_id = self.data_type_dict[data_type]
            data_item = DataItem(s, inputs, target, data_type, data_type_id)
            self.data_items.append(data_item)
