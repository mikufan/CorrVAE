import pandas as pd
from torch.utils.data import Dataset, Subset
import re
import logging
import numpy as np
from tqdm import tqdm
import sys

pd.options.mode.chained_assignment = None


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
    def __init__(self, sample_id, input_feat, target_feat, data_type, data_type_id, ptm_inputs):
        self.sample_id = sample_id
        self.feat_list = input_feat.index
        self.input_feat = input_feat.values
        self.target_feat = target_feat.values
        self.data_type = data_type
        self.data_type_id = data_type_id
        self.ptm_inputs = ptm_inputs
        if isinstance(ptm_inputs, pd.Series):
            self.ptm_inputs = self.ptm_inputs.values

        self.feat_mask = np.ones(len(input_feat))

        self.ptm_mask = None
        if self.ptm_inputs is not None:
            self.ptm_mask = np.ones(len(ptm_inputs))


def process_ptm_df(ptm_df):
    ptm_dict = {value: ptm_df.index[ptm_df['gene_idx'] == value].tolist() for value in ptm_df['gene_idx'].unique()}
    max_length = max(len(indices) for indices in ptm_dict.values())
    return ptm_dict, max_length


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


def create_sparse_array(shape, zero_percentage):
    # 创建全1数组
    arr = np.ones(shape)

    # 计算需要设置为0的元素数量
    total_elements = np.prod(shape)
    num_zeros = int(total_elements * zero_percentage)

    # 随机选择位置设置为0
    indices = np.random.choice(total_elements, num_zeros, replace=False)
    arr.ravel()[indices] = 0

    return arr


def scale_target(target_df, scale=1000000):
    scaled_target = target_df.map(lambda x: np.power(2, x)) / scale
    return scaled_target


def scale_input(input_df, scale=1000000):
    scaled_input = input_df.map(lambda x:np.power(2,x)) /scale
    return scaled_input


class OmicDataset(Dataset):
    def __init__(self, input_data, target_data, ptm_input, ac_input, use_ptm, use_ac, sample_dict, data_class,
                 data_type_dict=None, feat_dict=None, no_zeros=False, scale_target=False, scale_input=False):
        self.data_items = []
        self.input_df = pd.read_csv(input_data, index_col=0)
        self.target_df = pd.read_csv(target_data, index_col=0)
        self.sample_type_dict = read_sample_dict(sample_dict, data_class)
        self.no_zeros = no_zeros
        self.scale_target = scale_target
        self.scale_input = scale_input
        if data_type_dict is None:
            self.data_type_dict = dict()
        else:
            self.data_type_dict = data_type_dict
        if feat_dict is None:
            self.feat_list = []
        else:
            self.feat_list = self.get_feat_list(feat_dict)
        if use_ptm:
            self.ptm_df = pd.read_csv(ptm_input, index_col=0)
            self.ptm_df.replace([np.inf, -np.inf], 1e-5, inplace=True)
        else:
            self.ptm_df = None
        if use_ac:
            self.ac_df = pd.read_csv(ac_input, index_col=0)
        else:
            self.ac_df = None
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
        # if self.ptm_df is not None:
        #     ptm_dict, ptm_length = process_ptm_df(self.ptm_df)
        # else:
        #     ptm_dict = None
        #     ptm_length = 0
        # for s in input_samples:
        for s in tqdm(input_samples, mininterval=2, desc=' -Tot it %d' % len(input_samples),
                      leave=True, file=sys.stdout):
            # print(s)
            # TODO: merge in utils
            if self.sample_type_dict.get(s) is None:
                continue
            inputs = self.input_df[s]
            target = self.target_df[s]
            if self.scale_target:
                target = scale_target(target)
            if self.scale_input:
                inputs = scale_input(inputs)
            # small_value_indices = (inputs < 1e-4) | (target < 1e-4)
            small_value_indices = (target < 1e-4)
            if not self.no_zeros:
                # inputs[small_value_indices] = 1e-5
                target[small_value_indices] = 1e-5

            if self.ptm_df is not None and s in self.ptm_df.columns:
                ptm_inputs = self.ptm_df[s]
            elif self.ptm_df is not None:
                ptm_inputs = np.zeros(len(self.ptm_df))
            else:
                ptm_inputs = None
            ptm_small_value_idx_num = []
            # if ptm_inputs is not None and s in self.ptm_df.columns:
            #     small_value_genes = inputs.index[inputs == 1e-5]
            #     ptm_small_value_indices = self.ptm_df[self.ptm_df['gene_idx'].isin(small_value_genes)].index
            #     ptm_small_value_idx_num = self.ptm_df.index.get_indexer(ptm_small_value_indices)
            #     if not self.no_zeros:
            #         ptm_inputs[ptm_small_value_indices] = 1e-5
            # data_type: cancer type
            data_type = self.sample_type_dict[s]
            if update_dict:
                update_type_dict(data_type, self.data_type_dict)
            data_type_id = self.data_type_dict[data_type]
            data_item = DataItem(s, inputs, target, data_type, data_type_id, ptm_inputs)
            data_item.feat_mask[small_value_indices] = 0.0
            # if ptm_inputs is not None:
            #     data_item.ptm_mask[ptm_small_value_idx_num] = 0.0
            self.data_items.append(data_item)
