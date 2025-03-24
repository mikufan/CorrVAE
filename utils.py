import pandas as pd
from torch.utils.data import Dataset, Subset
import re
import logging
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
from scipy.stats import pearsonr


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


def calculate_slopes(A, B, threshold=1e-4):
    # 获取type的唯一值和需要计算的列名
    types = A.iloc[:, -1].unique()
    value_cols = A.columns[:-1]

    # 创建结果DataFrame
    result_df = pd.DataFrame(index=types, columns=value_cols)
    bias_df = pd.DataFrame(index=types, columns=value_cols)
    r2_df = pd.DataFrame(index=types, columns=value_cols)
    pred_df = pd.DataFrame(index=A.index, columns=A.columns)
    pred_df.iloc[:, -1] = A.iloc[:, -1]
    # 对每个type进行循环
    for t in types:
        # 获取当前type的行索引
        print("Calculating slope for " + t)
        mask = A.iloc[:, -1] == t

        # 对每列进行循环
        for col in tqdm(value_cols, mininterval=2, desc=' -Tot it %d' % len(value_cols),
                        leave=True, file=sys.stdout):
            # 提取对应的x和y值
            x = A.loc[mask, col]
            y = B.loc[mask, col]

            # # 筛选出大于阈值的数据点
            # 筛选出大于阈值的数据点
            valid_mask = (x >= threshold) & (y >= threshold)
            x_filtered = x[valid_mask].values.reshape(-1, 1)
            y_filtered = y[valid_mask].values
            filtered_index = valid_mask[valid_mask].index
            # 只有当有足够的有效数据点时才进行计算
            if len(x_filtered) >= 2:  # 至少需要2个点才能计算斜率
                # 计算线性回归的斜率
                model = LinearRegression()
                model.fit(x_filtered, y_filtered)
                preds = model.predict(x_filtered)
                slope = model.coef_[0]
                bias = model.intercept_
                # 存储斜率值
                result_df.at[t, col] = slope
                bias_df.at[t, col] = bias
                # pred_df.loc[filtered_index, col] = preds
                r2 = r2_score(y_filtered, preds)
                r2_df.at[t, col] = r2
    result_df = result_df.fillna(0)
    bias_df = bias_df.fillna(0)
    # pred_df = pred_df.fillna(0)
    r2_df = r2_df.fillna(0)
    return result_df, bias_df, r2_df


def calculate_mse(input, target, slope, bias, threshold=1e-4):
    # 初始化结果DataFrame
    result = pd.DataFrame(index=slope.index, columns=slope.columns)
    pred = pd.DataFrame(index=target.index, columns=target.columns)
    pred['type'] = target['type']
    # 获取数值列（除了type列）
    # value_cols = input.columns.difference(['type'])
    value_cols = input.columns[:-1]
    y_preds = []
    y_true = []

    # 对每个type计算
    for type_val in input['type'].unique():
        # 获取当前type的行
        mask = input['type'] == type_val
        print("Calculating MSE for " + type_val)
        type_index = mask[mask].index.values
        # 对每个数值列计算
        for col in tqdm(value_cols, mininterval=2, desc=' -Tot it %d' % len(value_cols),
                        leave=True, file=sys.stdout):
            # 获取当前type和列的值

            a_values = input.loc[mask, col]
            b_values = target.loc[mask, col]
            c_value = slope.loc[type_val, col]
            d_value = bias.loc[type_val, col]
            #
            # valid_mask = (a_values >= threshold) & (b_values >= threshold)
            # filtered_a_values = a_values[valid_mask]
            # filtered_b_values = b_values[valid_mask]

            # 计算预测值：a * c + d
            predicted = a_values * c_value + d_value
            pred.loc[type_index, col] = predicted
            y_preds.append(predicted.values)
            y_true.append(b_values.values)

            # 计算与实际值的均方误差
            mse = ((predicted - b_values) ** 2).mean()

            # 存储结果
            result.at[type_val, col] = mse
    y_true = np.hstack(y_true)
    y_pred = np.hstack(y_preds)
    result = result.fillna(0)
    # 计算R²
    r2 = r2_score(y_true, y_pred)
    pcc, pv = pearsonr(y_true, y_pred)
    return result, r2, pcc


class DataItem(object):
    def __init__(self, sample_id, input_feat, target_feat, data_type, data_type_id, ph_inputs=None,ac_inputs=None):
        self.sample_id = sample_id
        self.feat_list = input_feat.index
        self.input_feat = input_feat.values
        self.target_feat = target_feat.values
        self.data_type = data_type
        self.data_type_id = data_type_id

        self.ph_inputs = ph_inputs
        if isinstance(ph_inputs, pd.Series):
            self.ph_inputs = self.ph_inputs.values
        self.ac_inputs = ac_inputs
        if isinstance(ac_inputs, pd.Series):
            self.ac_inputs = self.ac_inputs.values


def read_sample_dict(sample_dict, data_class):
    sample_dict_df = pd.read_csv(sample_dict, index_col=0)
    sample_type_dict = {}
    for sample_type in sample_dict_df.index:
        if data_class != "TUMOR AND NORMAL":
            if sample_type.split("_")[1] != data_class:
                continue
            sample_list = sample_dict_df.loc[sample_type, :].dropna()
            for s in sample_list:
                sample_type_dict[s] = sample_type.split("_")[0]
        else:
            sample_list = sample_dict_df.loc[sample_type, :].dropna()
            for s in sample_list:
                sample_type_dict[s] = sample_type
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
    def __init__(self, input_data, target_data, sample_dict, data_class, data_type_dict=None, feat_dict=None,
                 ph_input=None, ac_input=None):
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

        if ph_input is not None:
            self.ph_df = pd.read_csv(ph_input, index_col=0)
            self.ph_df.replace([np.inf, -np.inf], 1e-5, inplace=True)
        else:
            self.ph_df = None
        if ac_input is not None:
            self.ac_df = pd.read_csv(ac_input, index_col=0)
        else:
            self.ac_df = None
        self.read_data(data_class)
        self.type_name_dict = {v: k for k, v in self.data_type_dict.items()}

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
        if data_class != "TUMOR AND NORMAL":
            if data_class == "NORMAL":
                input_samples = [col for col in self.input_df.columns if col.endswith('.N')]
            else:
                input_samples = [col for col in self.input_df.columns if not col.endswith('.N')]
        else:
            input_samples = [col for col in self.input_df.columns]
        update_dict = False
        if len(self.data_type_dict) == 0:
            update_dict = True
        for s in input_samples:
            # TODO: merge in utils
            if self.sample_type_dict.get(s) is None:
                continue
            inputs = self.input_df[s]
            target = self.target_df[s]
            small_value_indices = (inputs < 1e-4) | (target < 1e-4)
            # inputs[small_value_indices] = 1e-5
            # target[small_value_indices] = 1e-5
            # data_type: cancer type
            data_type = self.sample_type_dict[s]
            if update_dict:
                update_type_dict(data_type, self.data_type_dict)
            # ph_input
            if self.ph_df is not None and s in self.ph_df.columns:
                ph_inputs = self.ph_df[s]
            elif self.ph_df is not None:
                ph_inputs = np.zeros(len(self.ph_df))
            else:
                ph_inputs = None

            if self.ac_df is not None and s in self.ac_df.columns:
                ac_inputs = self.ac_df[s]
            elif self.ph_df is not None:
                ac_inputs = np.zeros(len(self.ac_df))
            else:
                ac_inputs = None

            data_type_id = self.data_type_dict[data_type]
            data_item = DataItem(s, inputs, target, data_type, data_type_id, ph_inputs=ph_inputs, ac_inputs=ac_inputs)
            self.data_items.append(data_item)

    def get_linear(self, indices):
        train_data_items = []
        input_samples = []
        for i in indices:
            train_data_items.append(self.data_items[i])
        for item in train_data_items:
            input_samples.append(item.sample_id)
        train_input_df = self.input_df.loc[:, input_samples]
        train_target_df = self.target_df.loc[:, input_samples]

        train_input_df.index = self.feat_list
        train_target_df.index = self.feat_list
        t_train_input_df = train_input_df.T
        t_train_target_df = train_target_df.T
        sample_type_list = []
        for s in t_train_input_df.index:
            sample_type = self.sample_type_dict.get(s)
            sample_type_list.append(sample_type)
        t_train_input_df["type"] = sample_type_list
        t_train_target_df["type"] = sample_type_list
        slope_df, bias_df, r2_df = calculate_slopes(t_train_input_df, t_train_target_df)
        return slope_df, bias_df, r2_df, t_train_target_df

    def get_linear_mse(self, indices, slope_df, bias_df):
        test_data_items = []
        input_samples = []
        for i in indices:
            test_data_items.append(self.data_items[i])
        for item in test_data_items:
            input_samples.append(item.sample_id)
        input_df = self.input_df.loc[:, input_samples]
        target_df = self.target_df.loc[:, input_samples]
        input_df.index = self.feat_list
        target_df.index = self.feat_list
        t_input_df = input_df.T
        t_target_df = target_df.T
        sample_type_list = []
        for s in t_input_df.index:
            sample_type = self.sample_type_dict.get(s)
            sample_type_list.append(sample_type)
        t_input_df["type"] = sample_type_list
        t_target_df["type"] = sample_type_list
        mse_df, r2, pcc = calculate_mse(t_input_df, t_target_df, slope_df, bias_df)
        return mse_df, r2, pcc
