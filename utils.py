import pandas as pd
from torch.utils.data import Dataset, Subset
import re
import logging
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
from scipy.stats import pearsonr,spearmanr


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
    def __init__(self, sample_id, input_feat, target_feat, data_type, data_type_id, ph_inputs=None, ac_inputs=None):
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
        if data_class != "TUMOR_AND_NORMAL":
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


def stratified_3_split(dataset, train_ratio=0.8, val_ratio=0.1, type_num=None, random_state=42):
    np.random.seed(random_state)

    dataset_size = len(dataset)
    labels = [dataset[i].data_type_id for i in range(dataset_size)]

    # 创建类别对应的索引字典
    if type_num is None:
        type_num = len(set(labels))
    indices_per_class = {i: [] for i in range(type_num)}
    for idx, label in enumerate(labels):
        indices_per_class[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_indices in indices_per_class.values():
        np.random.shuffle(class_indices)
        n_total = len(class_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        train_indices.extend(class_indices[:n_train])
        val_indices.extend(class_indices[n_train:n_train + n_val])
        test_indices.extend(class_indices[n_train + n_val:])

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    np.random.shuffle(test_indices)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


class OmicDataset(Dataset):
    def __init__(self, input_data, target_data, sample_dict, data_class, data_type_dict=None, feat_dict=None,
                 ph_input=None, ac_input=None, common_type=False,common_feat=None,ood_data=False,common_ph_feat=None):
        self.data_items = []
        if not ood_data:
            self.input_df = pd.read_csv(input_data, index_col=0)
        else:
            self.input_df = pd.read_csv(input_data, index_col=0, sep="\t") + 7
        self.target_df = pd.read_csv(target_data, index_col=0)
        if sample_dict is not None:
            self.sample_type_dict = read_sample_dict(sample_dict, data_class)
        else:
            self.sample_type_dict = None
        self.common_type = common_type
        if data_type_dict is None:
            self.data_type_dict = dict()
        else:
            self.data_type_dict = data_type_dict
        if feat_dict is None:
            self.feat_list = []
        else:
            if common_feat is None:
                self.feat_list = self.get_feat_list(feat_dict)
            else:
                self.feat_list = common_feat

        if ph_input is not None:
            if not ood_data:
                self.ph_df = pd.read_csv(ph_input, index_col=0)
                self.ph_df.replace([np.inf, -np.inf], 1e-5, inplace=True)
                self.ph_feat_list = self.get_ph_feat_list(feat_dict)
            else:
                self.ph_df = pd.read_csv(ph_input, index_col=0, sep='\t')
                self.ph_feat_list = common_ph_feat
        else:
            self.ph_df = None
            self.ph_feat_list = []
        if ac_input is not None:
            self.ac_df = pd.read_csv(ac_input, index_col=0)
            self.ac_feat_list = self.get_ac_feat_list()
        else:
            self.ac_df = None
            self.ac_feat_list = []
        if self.sample_type_dict is not None:

            self.read_data(data_class)
            self.type_name_dict = {v: k for k, v in self.data_type_dict.items()}
        else:
            self.read_ood_data(data_class)
            self.type_name_dict = None

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

    def get_ph_feat_list(self, feat_dict):
        feat_list = []
        feat_df = pd.read_csv(feat_dict, index_col=0)
        for feat_idx in self.ph_df.index:
            gene_idx, ph_idx, site, pep, num = feat_idx.split("|")
            if gene_idx in feat_df['gene_id'].values:
                gene_name = feat_df[feat_df['gene_id'] == gene_idx]['gene_name'].values[0]
            else:
                gene_name = gene_idx
            feat_name = gene_name + "_" + site + "_" + pep
            feat_list.append(feat_name)
        return feat_list

    def get_ac_feat_list(self):
        feat_list = []
        for feat_idx in self.ac_df.index:
            feat_list.append(feat_idx)
        return feat_list

    def read_data(self, data_class):
        if data_class != "TUMOR_AND_NORMAL":
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
            elif self.ac_df is not None:
                ac_inputs = np.zeros(len(self.ac_df))
            else:
                ac_inputs = None

            data_type_id = self.data_type_dict[data_type]
            common_type_list = [0, 1, 2, 3, 4, 5, 8, 9, 10, 14]
            if self.common_type:
                if data_type_id not in common_type_list:
                    continue
            data_item = DataItem(s, inputs, target, data_type, data_type_id, ph_inputs=ph_inputs, ac_inputs=ac_inputs)
            self.data_items.append(data_item)

    def read_ood_data(self, data_class):
        input_samples = [col for col in self.input_df.columns]
        target_samples = [col for col in self.target_df.columns]
        common_input_samples = [s for s in input_samples if s in target_samples]
        if self.ph_df is not None:
            common_input_samples = [s for s in common_input_samples if s in self.ph_df.columns]

        intersect_feat = []
        intersect_ph_feat = []
        intersect_ph = []
        self.target_df.index = self.target_df.index.str.split('|').str[0]
        for feat in self.feat_list:
            if feat in self.input_df.index.values and feat in self.target_df.index.values:
                intersect_feat.append(feat)
        if self.ph_df is not None:
            common_ph_feat_list = []
            self.ph_df = self.ph_df.iloc[:, 2:]
            for ph in self.ph_feat_list:
                common_ph_feat_list.append(ph.split("_")[0]+"_"+ph.split("_")[1])
            common_ph_feat_list = np.array(common_ph_feat_list)
            common_ph_feat_list = np.unique(common_ph_feat_list)
            data_ph_list = []
            for ph in self.ph_df.index.values:
                ph_feat = ph.split("_")[0]+"_"+ph.split("_")[-1]
                data_ph_list.append(ph_feat)
            self.ph_df.index = data_ph_list
            self.ph_df = self.ph_df[~self.ph_df.index.duplicated(keep='first')]
            for ph in self.ph_feat_list:
                ph_feat = ph.split("_")[0] + "_" + ph.split("_")[1]
                if ph_feat in data_ph_list:
                    intersect_ph_feat.append(ph_feat)
                    intersect_ph.append(ph)
            # for feat in
        input_sub = self.input_df.loc[intersect_feat, input_samples]
        target_sub = self.target_df.loc[intersect_feat, input_samples]
        input_sub = input_sub[~input_sub.index.duplicated(keep='first')]
        target_sub = target_sub[~target_sub.index.duplicated(keep='first')]
        input_sub = input_sub.reindex(index=self.feat_list, fill_value=1e-6)
        target_sub = target_sub.reindex(index=self.feat_list, fill_value=1e-6)
        ph_input_s = pd.DataFrame(np.full((len(self.ph_feat_list), len(common_input_samples)), 1e-6))
        if self.ph_df is not None:
            ph_input_sub = self.ph_df.loc[intersect_ph_feat, common_input_samples]
            ph_input_sub.index = intersect_ph
            ph_input_sub = ph_input_sub+20
            positions = [self.ph_feat_list.index(x) for x in intersect_ph if x in self.ph_feat_list]
            ph_input_s.index = self.ph_feat_list
            ph_input_s.columns = common_input_samples
            ph_input_s.iloc[positions] = ph_input_sub
        # for s in input_samples:
        for s in tqdm(input_samples, mininterval=2, desc=' -Tot it %d' % len(input_samples),
                        leave=True, file=sys.stdout):
            inputs = input_sub[s]
            target = target_sub[s]
            data_type = "UNKNOWN"
            data_type_id = -1
            # ph_input
            if self.ph_df is not None and s in common_input_samples:
                ph_inputs = ph_input_s[s]
            elif self.ph_df is not None:
                ph_inputs = np.zeros(len(self.ph_feat_list))
            else:
                ph_inputs = None

            if self.ac_df is not None and s in self.ac_df.columns:
                ac_inputs = self.ac_df[s]
            elif self.ac_df is not None:
                ac_inputs = np.zeros(len(self.ac_df))
            else:
                ac_inputs = None
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

    def linear_fit(self, indices, slope_df, bias_df, model_class, data_set):
        linear_mse_df, r2, pcc = self.get_linear_mse(indices, slope_df, bias_df)
        linear_mse_df.to_csv(
            f"../output/corr_vae_model/corr_vae_model_TUMOR_AND_NORMAL_{data_set}_linear_mse_{model_class}.csv")
        print(f"Linear {data_set} {model_class} r2: " + str(r2))
        print(f"Linear {data_set} {model_class} pcc: " + str(pcc))
