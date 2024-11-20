import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import r2_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
import umap
from tqdm import tqdm
import sys
import pandas as pd
import utils
import hdbscan
from sklearn_extra.cluster import KMedoids
import seaborn as sns
from scipy import stats


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # print(recon_loss)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(KLD)
    mean_loss = recon_loss + KLD / logvar.numel()
    return mean_loss


class CorrPtmVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, embedding_dim, n_feats, type_embedding_dim, n_types,
                 device, n_ptms=0, n_acs=0, no_type=False):
        super(CorrPtmVAE, self).__init__()

        self.device = device
        # Encoder
        self.feat_fc = nn.Linear(n_feats, hidden_dim)
        self.fc1 = nn.Linear(n_feats * embedding_dim, hidden_dim)
        # self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        if n_ptms == 0 and n_acs == 0:
            mid_dim = hidden_dim
        elif (n_ptms != 0 and n_acs == 0) or (n_ptms == 0 and n_acs != 0):
            mid_dim = 2 * hidden_dim
        else:
            mid_dim = 3 * hidden_dim
        self.merge_hidden = nn.Sequential(nn.Linear(mid_dim, hidden_dim),
                                          nn.ReLU())
        if no_type:
            type_embedding_dim = 0
        self.fc2_mu = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        self.nc_fc1 = nn.Linear(embedding_dim, hidden_dim)

        # cnn encoder
        self.ptm_encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.mlp_ptm_encoder = nn.Sequential(
            nn.Linear(n_ptms, hidden_dim),
            nn.ReLU()
        )

        self.n_feats = n_feats
        self.n_ptms = n_ptms
        # self.ptm_encoder = nn.Sequential(
        #     nn.Linear(n_ptms, 4*hidden_dim),
        #     # nn.ReLU()
        # )
        self.dropout = nn.Dropout(0.3)

        # Embedding
        self.protein_embedding = nn.Embedding(n_feats, embedding_dim)
        self.type_embedding = nn.Embedding(n_types, type_embedding_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim + type_embedding_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_feats)
        self.nc_fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.decode_embedding = nn.Linear(embedding_dim, n_feats)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger = None

    def encode(self, x, data_type, ptm_x=None, no_type=False):
        h1 = torch.relu(self.feat_fc(x))
        if self.n_ptms > 0:
            ptm_x = torch.unsqueeze(ptm_x, 1)
            # ptm_x_num = ptm_x.cpu().detach().numpy()
            # ptm_h = self.ptm_encoder(ptm_x)
            ptm_h = self.mlp_ptm_encoder(ptm_x)
            # ptm_h = torch.squeeze(ptm_h, 2)
            ptm_h = torch.squeeze(ptm_h, 1)
            h1 = torch.cat((h1, ptm_h), 1)
            h1 = self.merge_hidden(h1)
            h1 = torch.relu(h1)
        emb_type = self.type_embedding(data_type)
        if not no_type:
            h1 = torch.cat((h1, emb_type), 1)
        return self.fc2_mu(h1), self.fc2_logvar(h1), emb_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, emb_types, no_type):
        if not no_type:
            h3 = torch.cat((z, emb_types), 1)
        else:
            h3 = z
        h3 = torch.relu(self.fc3(h3))
        h3 = self.dropout(h3)
        # z_numpy = z.cpu().detach().numpy()

        recon_y = self.decode_embedding(h3)
        return recon_y

    def forward(self, x, y, data_type, mask, ptm_x=None, is_train=True, use_mask=False, no_type=False,
                input_mask=None, return_z=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if input_mask is not None:
            input_mask = input_mask.to(self.device)
        # if not use_mask:
        #     x = torch.mul(x, mask)
        #     x = torch.mul(x, input_mask)
        if ptm_x is not None:
            ptm_x = ptm_x.to(self.device)
        data_type = data_type.to(self.device)
        mu, logvar, emb_type = self.encode(x, data_type, ptm_x, no_type)
        logvar = torch.clamp(logvar, min=-10, max=10)

        # log_var_numpy = logvar.cpu().detach().numpy()
        # mu_var_numpy = mu.cpu().detach().numpy()
        z = self.reparameterize(mu, logvar)
        recon_y = self.decode(z, emb_type, no_type)
        if use_mask:
            recon_y = torch.mul(recon_y, mask)
        if is_train:
            vae_loss = vae_loss_function(recon_y, y, mu, logvar)
            return vae_loss
        else:
            if not return_z:
                return recon_y, y
            else:
                return z, recon_y, y

    def model_training(self, train_data, n_epochs, valid_data, model_path, use_mask=False, no_type=False):
        self.logger = utils.get_logger('output/log/train_log.log')
        self.logger.info("Model constructed.")
        self.logger.info("Start training ...")
        best_result = 0.0
        # n_feats = self.gene_embedding.weight.data.shape[0]
        # input_genes = torch.tensor([i for i in range(self.n_feats)])
        for n in range(n_epochs):
            self.train()
            n_batch = len(train_data)
            total_loss = 0.0
            for data_batch in tqdm(train_data, mininterval=2, desc=' -Tot it %d' % n_batch,
                                   leave=True, file=sys.stdout):
                batch_size = len(data_batch)
                batch_input = [d.input_feat for d in data_batch]
                batch_target = [d.target_feat for d in data_batch]
                batch_data_type = [d.data_type_id for d in data_batch]
                if use_mask:
                    batch_mask = [d.feat_mask for d in data_batch]
                else:
                    # batch_mask = utils.create_sparse_array((batch_size, self.n_feats), 0.3)
                    batch_mask = None
                batch_input_mask = utils.create_sparse_array((batch_size, self.n_feats), 0.8)
                batch_ptm_input = [d.ptm_inputs for d in data_batch] if self.n_ptms > 0 else None
                batch_input = torch.tensor(batch_input, dtype=torch.float32)
                batch_target = torch.tensor(batch_target, dtype=torch.float32)
                batch_data_type = torch.tensor(batch_data_type)
                if batch_mask is not None:
                    batch_mask = torch.tensor(batch_mask, dtype=torch.float32)
                batch_input_mask = torch.tensor(batch_input_mask, dtype=torch.float32)
                if batch_ptm_input is not None:
                    batch_ptm_input = torch.tensor(batch_ptm_input, dtype=torch.float32)
                self.optimizer.zero_grad()
                batch_loss = self.forward(batch_input, batch_target, batch_data_type, batch_mask, batch_ptm_input,
                                          use_mask=use_mask, no_type=no_type)
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss
            self.logger.info("Loss in epoch " + str(n) + ": " + str(total_loss.cpu().detach().numpy() / n_batch))
            n_valid_batch = len(valid_data)
            self.eval()
            all_targets = []
            all_preds = []
            for valid_data_batch in tqdm(valid_data, mininterval=2, desc=' -Tot it %d' % n_valid_batch,
                                         leave=True, file=sys.stdout):
                batch_size = len(valid_data_batch)
                valid_batch_input = [d.input_feat for d in valid_data_batch]
                valid_batch_target = [d.target_feat for d in valid_data_batch]
                valid_batch_data_type = [d.data_type_id for d in valid_data_batch]
                if use_mask:
                    valid_batch_mask = [d.feat_mask for d in valid_data_batch]
                else:
                    # valid_batch_mask = np.ones((batch_size, self.n_feats))
                    valid_batch_mask = None
                # valid_batch_input_mask = np.ones((batch_size, self.n_feats))
                valid_batch_ptm_input = [d.ptm_inputs for d in valid_data_batch] if self.n_ptms > 0 else None
                if valid_batch_ptm_input is not None:
                    valid_batch_ptm_input = torch.tensor(valid_batch_ptm_input, dtype=torch.float32)
                valid_batch_input = torch.tensor(valid_batch_input, dtype=torch.float32)
                valid_batch_target = torch.tensor(valid_batch_target, dtype=torch.float32)
                if valid_batch_mask is not None:
                    valid_batch_mask = torch.tensor(valid_batch_mask, dtype=torch.float32)
                # valid_batch_input_mask = torch.tensor(valid_batch_input_mask, dtype=torch.float32)
                valid_batch_data_type = torch.tensor(valid_batch_data_type)
                preds, targets = self.forward(valid_batch_input, valid_batch_target, valid_batch_data_type,
                                              valid_batch_mask, valid_batch_ptm_input, is_train=False,
                                              use_mask=use_mask, no_type=no_type)
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                for i in range(batch_size):
                    all_targets.append(targets[i])
                    all_preds.append(preds[i])
            all_targets = np.hstack(all_targets)
            all_preds = np.hstack(all_preds)
            # if use_mask:
            #
            indices = np.where(all_targets != 1e-5)[0]
            all_targets = all_targets[all_targets != 1e-5]
            all_preds = all_preds[indices]
            #     slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_preds)
            #     total_r2 = r_value ** 2
            # else:
            slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_preds)
            total_r2 = r_value ** 2
            self.logger.info("Total validation R2 score in epoch " + str(n) + ": " + str(total_r2))
            if total_r2 > best_result:
                best_result = total_r2
                torch.save(self, model_path + "/trained_corr_ptm_vae_model.pt")
            # torch.save(self, model_path + "/trained_corr_ptm_vae_model.pt")
            self.logger.info("Best result so far: " + str(best_result))

    def model_test(self, data_loader, result_path=None, use_mask=False, no_type=False, return_z=False):
        n_batch = len(data_loader)
        data_type_dict = data_loader.dataset.dataset.data_type_dict
        id_2_type_dict = {}
        for key, value in data_type_dict.items():
            id_2_type_dict[value] = key
        self.eval()
        all_preds = []
        all_targets = []
        all_types = []
        all_inputs = []
        n_feats = self.n_feats
        # n_feats = self.gene_embedding.weight.data.shape[0]
        # input_genes = torch.tensor([i for i in range(self.n_feats)])
        for test_data_batch in tqdm(data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                    leave=True, file=sys.stdout):
            batch_size = len(test_data_batch)
            batch_input = [d.input_feat for d in test_data_batch]
            batch_target = [d.target_feat for d in test_data_batch]
            batch_data_type = [d.data_type_id for d in test_data_batch]

            if use_mask:
                batch_mask = [d.feat_mask for d in test_data_batch]
            else:
                # batch_mask = np.ones((batch_size, self.n_feats))
                batch_mask = None
            # batch_input_mask = np.ones((batch_size, self.n_feats))
            batch_ptm_input = [d.ptm_inputs for d in test_data_batch] if self.n_ptms > 0 else None
            batch_input = torch.tensor(batch_input, dtype=torch.float32)
            batch_target = torch.tensor(batch_target, dtype=torch.float32)
            batch_data_type = torch.tensor(batch_data_type)
            if batch_mask is not None:
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32)
            if batch_ptm_input is not None:
                batch_ptm_input = torch.tensor(batch_ptm_input, dtype=torch.float32)
            preds, targets = self.forward(batch_input, batch_target, batch_data_type, batch_mask,
                                          batch_ptm_input, is_train=False, use_mask=use_mask, no_type=no_type)
            preds = preds.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            for i in range(batch_size):
                all_targets.append(targets[i])
                all_preds.append(preds[i])
                all_types.append(batch_data_type[i])
                all_inputs.append(batch_input[i].cpu().detach().numpy())
        all_targets_array = np.hstack(all_targets)
        all_preds_array = np.hstack(all_preds)

        all_targets_df = pd.DataFrame(all_targets)
        all_preds_df = pd.DataFrame(all_preds)
        all_inputs_df = pd.DataFrame(all_inputs)
        # if use_mask:
        indices = np.where(all_targets_array != 1e-5)[0]
        r2_targets = all_targets_array[all_targets_array != 1e-5]
        r2_preds = all_preds_array[indices]
        # total_r2 = r2_score(r2_targets, r2_preds)
        slope, intercept, r_value, p_value, std_err = stats.linregress(r2_targets, r2_preds)
        total_r2 = r_value ** 2
        #     # print(f"R2:{r_squared}")
        # else:
        # slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_preds)
        # total_r2 = r_value ** 2
        # total_r2 = r2_score(all_targets, all_preds)
        res_array = np.column_stack((all_targets_array, all_preds_array))
        res_df = pd.DataFrame(res_array, columns=['target', 'prediction'])
        res_df.to_csv(result_path + ".csv")
        all_diffs = (all_targets_array - all_preds_array) ** 2
        all_diffs = all_diffs.reshape(-1, n_feats)
        all_types = np.hstack(all_types)
        df = pd.DataFrame(all_diffs)
        all_type_names = [id_2_type_dict[type_id] for type_id in all_types]
        df['label'] = all_type_names
        grouped_means = df.groupby('label').mean()
        grouped_means_df = pd.DataFrame(grouped_means)
        grouped_means_df.columns = data_loader.dataset.dataset.feat_list
        grouped_means_df.to_csv(result_path + "_diff.csv")
        all_targets_df.columns = data_loader.dataset.dataset.feat_list
        all_preds_df.columns = data_loader.dataset.dataset.feat_list
        all_inputs_df.columns = data_loader.dataset.dataset.feat_list
        all_targets_df.index = all_type_names
        all_preds_df.index = all_type_names
        all_inputs_df.index = all_type_names
        all_targets_df.to_csv(result_path + "_targets.csv")
        all_preds_df.to_csv(result_path + "_preds.csv")
        all_inputs_df.to_csv(result_path + "_inputs.csv")
        return total_r2

    def visualize_emb(self, feat_list, model_class, method="tsne", cluster_method="kmeans", num_clusters=9):
        if model_class == 1:
            class_name = "TUMOR"
        else:
            class_name = "NORMAL"
        # embeds = self.protein_embedding.weight.data.cpu().detach().numpy()
        embeds = self.decode_embedding.weight.data.cpu().detach().numpy()
        print(embeds.shape)
        # 降维并可视化
        if method == "tsne":
            tsne = TSNE(n_components=2, random_state=42)
            data_2d = tsne.fit_transform(embeds)
        else:
            umap_reducer = umap.UMAP(n_components=2, random_state=42)
            data_2d = umap_reducer.fit_transform(embeds)
        # 聚类
        np.savetxt(f"output/corr_ptm_vae_model/{class_name}.txt", data_2d, fmt='%f', delimiter=' ')
        if cluster_method == "dbscan":
            method_name = "DBSCAN"
            clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30, prediction_data=True).fit(data_2d)
            # clusters = hdbscan.HDBSCAN(min_samples=15, min_cluster_size=100).fit_predict(data_2d)
            clusters = clusterer.fit_predict(data_2d)
            membership_probabilities = hdbscan.all_points_membership_vectors(clusterer)
            # 对于噪声点，找到最大概率的聚类并重新分配
            noise_points_indices = np.where(clusters == -1)[0]
            for idx in noise_points_indices:
                noise_point_probs = membership_probabilities[idx]
                if np.max(noise_point_probs) > 0:  # 如果有非零的聚类概率
                    clusters[idx] = np.argmax(noise_point_probs)  # 分配到概率最大的聚类
        elif cluster_method == "kmedoids":
            method_name = "KMEDOIDS"
            kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
            clusters = kmedoids.fit_predict(embeds)
        else:
            method_name = "KMEANS"
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeds)
        plt.figure(figsize=(10, 7))
        if cluster_method == "dbscan":
            unique_clusters = np.unique(clusters)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
            for cluster in unique_clusters:
                if cluster == -1:
                    # 噪声点
                    cluster_data = data_2d[clusters == cluster]
                    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Noise', s=6, c='black')
                else:
                    cluster_data = data_2d[clusters == cluster]
                    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[cluster],
                                label=f'Cluster {cluster + 1}', s=6)
        else:
            for i in range(num_clusters):
                cluster_data = data_2d[clusters == i]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', s=6)
        if method == "tsne":
            plt.title('t-SNE Visualization of Gene Embeddings')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig(f"../output/model_fig/corr_ptm_vae_embeddings_{num_clusters}_{class_name}_{method_name}")
        else:
            plt.title('UMAP Visualization of Gene Embeddings')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.legend(title='Cluster')
            plt.savefig(f"output/model_fig/corr_ptm_vae_umap_embeddings_{num_clusters}_{class_name}_{method_name}")
        feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        feat_cluster_df.to_csv(
            f"output/clusters/corr_ptm_vae_gene_clusters_{num_clusters}_{class_name}_{method_name}.csv")
