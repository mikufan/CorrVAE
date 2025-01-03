import utils
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import hdbscan
import pandas as pd
from sklearn_extra.cluster import KMedoids
import umap
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import r2_score


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # print(recon_loss)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(KLD)
    mean_loss = recon_loss + KLD / logvar.numel()
    return mean_loss


class VAEPairModel(nn.Module):
    def __init__(self, corr_vae_model, hidden_dim, latent_dim, embedding_dim, n_feats, type_embedding_dim,
                 device, n_ptms=0, n_acs=0, no_type=False):
        super(VAEPairModel, self).__init__()

        self.device = device
        self.corr_vae_model = corr_vae_model
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
        self.dropout = nn.Dropout(0.3)

        # Embedding
        self.type_embedding = corr_vae_model.type_embedding
        # Decoder
        # self.fc3_x = nn.Linear(latent_dim + type_embedding_dim, hidden_dim)
        # self.fc3_y = nn.Linear(latent_dim + type_embedding_dim, hidden_dim)
        self.fc3_x = nn.Linear(latent_dim + type_embedding_dim, embedding_dim)
        self.fc3_y = nn.Linear(latent_dim + type_embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(hidden_dim, n_feats)
        self.nc_fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.decode_embedding = nn.Linear(embedding_dim, n_feats)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger = None

    def forward(self, x, y, z, data_type, ph_x=None, ac_x=None, is_train=True, no_type=False):
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.detach()
        z = z.to(self.device)
        if ph_x is not None:
            ph_x = ph_x.to(self.device)
        data_type = data_type.to(self.device)
        emb_type = self.type_embedding(data_type)
        recon_x, recon_y = self.decode(z, x, y, emb_type, no_type)
        if is_train:
            recon_loss_x = nn.functional.mse_loss(recon_x, x, reduction='mean')
            recon_loss_y = nn.functional.mse_loss(recon_y, y, reduction='mean')
            loss = recon_loss_x + recon_loss_y
            return loss
        else:
            return recon_x, x, recon_y, y

    def decode(self, z, x, y, emb_type, no_type=False):
        type_z = torch.cat((z, emb_type), dim=1)
        h3_x = torch.relu(self.fc3_x(type_z))
        h3_y = torch.relu(self.fc3_y(type_z))
        # z_numpy = z.cpu().detach().numpy()
        # h3_numpy = self.fc3.weight.data.cpu().detach().numpy()
        # result = np.dot(z_numpy, h3_numpy)
        recon_x = self.decode_embedding(h3_x)
        recon_y = self.decode_embedding(h3_y)
        # h4 = self.fc4(h3)
        # h4 = h4.unsqueeze(-1)
        # emb_feat = self.protein_embedding(gene_feats)
        # recon_y = h4 * emb_feat
        # y = y.unsqueeze(-1)
        # emb_y = y * emb_feat
        # emb_y = emb_y.view(emb_feat.shape[0], -1)
        # recon_y = recon_y.view(emb_feat.shape[0], -1)
        # return torch.sigmoid(self.fc4(h3))
        return recon_x, recon_y

    def pair_model_training(self, train_data, n_epochs, valid_data, model_path, no_type=False):
        self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger.info("Model constructed.")
        self.logger.info("Start training ...")
        best_result = 0.0
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
                batch_ph_input = [d.ph_inputs for d in data_batch] \
                    if train_data.dataset.ph_df is not None else None
                batch_input = torch.tensor(batch_input, dtype=torch.float32)
                batch_target = torch.tensor(batch_target, dtype=torch.float32)
                batch_data_type = torch.tensor(batch_data_type)
                if batch_ph_input is not None:
                    batch_ph_input = torch.tensor(batch_ph_input, dtype=torch.float32)
                batch_z = self.corr_vae_model.z_forward(batch_input, batch_data_type, ph_x=batch_ph_input)
                self.optimizer.zero_grad()

                batch_loss = self.forward(batch_input, batch_target, batch_z, batch_data_type, batch_ph_input,
                                          no_type=no_type)
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss
            self.logger.info("Loss in epoch " + str(n) + ": " + str(total_loss.cpu().detach().numpy() / n_batch))
            valid_data = train_data
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
                valid_batch_ph_input = [d.ph_inputs for d in valid_data_batch] \
                    if valid_data.dataset.ph_df is not None else None
                valid_batch_input = torch.tensor(valid_batch_input, dtype=torch.float32)
                valid_batch_target = torch.tensor(valid_batch_target, dtype=torch.float32)
                valid_batch_data_type = torch.tensor(valid_batch_data_type)
                if valid_batch_ph_input is not None:
                    valid_batch_ph_input = torch.tensor(valid_batch_ph_input, dtype=torch.float32)
                valid_batch_z = self.corr_vae_model.z_forward(valid_batch_input, valid_batch_data_type,
                                                              ph_x=valid_batch_ph_input)
                preds_x, targets_x, preds_y, targets_y = self.forward(valid_batch_input, valid_batch_target,
                                                                      valid_batch_z, valid_batch_data_type,
                                                                      ph_x=valid_batch_ph_input, is_train=False,
                                                                      no_type=no_type)
                preds_x = preds_x.cpu().detach().numpy()
                targets_x = targets_x.cpu().detach().numpy()
                preds_y = preds_y.cpu().detach().numpy()
                targets_y = targets_y.cpu().detach().numpy()
                for i in range(batch_size):
                    all_targets.append(np.concatenate((targets_x[i], targets_y[i])))
                    all_preds.append(np.concatenate((preds_x[i], preds_y[i])))

            all_targets = np.hstack(all_targets)
            all_preds = np.hstack(all_preds)

            indices = np.where(all_targets != 1e-5)[0]
            all_targets = all_targets[all_targets != 1e-5]
            all_preds = all_preds[indices]
            #     slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_preds)
            #     total_r2 = r_value ** 2
            # else:
            # slope, intercept, r_value, p_value, std_err = stats.linregress(all_targets, all_preds)
            # total_r2 = r_value ** 2
            total_r2 = r2_score(all_targets, all_preds)
            self.logger.info("Total validation R2 score in epoch " + str(n) + ": " + str(total_r2))
            # if total_r2 > best_result:
            #     best_result = total_r2
            #     torch.save(self, model_path + "/trained_corr_ptm_vae_model.pt")
            torch.save(self, model_path + "/trained_corr_pair_vae_model.pt")
            # self.logger.info("Best result so far: " + str(best_result))

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
        np.savetxt(f"../output/vae_pair_model/{class_name}.txt", data_2d, fmt='%f', delimiter=' ')
        if cluster_method == "dbscan":
            method_name = "DBSCAN"
            clusterer = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=5, prediction_data=True).fit(data_2d)
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
            plt.savefig(f"../output/model_fig/vae_pair_embeddings_{num_clusters}_{class_name}_{method_name}")
        else:
            plt.title('UMAP Visualization of Gene Embeddings')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.legend(title='Cluster')
            plt.savefig(f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{class_name}_{method_name}")
        feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        feat_cluster_df.to_csv(
            f"../output/clusters/vae_pair_gene_clusters_{num_clusters}_{class_name}_{method_name}.csv")
