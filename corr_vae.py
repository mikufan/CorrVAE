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


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # print(recon_loss)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(KLD)
    mean_loss = recon_loss + KLD / logvar.numel()
    return mean_loss


class CorrVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, embedding_dim, n_feats, type_embedding_dim, n_types,
                 device, is_nc=False):
        super(CorrVAE, self).__init__()

        self.device = device
        # Encoder
        self.feat_fc = nn.Linear(n_feats, hidden_dim)
        self.fc1 = nn.Linear(n_feats * embedding_dim, hidden_dim)
        # self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        self.nc_fc1 = nn.Linear(embedding_dim, hidden_dim)

        # Embedding
        self.gene_embedding = nn.Embedding(n_feats, embedding_dim)
        self.protein_embedding = nn.Embedding(n_feats, embedding_dim)
        self.type_embedding = nn.Embedding(n_types, type_embedding_dim)
        # Decoder
        # self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(latent_dim + type_embedding_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_feats)
        self.nc_fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.decode_embedding = nn.Linear(embedding_dim, n_feats)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger = None

    def encode(self, x, data_type, gene_feats):
        h1 = torch.relu(self.feat_fc(x))
        emb_type = self.type_embedding(data_type)
        h1 = torch.cat((h1, emb_type), 1)
        return self.fc2_mu(h1), self.fc2_logvar(h1), emb_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps_numpy = eps.cpu().detach().numpy()
        return mu + eps * std

    def decode(self, z, y, emb_types):

        h3 = torch.cat((z, emb_types), 1)
        h3 = torch.relu(self.fc3(h3))
        recon_y = self.decode_embedding(h3)

        return recon_y, y

    def forward(self, x, y, data_type, gene_feats, is_train=True):
        x = x.to(self.device)
        y = y.to(self.device)
        data_type = data_type.to(self.device)
        gene_feats = gene_feats.to(self.device)
        mu, logvar, emb_type = self.encode(x, data_type, gene_feats)
        logvar = torch.clamp(logvar, min=-10, max=10)
        # log_var_numpy = logvar.cpu().detach().numpy()
        # mu_var_numpy = mu.cpu().detach().numpy()
        z = self.reparameterize(mu, logvar)
        recon_y, y = self.decode(z, y, emb_type)
        if is_train:
            vae_loss = vae_loss_function(recon_y, y, mu, logvar)
            return vae_loss
        else:
            return recon_y, y

    def model_training(self, train_data, n_epochs, valid_data, model_path):
        self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger.info("Model constructed.")
        self.logger.info("Start training ...")
        best_result = 0.0
        n_feats = self.gene_embedding.weight.data.shape[0]
        input_genes = torch.tensor([i for i in range(n_feats)])
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
                batch_input = torch.tensor(batch_input, dtype=torch.float32)
                batch_target = torch.tensor(batch_target, dtype=torch.float32)
                batch_input_genes = input_genes.unsqueeze(0).expand(batch_size, -1)
                batch_data_type = torch.tensor(batch_data_type)
                self.optimizer.zero_grad()
                batch_loss = self.forward(batch_input, batch_target, batch_data_type, batch_input_genes)
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
                valid_batch_input = torch.tensor(valid_batch_input, dtype=torch.float32)
                valid_batch_target = torch.tensor(valid_batch_target, dtype=torch.float32)
                valid_batch_data_type = torch.tensor(valid_batch_data_type)
                valid_batch_input_genes = input_genes.unsqueeze(0).expand(batch_size, -1)
                preds, targets = self.forward(valid_batch_input, valid_batch_target, valid_batch_data_type,
                                              valid_batch_input_genes, False)
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                for i in range(batch_size):
                    all_targets.append(targets[i])
                    all_preds.append(preds[i])
            all_targets = np.hstack(all_targets)
            all_preds = np.hstack(all_preds)
            total_r2 = r2_score(all_targets, all_preds)
            self.logger.info("Total validation R2 score in epoch " + str(n) + ": " + str(total_r2))
            if total_r2 > best_result:
                best_result = total_r2
                torch.save(self, model_path + "/trained_corr_vae_model.pt")

    def model_test(self, data_loader, result_path=None):
        n_batch = len(data_loader)
        data_type_dict = data_loader.dataset.dataset.data_type_dict
        id_2_type_dict = {}
        for key,value in data_type_dict.items():
            id_2_type_dict[value] = key
        self.eval()
        all_preds = []
        all_targets = []
        all_types = []
        n_feats = self.gene_embedding.weight.data.shape[0]
        input_genes = torch.tensor([i for i in range(n_feats)])
        for test_data_batch in tqdm(data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                    leave=True, file=sys.stdout):
            batch_size = len(test_data_batch)
            batch_input = [d.input_feat for d in test_data_batch]
            batch_target = [d.target_feat for d in test_data_batch]
            batch_data_type = [d.data_type_id for d in test_data_batch]
            batch_input = torch.tensor(batch_input, dtype=torch.float32)
            batch_target = torch.tensor(batch_target, dtype=torch.float32)
            batch_data_type = torch.tensor(batch_data_type)
            batch_input_genes = input_genes.unsqueeze(0).expand(batch_size, -1)
            preds, targets = self.forward(batch_input, batch_target, batch_data_type,
                                          batch_input_genes, False)
            preds = preds.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            for i in range(batch_size):
                all_targets.append(targets[i])
                all_preds.append(preds[i])
                all_types.append(batch_data_type[i])
        all_targets = np.hstack(all_targets)
        all_preds = np.hstack(all_preds)
        res_array = np.column_stack((all_targets, all_preds))
        res_df = pd.DataFrame(res_array, columns=['target', 'prediction'])
        res_df.to_csv(result_path+".csv")
        all_diffs = (all_targets-all_preds) ** 2
        all_diffs = all_diffs.reshape(-1, n_feats)
        all_types = np.hstack(all_types)
        df = pd.DataFrame(all_diffs)
        all_type_names = [id_2_type_dict[type_id] for type_id in all_types ]
        df['label'] = all_type_names
        grouped_means = df.groupby('label').mean()
        grouped_means_df = pd.DataFrame(grouped_means)
        grouped_means_df.columns = data_loader.dataset.dataset.feat_list
        grouped_means_df.to_csv(result_path+"_diff.csv")
        total_r2 = r2_score(all_targets, all_preds)
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
        np.savetxt(f"../output/corr_vae_model/{class_name}.txt", data_2d, fmt='%f', delimiter=' ')
        if cluster_method == "dbscan":
            method_name = "DBSCAN"
            clusterer = hdbscan.HDBSCAN(min_samples=25, min_cluster_size=100, prediction_data=True).fit(data_2d)
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
            for cluster in unique_clusters:
                if cluster == -1:
                    # 噪声点
                    cluster_data = data_2d[clusters == cluster]
                    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label='Noise', s=6, c='black')
                else:
                    cluster_data = data_2d[clusters == cluster]
                    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster + 1}', s=6)
        else:
            for i in range(num_clusters):
                cluster_data = data_2d[clusters == i]
                plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', s=6)
        if method == "tsne":
            plt.title('t-SNE Visualization of Gene Embeddings')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig(f"../output/model_fig/corr_model_vae_embeddings_{num_clusters}_{class_name}_{method_name}")
        else:
            plt.title('UMAP Visualization of Gene Embeddings')
            plt.xlabel('UMAP Component 1')
            plt.ylabel('UMAP Component 2')
            plt.legend(title='Cluster')
            plt.savefig(f"../output/model_fig/corr_model_vae_umap_embeddings_{num_clusters}_{class_name}_{method_name}")
        feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        feat_cluster_df.to_csv(f"../output/clusters/corr_model_vae_gene_clusters_{num_clusters}_{class_name}_{method_name}.csv")

    def visualize_spectral_emb(self, feat_list, model_class):
        embeds = self.decode_embedding.weight.data.cpu().detach().numpy()
        if model_class == 1:
            class_name = "TUMOR"
        else:
            class_name = "NORMAL"
        # 聚类
        num_clusters = 4
        spec_cluster = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
        clusters = spec_cluster.fit_predict(embeds)
        # 降维并可视化
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        data_2d = umap_reducer.fit_transform(embeds)

        plt.figure(figsize=(10, 7))
        for i in range(num_clusters):
            cluster_data = data_2d[clusters == i]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', s=6)

        plt.title('UMAP Visualization of Gene Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend(title='Cluster')
        plt.savefig(f"../output/model_fig/vae_umap_spec_embeddings_{num_clusters}_{class_name}")

        feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        feat_cluster_df.to_csv(f"../output/clusters/vae_gene_spec_clusters_{num_clusters}_{class_name}.csv")
