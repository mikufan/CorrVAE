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
from torch.utils.data import Dataset, Subset
import logging
from sklearn.metrics.pairwise import cosine_similarity


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # print(recon_loss)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(KLD)
    mean_loss = recon_loss + KLD / logvar.numel()
    return mean_loss


def get_um_idx(type_list, name_dict):
    data_type_names = []
    if isinstance(type_list, torch.Tensor):
        type_list = type_list.cpu().detach().numpy()
    normal_idx = []
    tumor_idx = []
    for i, t in enumerate(type_list):
        data_type_name = name_dict[t]
        if data_type_name.endswith("NORMAL"):
            normal_idx.append(i)
        else:
            tumor_idx.append(i)
    return tumor_idx, normal_idx


def get_common_um_idx(type_list, name_dict):
    if isinstance(type_list, torch.Tensor):
        type_list = type_list.cpu().detach().numpy()
    normal_idx = []
    tumor_idx = []
    for i, t in enumerate(type_list):
        data_type_name = name_dict[t]
        if data_type_name.endswith("NORMAL"):
            normal_idx.append(i)
        else:
            type_name = data_type_name.split("_")[0]
            type_list = ["CCRCC", "LSCC", "LUAD", "HNSCC", "PDAC"]
            if type_name in type_list:
                tumor_idx.append(i)
    return tumor_idx, normal_idx


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

    def forward_y(self, y, z, data_type, type_name_dict, ph_x=None, ac_x=None, is_train=True, no_type=False):
        y = y.to(self.device)
        z = z.detach()
        z = z.to(self.device)
        if ph_x is not None:
            ph_x = ph_x.to(self.device)
        if ac_x is not None:
            ac_x = ac_x.to(self.device)
        data_type = data_type.to(self.device)
        emb_type = self.type_embedding(data_type)
        um_decode = True if \
            self.decode_embedding.out_features > self.corr_vae_model.decode_embedding.out_features else False
        if um_decode:
            tumor_idx, normal_idx = get_um_idx(data_type, type_name_dict)
            recon_y = self.decode_y_um(z, emb_type, normal_idx, tumor_idx, no_type)
        else:
            recon_y = self.decode_y(z, emb_type, no_type)
        if is_train:
            recon_loss_y = nn.functional.mse_loss(recon_y, y, reduction='mean')
            loss = recon_loss_y
            return loss
        else:
            return recon_y, y

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

    def decode_y(self, z, emb_type, no_type=False):
        type_z = torch.cat((z, emb_type), dim=1)
        h3_y = torch.relu(self.fc3_y(type_z))

        recon_y = self.decode_embedding(h3_y)
        return recon_y

    def decode_y_um(self, z, emb_type, normal_idx, tumor_idx, no_type=False):
        type_z = torch.cat((z, emb_type), dim=1)
        h3_y = torch.relu(self.fc3_y(type_z))
        normal_idx = torch.tensor(normal_idx, dtype=torch.long)
        tumor_idx = torch.tensor(tumor_idx, dtype=torch.long)
        normal_idx = normal_idx.to(self.device)
        tumor_idx = tumor_idx.to(self.device)
        all_idx = torch.cat([normal_idx, tumor_idx])

        n_feats = int(self.decode_embedding.out_features / 2)
        all_recon_y = self.decode_embedding(h3_y)
        normal_recon_y = all_recon_y.index_select(dim=0, index=normal_idx)
        normal_recon_y = normal_recon_y[:, :n_feats]
        tumor_recon_y = all_recon_y.index_select(dim=0, index=tumor_idx)[:, n_feats:]
        recon_y = torch.cat([normal_recon_y, tumor_recon_y], dim=0)
        recon_y = recon_y[torch.argsort(all_idx)]
        return recon_y

    def pair_model_training(self, train_data, n_epochs, model_path, no_type=False, decode_type="all"):
        self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger.info("Model constructed.")
        self.logger.info("Start training ...")
        best_result = 0.0
        for n in range(n_epochs):
            self.train()
            n_batch = len(train_data)
            total_loss = 0.0
            train_ph_df = train_data.dataset.dataset.ph_df \
                if isinstance(train_data.dataset, Subset) else train_data.dataset.ph_df
            train_ac_df = train_data.dataset.dataset.ac_df \
                if isinstance(train_data.dataset, Subset) else train_data.dataset.ac_df
            type_name_dict = train_data.dataset.dataset.type_name_dict \
                if isinstance(train_data.dataset, Subset) else train_data.dataset.type_name_dict
            for data_batch in tqdm(train_data, mininterval=2, desc=' -Tot it %d' % n_batch,
                                   leave=True, file=sys.stdout):
                batch_size = len(data_batch)
                batch_input = [d.input_feat for d in data_batch]
                batch_target = [d.target_feat for d in data_batch]
                batch_data_type = [d.data_type_id for d in data_batch]
                batch_ph_input = [d.ph_inputs for d in data_batch] \
                    if train_ph_df is not None else None
                batch_ac_input = [d.ac_inputs for d in data_batch] \
                    if train_ac_df is not None else None
                batch_input = torch.tensor(batch_input, dtype=torch.float32)
                batch_target = torch.tensor(batch_target, dtype=torch.float32)
                batch_data_type = torch.tensor(batch_data_type)
                if batch_ph_input is not None:
                    batch_ph_input = torch.tensor(batch_ph_input, dtype=torch.float32)
                if batch_ac_input is not None:
                    batch_ac_input = torch.tensor(batch_ac_input, dtype=torch.float32)
                batch_z = self.corr_vae_model.z_forward(batch_input, batch_data_type, ph_x=batch_ph_input,
                                                        ac_x=batch_ac_input)
                self.optimizer.zero_grad()
                if decode_type == "y":
                    batch_loss = self.forward_y(batch_target, batch_z, batch_data_type, type_name_dict, batch_ph_input,
                                                batch_ac_input, no_type=no_type)
                else:
                    batch_loss = self.forward(batch_input, batch_target, batch_z, batch_data_type, batch_ph_input,
                                              batch_ac_input, no_type=no_type)
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss
            self.logger.info("Loss in epoch " + str(n) + ": " + str(total_loss.cpu().detach().numpy() / n_batch))
            valid_data = train_data
            n_valid_batch = len(valid_data)
            self.eval()
            all_targets = []
            all_preds = []
            all_zs = []
            for valid_data_batch in tqdm(valid_data, mininterval=2, desc=' -Tot it %d' % n_valid_batch,
                                         leave=True, file=sys.stdout):
                batch_size = len(valid_data_batch)
                valid_batch_input = [d.input_feat for d in valid_data_batch]
                valid_batch_target = [d.target_feat for d in valid_data_batch]
                valid_batch_data_type = [d.data_type_id for d in valid_data_batch]
                valid_batch_ph_input = [d.ph_inputs for d in valid_data_batch] \
                    if train_ph_df is not None else None
                valid_batch_ac_input = [d.ac_inputs for d in valid_data_batch] \
                    if train_ac_df is not None else None
                valid_batch_input = torch.tensor(valid_batch_input, dtype=torch.float32)
                valid_batch_target = torch.tensor(valid_batch_target, dtype=torch.float32)
                valid_batch_data_type = torch.tensor(valid_batch_data_type)
                if valid_batch_ph_input is not None:
                    valid_batch_ph_input = torch.tensor(valid_batch_ph_input, dtype=torch.float32)
                if valid_batch_ac_input is not None:
                    valid_batch_ac_input = torch.tensor(valid_batch_ac_input, dtype=torch.float32)
                valid_batch_z = self.corr_vae_model.z_forward(valid_batch_input, valid_batch_data_type,
                                                              ph_x=valid_batch_ph_input,ac_x=valid_batch_ac_input)
                if decode_type == "y":
                    preds_y, targets_y = self.forward_y(valid_batch_target, valid_batch_z, valid_batch_data_type,
                                                        type_name_dict, ph_x=valid_batch_ph_input,
                                                        ac_x=valid_batch_ac_input, is_train=False, no_type=no_type)
                    preds_y = preds_y.cpu().detach().numpy()
                    targets_y = targets_y.cpu().detach().numpy()
                    for i in range(batch_size):
                        all_targets.append(targets_y[i])
                        all_preds.append(preds_y[i])
                else:
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
            total_r2 = r2_score(all_targets, all_preds)
            self.logger.info("Total validation R2 score in epoch " + str(n) + ": " + str(total_r2))
            pcc, _ = stats.pearsonr(all_targets, all_preds)
            print("PCC in epoch " + str(n) + ": " + str(pcc))
            # if total_r2 > best_result:
            #     best_result = total_r2
            #     torch.save(self, model_path + "/trained_corr_ptm_vae_model.pt")
            torch.save(self, model_path + "/trained_corr_pair_vae_model.pt")
            # self.logger.info("Best result so far: " + str(best_result))

    def pair_model_inference(self, infer_data, no_type=False, decode_type="all"):
        self.logger = utils.get_logger('../output/log/infer_log.log')
        self.logger.info("Start inference ...")
        n_infer_batch = len(infer_data)
        self.eval()
        all_targets = []
        all_preds = []
        all_zs = []
        all_types = []
        infer_ph_df = infer_data.dataset.dataset.ph_df \
            if isinstance(infer_data.dataset, Subset) else infer_data.dataset.ph_df
        type_name_dict = infer_data.dataset.dataset.type_name_dict \
            if isinstance(infer_data.dataset, Subset) else infer_data.dataset.type_name_dict
        for infer_data_batch in tqdm(infer_data, mininterval=2, desc=' -Tot it %d' % n_infer_batch,
                                     leave=True, file=sys.stdout):
            batch_size = len(infer_data_batch)
            infer_batch_input = [d.input_feat for d in infer_data_batch]
            infer_batch_target = [d.target_feat for d in infer_data_batch]
            infer_batch_data_type = [d.data_type_id for d in infer_data_batch]
            infer_batch_ph_input = [d.ph_inputs for d in infer_data_batch] \
                if infer_ph_df is not None else None
            infer_batch_input = torch.tensor(infer_batch_input, dtype=torch.float32)
            infer_batch_target = torch.tensor(infer_batch_target, dtype=torch.float32)
            infer_batch_data_type = torch.tensor(infer_batch_data_type)
            if infer_batch_ph_input is not None:
                infer_batch_ph_input = torch.tensor(infer_batch_ph_input, dtype=torch.float32)
            infer_batch_z = self.corr_vae_model.z_forward(infer_batch_input, infer_batch_data_type,
                                                          ph_x=infer_batch_ph_input)
            if decode_type == "y":
                preds_y, targets_y = self.forward_y(infer_batch_target, infer_batch_z, infer_batch_data_type,
                                                    type_name_dict, ph_x=infer_batch_ph_input, is_train=False,
                                                    no_type=no_type)
                preds_y = preds_y.cpu().detach().numpy()
                targets_y = targets_y.cpu().detach().numpy()
                for i in range(batch_size):
                    all_targets.append(targets_y[i])
                    all_preds.append(preds_y[i])
            else:
                preds_x, targets_x, preds_y, targets_y = self.forward(infer_batch_input, infer_batch_target,
                                                                      infer_batch_z, infer_batch_data_type,
                                                                      ph_x=infer_batch_ph_input, is_train=False,
                                                                      no_type=no_type)
                preds_x = preds_x.cpu().detach().numpy()
                targets_x = targets_x.cpu().detach().numpy()
                preds_y = preds_y.cpu().detach().numpy()
                targets_y = targets_y.cpu().detach().numpy()
                for i in range(batch_size):
                    all_targets.append(np.concatenate((targets_x[i], targets_y[i])))
                    all_preds.append(np.concatenate((preds_x[i], preds_y[i])))
            all_zs.append(infer_batch_z.cpu().detach().numpy())
            all_types.append(infer_batch_data_type)
        all_targets = np.hstack(all_targets)
        all_preds = np.hstack(all_preds)
        all_zs = np.vstack(all_zs)
        all_types = np.hstack(all_types)
        total_r2 = r2_score(all_targets, all_preds)
        self.logger.info("Total R2 score in inferred data: " + str(total_r2))
        pcc, _ = stats.pearsonr(all_targets, all_preds)
        print("PCC in inferred data: " + str(pcc))
        return total_r2, pcc, all_zs, all_types

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
        elif cluster_method == "spectral":
            method_name = "SPECTRAL"
            spec_cluster = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=42)
            clusters = spec_cluster.fit_predict(embeds)
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

    def visualize_emb_um(self, feat_list, cluster_method="kmeans", num_clusters=9):
        n_feats = len(feat_list)
        normal_embeds = self.decode_embedding.weight.data.cpu().detach().numpy()[:n_feats, :]
        tumor_embeds = self.decode_embedding.weight.data.cpu().detach().numpy()[n_feats:, :]
        all_embeds = np.concatenate((normal_embeds, tumor_embeds), axis=0)
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        normal_data_2d = umap_reducer.fit_transform(normal_embeds)
        tumor_data_2d = umap_reducer.fit_transform(tumor_embeds)
        all_data_2d = umap_reducer.fit_transform(all_embeds)
        np.savetxt(f"../output/vae_pair_model/UM_NORMAL.txt", normal_data_2d, fmt='%f', delimiter=' ')
        np.savetxt(f"../output/vae_pair_model/UM_TUMOR.txt", tumor_data_2d, fmt='%f', delimiter=' ')
        method_name = "SPECTRAL"
        plt.figure(figsize=(10, 7))
        normal_class_name = "UM_NORMAL"
        tumor_class_name = "UM_TUMOR"
        tumor_normal_class_name = "UM_TUMOR_NORMAL"
        tumor_normal_cluster_class_name = "UM_TUMOR_NORMAL_CLUSTER"
        plt.title('UMAP Visualization of Gene Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        # plt.legend(title='Cluster')
        # plt.savefig(f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{normal_class_name}_{method_name}")
        # plt.savefig(f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{tumor_class_name}_{method_name}")
        plt.savefig(
            f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{tumor_normal_class_name}_{method_name}_TUMOR")
        # feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        # feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        # feat_cluster_df.to_csv(
        #     f"../output/clusters/vae_pair_gene_clusters_{num_clusters}_{class_name}_{method_name}.csv")
        plt.figure(figsize=(10, 7))
        plt.scatter(normal_data_2d[:, 0], normal_data_2d[:, 1], s=6, c="blue")
        plt.title('UMAP Visualization of Gene Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.savefig(
            f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{normal_class_name}")
        plt.figure(figsize=(10, 7))
        plt.scatter(tumor_data_2d[:, 0], tumor_data_2d[:, 1], s=6, c="red")
        plt.title('UMAP Visualization of Gene Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.savefig(
            f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{tumor_class_name}")
        tumor_clusters_df = pd.read_csv(
            f"../output/clusters/vae_pair_gene_clusters_{num_clusters}_TUMOR_{method_name}_7895_ah.csv")
        normal_clusters_df = pd.read_csv(
            f"../output/clusters/vae_pair_gene_clusters_{num_clusters}_NORMAL_{method_name}_7895_ah.csv")
        tumor_ah_clusters = tumor_clusters_df["Cluster"]
        normal_ah_clusters = normal_clusters_df["Cluster"]
        tumor_colors = ["#FFA500", "#ADD8E6", "#5F9EA0", "#D2691E", "#228B22", "#FFE4E1", "#9370DB", "#6A5ACD"]
        normal_colors = ["#B0CEFF", "#FCBBB6", "#98FB98", "#FF0000", "#6EC55C", "#48D1CC", "#BDB76B", "#FAB2F1"]
        plt.figure(figsize=(10, 7))
        all_normal_data_2d = all_data_2d[:n_feats, :]
        all_tumor_data_2d = all_data_2d[n_feats:, :]
        np.savetxt(f"../output/vae_pair_model/UM_ALL_NORMAL.txt", all_normal_data_2d, fmt='%f', delimiter=' ')
        np.savetxt(f"../output/vae_pair_model/UM_ALL_TUMOR.txt", all_tumor_data_2d, fmt='%f', delimiter=' ')
        for cluster in range(num_clusters):
            cluster_data = all_normal_data_2d[normal_ah_clusters == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=normal_colors[cluster],
                        label=f'Cluster {cluster + 1}', s=6)
        for cluster in range(num_clusters):
            cluster_data = all_tumor_data_2d[tumor_ah_clusters == cluster]
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=tumor_colors[cluster],
                        label=f'Cluster {cluster + 1}', s=6)
        plt.title('UMAP Visualization of Gene Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.savefig(
            f"../output/model_fig/vae_pair_umap_embeddings_{num_clusters}_{tumor_normal_cluster_class_name}_{method_name}")

    def lv_visualize(self, zs, types, type_name_dict):
        umap_reducer = umap.UMAP(n_components=2, random_state=42)
        zs_2d = umap_reducer.fit_transform(zs)
        type_color_dict = {"BRCA_TUMOR":"#FFB6C1","CCRCC_TUMOR":"#DC143C","HNSCC_TUMOR":"#DB7093","HGSC_TUMOR":"#FF69B4",
                           "LUAD_TUMOR":"#FF1493","LSCC_TUMOR":"#C71585","GBM_TUMOR":"#FF00FF","COAD_TUMOR":"#FFFF00",
                           "PDAC_TUMOR":"#CD853F",
                           "UCEC_TUMOR":"#FFA500","CCRCC_NORMAL":"#0000FF","HNSCC_NORMAL":"#00BFFF", "LUAD_NORMAL":"#00CED1",
                           "LSCC_NORMAL":"#2E8B57","PDAC_NORMAL":"#E1FFFF"}
        # draw_list = [0,10] #LSCC
        # draw_list = [1,9]#CCRCC
        # draw_list = [2,5]#HNSCC
        # draw_list = [3,8]#LUAD
        draw_list = [4,14]#PDAC
        # tsne = TSNE(n_components=2, random_state=42)
        # zs_2d = tsne.fit_transform(zs)
        zs_df = pd.DataFrame(data={"X": zs_2d[:, 0], "Y": zs_2d[:, 1], "DataType": types})
        # tumor_idx, normal_idx = get_common_um_idx(types, type_name_dict)
        tumor_idx, normal_idx = get_um_idx(types, type_name_dict)
        unique_types = np.unique(types)
        plt.figure(figsize=(10, 7))
        for i in unique_types:
            if i in draw_list:
                type_idx = zs_df[zs_df['DataType'] == i].index.tolist()
                type_name = type_name_dict[i]
                plt.scatter(zs_2d[type_idx, 0], zs_2d[type_idx, 1], s=40, c=type_color_dict[type_name])
            else:
                continue
        # plt.scatter(zs_2d[tumor_idx, 0], zs_2d[tumor_idx, 1], s=20, c="red")
        # plt.scatter(zs_2d[normal_idx, 0], zs_2d[normal_idx, 1], s=20, c="blue")
        plt.title('UMAP Visualization of Sample Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')

        plt.savefig(
            f"../output/model_fig/vae_pair_umap_embeddings_samples_TUMOR_NORMAL_PDAC.svg", format="svg")

    def distance_analysis(self, feat_list):
        n_feats = len(feat_list)
        normal_embeds = self.decode_embedding.weight.data.cpu().detach().numpy()[:n_feats, :]
        tumor_embeds = self.decode_embedding.weight.data.cpu().detach().numpy()[n_feats:, :]
        embeds_diff = tumor_embeds - normal_embeds
        distance = np.linalg.norm(embeds_diff, axis=1)
        distance_data = {"Gene": feat_list, "Dist": distance}
        distance_df = pd.DataFrame(data=distance_data)
        distance_df.to_csv("../output/vae_pair_model/TUMOR_NORMAL_dist_7895_ah.csv")
        cos_sim = []
        for i in range(n_feats):
            sim = cosine_similarity([normal_embeds[i]], [tumor_embeds[i]])
            cos_sim.append(sim[0, 0])
        cos_data = {"Gene": feat_list, "Cosine": cos_sim}
        cos_df = pd.DataFrame(data=cos_data)
        cos_df.to_csv("../output/vae_pair_model/TUMOR_NORMAL_cos_7895_ah.csv")
        column_names = [f'feature_{i}' for i in range(128)]
        normal_embeds_df = pd.DataFrame(normal_embeds, index=feat_list, columns=column_names)
        tumor_embeds_df = pd.DataFrame(tumor_embeds, index=feat_list, columns=column_names)
        normal_embeds_df.to_csv("../output/vae_pair_model/normal_embeds_7895_ah.csv")
        tumor_embeds_df.to_csv("../output/vae_pair_model/tumor_embeds_7895_ah.csv")
