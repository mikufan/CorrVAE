import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
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
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F
from linformer import Linformer, LinformerSelfAttention
from einops import rearrange


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # print(recon_loss)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(KLD)
    mean_loss = recon_loss + KLD / logvar.numel()
    return mean_loss


class CustomLinformerBlock(nn.Module):
    def __init__(self, dim, seq_len, heads, k):
        super().__init__()
        self.attn = LinformerSelfAttention(dim, seq_len, heads=heads, k=k)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 注意力 + 残差 + Norm
        x = self.attn(x) + x
        x = self.norm1(x)
        # FFN + 残差 + Norm
        x = self.ffn(x) + x
        x = self.norm2(x)
        return x


class MultiOmicsTransformer(nn.Module):
    def __init__(self, input_embs, embedding_dim=64, n_heads=4, n_layers=2, hidden_dim=128, dropout=0.1):
        super(MultiOmicsTransformer, self).__init__()

        self.input_embedding = input_embs

        # 输入嵌入 × 表达量作为输入： shape = (batch_size, num_genes, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True  # Important for using (B, N, D) shape
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 可以加一个全局 pooling 或者分类/回归 head
        self.output_layer = nn.Linear(embedding_dim, 1)  # 举例：回归一个值，可替换

    def forward(self, expression_vector):
        """
        :param expression_vector: shape (batch_size, num_genes)
        """
        batch_size, num_genes = expression_vector.shape

        # 构造 gene ID 索引
        gene_ids = torch.arange(num_genes, device=expression_vector.device).unsqueeze(0).repeat(batch_size, 1)

        # 获取 embedding
        gene_embed = self.input_embedding(gene_ids)  # (batch_size, num_genes, embedding_dim)

        # 加权：embedding × 表达量（逐个基因）
        expr_weighted_embed = gene_embed * expression_vector.unsqueeze(-1)  # Broadcasting

        # Transformer 编码
        encoded = self.transformer_encoder(expr_weighted_embed)  # shape: (batch_size, num_genes, embedding_dim)

        return encoded


class MultiOmicsLinformer(nn.Module):
    def __init__(self, input_embs, embedding_dim=64, k=256, depth=1, n_heads=4, custom=False):
        super(MultiOmicsLinformer, self).__init__()
        self.input_embedding = input_embs
        self.custom = custom
        self.depth = depth
        self.heads = n_heads
        n_feats = input_embs.num_embeddings
        # 输入嵌入 × 表达量作为输入： shape = (batch_size, num_genes, embedding_dim)
        self.attn_map = None  # 用于存储attention map
        if not self.custom:
            self.encoder = Linformer(
                dim=embedding_dim,
                seq_len=n_feats,
                depth=depth,
                heads=n_heads,
                k=k,
            )
            attn = self.encoder.net.layers[-1][0].fn
            print(attn)
            self.encoder_blocks = None
            self._register_attention_hook()
        else:
            self.encoder = None
            self.encoder_blocks = nn.ModuleList([
                CustomLinformerBlock(embedding_dim, n_feats, n_heads, k)
                for _ in range(depth)
            ])

    def _register_attention_hook(self):
        final_attn_module = self.encoder.net.layers[-1][0].fn
        final_attn_module.register_forward_hook(self._attention_hook)

    def _attention_hook(self, module, input, output):
        x = input[0]  # shape: (B, N, D)

        q = module.to_q(x)
        kv = x

        k_proj = torch.einsum('bnd,nk->bkd', kv, module.proj_k)
        k = module.to_k(k_proj)

        q = rearrange(q, 'b n (h d) -> b h n d', h=module.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=module.heads)

        head_dim = q.shape[-1]
        scale = head_dim ** -0.5
        dots = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = dots.softmax(dim=-1)

        self.attn_map = attn.detach().cpu()

    def forward(self, expression_vector):
        """
        :param expression_vector: shape (batch_size, num_genes)
        """
        batch_size, num_genes = expression_vector.shape

        # 构造 gene ID 索引
        gene_ids = torch.arange(num_genes, device=expression_vector.device).unsqueeze(0).repeat(batch_size, 1)

        # 获取 embedding
        gene_embed = self.input_embedding(gene_ids)  # (batch_size, num_genes, embedding_dim)

        # 加权：embedding × 表达量（逐个基因）
        expr_weighted_embed = gene_embed * expression_vector.unsqueeze(-1)  # Broadcasting

        # Transformer 编码
        if not self.custom:
            encoded = self.encoder(expr_weighted_embed)  # shape: (batch_size, num_genes, embedding_dim)
        else:
            encoded = expr_weighted_embed
            for i, block in enumerate(self.encoder_blocks):
                encoded = block(encoded)
                # 如果是最后一层，手动重建 attention map
                if i == len(self.encoder_blocks) - 1:
                    # 重新计算 QK^T 注意力
                    q = block.attn.to_q(encoded)
                    k = block.attn.to_k(encoded)

                    q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
                    k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)

                    dots = torch.matmul(q, k.transpose(-1, -2)) * block.attn.scale  # (B, H, N, k)
                    attn = dots.softmax(dim=-1)

                    self.attn_map = attn.detach().cpu()  # 保存最终注意力图

        return encoded


def contrastive_loss(z, labels, temperature=0.5):
    """
    Compute contrastive loss on latent z using supervised NT-Xent loss.

    z: tensor of shape [batch_size, latent_dim]
    labels: tensor of shape [batch_size] with int type labels
    """
    z = F.normalize(z, dim=1)  # cosine similarity

    similarity_matrix = torch.matmul(z, z.T)  # [B, B]
    sim_exp = torch.exp(similarity_matrix / temperature)

    # Mask out diagonal (self-similarity)
    batch_size = z.shape[0]
    mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
    sim_exp = sim_exp.masked_fill(mask, 0)

    # Build positive mask: [B, B] with True at positions where labels match and i ≠ j
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    label_eq = label_eq & (~mask)

    # For each sample, compute numerator and denominator of contrastive loss
    numerator = (sim_exp * label_eq).sum(dim=1)
    denominator = sim_exp.sum(dim=1)

    # Add small epsilon to avoid log(0)
    loss = -torch.log((numerator + 1e-8) / (denominator + 1e-8)).mean()
    return loss


class CorrVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, embedding_dim, n_feats, type_embedding_dim, n_types,
                 n_phs, n_acs, device, no_types, trans_encoder, lin_encoder, con_loss, alpha, cus_line):
        super(CorrVAE, self).__init__()

        self.device = device
        self.no_types = no_types
        self.trans_encoder = trans_encoder
        self.lin_encoder = lin_encoder
        self.con_loss = con_loss
        self.alpha = alpha
        self.cus_line = cus_line
        # Encoder
        self.feat_fc = nn.Linear(n_feats, hidden_dim)
        self.fc1 = nn.Linear(n_feats * embedding_dim, hidden_dim)
        # self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        if not self.no_types:
            self.fc2_mu = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        else:
            self.fc2_mu = nn.Linear(hidden_dim, latent_dim)
        if not self.no_types:
            self.fc2_logvar = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        else:
            self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)

        if n_phs == 0 and n_acs == 0:
            mid_dim = hidden_dim
        elif (n_phs != 0 and n_acs == 0) or (n_phs == 0 and n_acs != 0):
            mid_dim = 2 * hidden_dim
        else:
            mid_dim = 3 * hidden_dim
        self.merge_hidden = nn.Sequential(nn.Linear(mid_dim, hidden_dim),
                                          nn.ReLU())
        self.mlp_ph_encoder = nn.Sequential(
            nn.Linear(n_phs, hidden_dim),
            nn.ReLU()
        )
        self.mlp_ac_encoder = nn.Sequential(
            nn.Linear(n_acs, hidden_dim),
            nn.ReLU()
        )

        # Embedding
        if self.trans_encoder or self.lin_encoder:
            if n_phs == 0 and n_acs == 0:
                self.gene_embedding = nn.Embedding(n_feats, embedding_dim)
            elif n_phs > 0 and n_acs == 0:
                self.gene_embedding = nn.Embedding(n_feats + n_phs, embedding_dim)
            elif n_phs > 0 and n_acs > 0:
                self.gene_embedding = nn.Embedding(n_feats + n_phs + n_acs, embedding_dim)
            elif n_phs == 0 and n_acs > 0:
                self.gene_embedding = nn.Embedding(n_feats + n_acs, embedding_dim)
            else:
                self.gene_embedding = None
        else:
            self.gene_embedding = None
        if self.trans_encoder:
            self.mo_trans_encoder = MultiOmicsTransformer(self.gene_embedding, embedding_dim)
        else:
            self.mo_trans_encoder = None
        if self.lin_encoder:
            self.mo_lin_encoder = MultiOmicsLinformer(self.gene_embedding, embedding_dim, custom=self.cus_line)
        else:
            self.mo_lin_encoder = None

        self.type_embedding = nn.Embedding(n_types, type_embedding_dim)
        # Decoder
        if not self.no_types:
            self.fc3 = nn.Linear(latent_dim + type_embedding_dim, embedding_dim)
        else:
            self.fc3 = nn.Linear(latent_dim, embedding_dim)
        self.fc4 = nn.Linear(hidden_dim, n_feats)
        self.nc_fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.decode_embedding = nn.Linear(embedding_dim, n_feats)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.logger = utils.get_logger('../output/log/train_log.log')

        self.logger = None

    def encode(self, x, data_type, ph_x=None, ac_x=None, no_type=False):
        h1 = torch.relu(self.feat_fc(x))
        if ph_x is not None and ac_x is None:
            ph_x = torch.unsqueeze(ph_x, 1)
            ph_h = self.mlp_ph_encoder(ph_x)
            ph_h = torch.squeeze(ph_h, 1)
            h1 = torch.cat((h1, ph_h), 1)
            h1 = self.merge_hidden(h1)
            h1 = torch.relu(h1)
        if ph_x is None and ac_x is not None:
            ac_x = torch.unsqueeze(ac_x, 1)
            ac_h = self.mlp_ac_encoder(ac_x)
            ac_h = torch.squeeze(ac_h, 1)
            h1 = torch.cat((h1, ac_h), 1)
            h1 = self.merge_hidden(h1)
            h1 = torch.relu(h1)
        if ph_x is not None and ac_x is not None:
            ph_x = torch.unsqueeze(ph_x, 1)
            ph_h = self.mlp_ph_encoder(ph_x)
            ph_h = torch.squeeze(ph_h, 1)
            h1 = torch.cat((h1, ph_h), 1)
            ac_x = torch.unsqueeze(ac_x, 1)
            ac_h = self.mlp_ac_encoder(ac_x)
            ac_h = torch.squeeze(ac_h, 1)
            h1 = torch.cat((h1, ac_h), 1)
            h1 = self.merge_hidden(h1)
            h1 = torch.relu(h1)
        if no_type is False:
            emb_type = self.type_embedding(data_type)
            h1 = torch.cat((h1, emb_type), 1)
            h1 = torch.relu(h1)
            return self.fc2_mu(h1), self.fc2_logvar(h1), emb_type
        else:
            emb_type = None
            mu_out = self.fc2_mu(h1)
            var_out = self.fc2_logvar(h1)
            return mu_out, var_out, emb_type

    def transformer_encode(self, x, data_type, ph_x=None, ac_x=None, no_type=False):
        input_x = x
        if ph_x is not None and ac_x is None:
            input_x = torch.cat((x, ph_x), 1)
        if ph_x is None and ac_x is not None:
            input_x = torch.cat((x, ac_x), 1)
        if ph_x is not None and ac_x is not None:
            input_x = torch.cat((x, ph_x, ac_x), 1)
        h1 = self.mo_trans_encoder(input_x)
        h1 = h1.mean(dim=1)  # (B, D)
        if no_type is False:
            emb_type = self.type_embedding(data_type)
            h1 = torch.cat((h1, emb_type), 1)
            return self.fc2_mu(h1), self.fc2_logvar(h1), emb_type
        else:
            emb_type = None
            mu_out = self.fc2_mu(h1)
            var_out = self.fc2_logvar(h1)
            return mu_out, var_out, emb_type

    def linformer_encode(self, x, data_type, ph_x=None, ac_x=None, no_type=False):
        input_x = x
        if ph_x is not None and ac_x is None:
            input_x = torch.cat((x, ph_x), 1)
        if ph_x is None and ac_x is not None:
            input_x = torch.cat((x, ac_x), 1)
        if ph_x is not None and ac_x is not None:
            input_x = torch.cat((x, ph_x, ac_x), 1)
        h1 = self.mo_lin_encoder(input_x)
        h1 = h1.mean(dim=1)  # (B, D)
        if no_type is False:
            emb_type = self.type_embedding(data_type)
            h1 = torch.cat((h1, emb_type), 1)
            return self.fc2_mu(h1), self.fc2_logvar(h1), emb_type
        else:
            emb_type = None
            mu_out = self.fc2_mu(h1)
            var_out = self.fc2_logvar(h1)
            return mu_out, var_out, emb_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps_numpy = eps.cpu().detach().numpy()
        return mu + eps * std

    def decode(self, z, y, emb_types, no_type=False):
        if not no_type:
            h3 = torch.cat((z, emb_types), 1)
            h3 = torch.relu(self.fc3(h3))
        else:
            h3 = z
            h3 = torch.relu(self.fc3(h3))
        recon_y = self.decode_embedding(h3)

        return recon_y, y

    def z_forward(self, x, data_type, ph_x=None, ac_x=None):
        x = x.to(self.device)
        data_type = data_type.to(self.device)
        ph_x = ph_x.to(self.device)
        ac_x = ac_x.to(self.device)
        mu, logvar, emb_type = self.encode(x, data_type, ph_x=ph_x, ac_x=ac_x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        return z

    def forward(self, x, y, data_type, is_train=True, ph_x=None, ac_x=None, no_type=False):
        x = x.to(self.device)
        y = y.to(self.device)
        data_type = data_type.to(self.device)
        if ph_x is not None:
            ph_x = ph_x.to(self.device)
        if ac_x is not None:
            ac_x = ac_x.to(self.device)
        if not self.trans_encoder:
            if self.lin_encoder:
                mu, logvar, emb_type = self.linformer_encode(x, data_type, ph_x=ph_x, ac_x=ac_x, no_type=no_type)
            else:
                mu, logvar, emb_type = self.encode(x, data_type, ph_x=ph_x, ac_x=ac_x, no_type=no_type)
        else:
            mu, logvar, emb_type = self.transformer_encode(x, data_type, ph_x=ph_x, ac_x=ac_x, no_type=no_type)
        logvar = torch.clamp(logvar, min=-10, max=10)
        z = self.reparameterize(mu, logvar)
        recon_y, y = self.decode(z, y, emb_type, no_type)
        if is_train:
            vae_loss = vae_loss_function(recon_y, y, mu, logvar)
            if not self.con_loss:
                return vae_loss
            else:
                type_con_loss = contrastive_loss(z, data_type)
                all_loss = vae_loss + self.alpha * type_con_loss
                return all_loss
        else:
            return recon_y, y

    def model_training(self, train_data, n_epochs, valid_data, model_path):
        self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger.info("Model constructed.")
        self.logger.info("Start training ...")
        best_result = 0.0
        # n_feats = self.gene_embedding.weight.data.shape[0]
        n_feats = self.decode_embedding.weight.data.shape[0]
        train_ph_df = train_data.dataset.dataset.ph_df \
            if isinstance(train_data.dataset, Subset) else train_data.dataset.ph_df
        train_ac_df = train_data.dataset.dataset.ac_df \
            if isinstance(train_data.dataset, Subset) else train_data.dataset.ac_df
        valid_ph_df = valid_data.dataset.dataset.ph_df \
            if isinstance(valid_data.dataset, Subset) else valid_data.dataset.ph_df
        valid_ac_df = valid_data.dataset.dataset.ac_df \
            if isinstance(valid_data.dataset, Subset) else valid_data.dataset.ac_df
        type_name_dict = train_data.dataset.dataset.type_name_dict \
            if isinstance(train_data.dataset, Subset) else train_data.dataset.type_name_dict
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
                batch_data_type = torch.tensor(batch_data_type)
                # ptm input
                batch_ph_input = [d.ph_inputs for d in data_batch] \
                    if train_ph_df is not None else None
                batch_ac_input = [d.ac_inputs for d in data_batch] \
                    if train_ac_df is not None else None
                if batch_ph_input is not None:
                    batch_ph_input = torch.tensor(batch_ph_input, dtype=torch.float32)
                if batch_ac_input is not None:
                    batch_ac_input = torch.tensor(batch_ac_input, dtype=torch.float32)

                self.optimizer.zero_grad()
                batch_loss = self.forward(batch_input, batch_target, batch_data_type, ph_x=batch_ph_input,
                                          ac_x=batch_ac_input, no_type=self.no_types)
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

                # ptm input
                valid_batch_ph_input = [d.ph_inputs for d in valid_data_batch] \
                    if valid_ph_df is not None else None
                valid_batch_ac_input = [d.ac_inputs for d in valid_data_batch] \
                    if valid_ac_df is not None else None
                if valid_batch_ph_input is not None:
                    valid_batch_ph_input = torch.tensor(valid_batch_ph_input, dtype=torch.float32)
                if valid_batch_ac_input is not None:
                    valid_batch_ac_input = torch.tensor(valid_batch_ac_input, dtype=torch.float32)

                preds, targets = self.forward(valid_batch_input, valid_batch_target, valid_batch_data_type,
                                              is_train=False, ph_x=valid_batch_ph_input, ac_x=valid_batch_ac_input,
                                              no_type=self.no_types)
                preds = preds.cpu().detach().numpy()
                targets = targets.cpu().detach().numpy()
                for i in range(batch_size):
                    all_targets.append(targets[i])
                    all_preds.append(preds[i])
            all_targets = np.hstack(all_targets)
            all_preds = np.hstack(all_preds)

            total_r2 = r2_score(all_targets, all_preds)
            pcc = pearsonr(all_targets, all_preds)
            self.logger.info("Total validation R2 score in epoch " + str(n) + ": " + str(total_r2))
            self.logger.info("Total validation PCC in epoch " + str(n) + ": " + str(pcc))
            if total_r2 > best_result:
                best_result = total_r2
                if not self.no_types:
                    torch.save(self, model_path + "/trained_corr_vae_model.pt")
                else:
                    torch.save(self, model_path + "/trained_corr_vae_model_no_type.pt")

    def model_test(self, data_loader, result_path=None):
        n_batch = len(data_loader)
        data_type_dict = data_loader.dataset.dataset.data_type_dict
        id_2_type_dict = {}
        for key, value in data_type_dict.items():
            id_2_type_dict[value] = key
        self.eval()
        all_inputs = []
        all_preds = []
        all_targets = []
        all_types = []
        all_data_ids = []
        # test_ph_df = data_loader.dataset.dataset.ph_df \
        #     if isinstance(data_loader.dataset, Subset) else data_loader.dataset.ph_df
        # test_ac_df = data_loader.dataset.dataset.ac_df \
        #     if isinstance(data_loader.dataset, Subset) else data_loader.dataset.ac_df
        n_feats = self.decode_embedding.weight.data.shape[0]
        for test_data_batch in tqdm(data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                    leave=True, file=sys.stdout):
            batch_size = len(test_data_batch)
            batch_input = [d.input_feat for d in test_data_batch]
            batch_target = [d.target_feat for d in test_data_batch]
            batch_data_type = [d.data_type_id for d in test_data_batch]
            batch_data_id = [d.sample_id for d in test_data_batch]
            batch_input = torch.tensor(batch_input, dtype=torch.float32)
            batch_target = torch.tensor(batch_target, dtype=torch.float32)
            batch_data_type = torch.tensor(batch_data_type)
            # ptm input
            batch_ph_input = [d.ph_inputs for d in test_data_batch] \
                if data_loader.dataset.dataset.ph_df is not None else None
            batch_ac_input = [d.ac_inputs for d in test_data_batch] \
                if data_loader.dataset.dataset.ac_df is not None else None
            if batch_ph_input is not None:
                batch_ph_input = torch.tensor(batch_ph_input, dtype=torch.float32)
            if batch_ac_input is not None:
                batch_ac_input = torch.tensor(batch_ac_input, dtype=torch.float32)

            preds, targets = self.forward(batch_input, batch_target, batch_data_type, is_train=False,
                                          ph_x=batch_ph_input, ac_x=batch_ac_input, no_type=self.no_types)
            preds = preds.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            for i in range(batch_size):
                all_targets.append(targets[i])
                all_preds.append(preds[i])
                all_types.append(batch_data_type[i])
                all_inputs.append(batch_input[i].cpu().detach().numpy())
                all_data_ids.append(batch_data_id[i])

        all_targets = np.hstack(all_targets)
        all_preds = np.hstack(all_preds)
        all_inputs = np.hstack(all_inputs)
        all_data_ids = np.hstack(all_data_ids)

        all_targets_array = all_targets.reshape(-1, n_feats)
        all_preds_array = all_preds.reshape(-1, n_feats)
        all_inputs_array = all_inputs.reshape(-1, n_feats)

        res_array = np.column_stack((all_targets, all_preds))
        res_df = pd.DataFrame(res_array, columns=['target', 'prediction'])
        res_df.to_csv(result_path + ".csv")
        all_diffs = (all_targets - all_preds) ** 2
        all_diffs = all_diffs.reshape(-1, n_feats)
        all_types = np.hstack(all_types)
        df = pd.DataFrame(all_diffs)
        all_type_names = [id_2_type_dict[type_id] for type_id in all_types]
        df['label'] = all_type_names
        grouped_means = df.groupby('label').mean()
        grouped_means_df = pd.DataFrame(grouped_means)
        grouped_means_df.columns = data_loader.dataset.dataset.feat_list
        grouped_means_df.to_csv(result_path + "_diff.csv")
        total_r2 = r2_score(all_targets, all_preds)
        pcc = pearsonr(all_targets, all_preds)
        scc = spearmanr(all_targets, all_preds)
        # Result dataframes
        all_targets_df = pd.DataFrame(all_targets_array)
        all_preds_df = pd.DataFrame(all_preds_array)
        all_inputs_df = pd.DataFrame(all_inputs_array)
        all_targets_df.columns = data_loader.dataset.dataset.feat_list
        all_preds_df.columns = data_loader.dataset.dataset.feat_list
        all_inputs_df.columns = data_loader.dataset.dataset.feat_list
        # all_targets_df.index = all_type_names
        # all_preds_df.index = all_type_names
        # all_inputs_df.index = all_type_names
        all_targets_df.index = all_data_ids
        all_preds_df.index = all_data_ids
        all_inputs_df.index = all_data_ids
        all_targets_df["TYPE"] = all_type_names
        all_inputs_df["TYPE"] = all_type_names
        all_preds_df["TYPE"] = all_type_names
        if not self.no_types:
            all_targets_df.to_csv(result_path + "_targets.csv")
            all_preds_df.to_csv(result_path + "_preds.csv")
            all_inputs_df.to_csv(result_path + "_inputs.csv")
        else:
            all_targets_df.to_csv(result_path + "_targets_no_type.csv")
            all_preds_df.to_csv(result_path + "_preds_no_type.csv")
            all_inputs_df.to_csv(result_path + "_inputs_no_type.csv")

        return total_r2, pcc, scc

    def model_ood_test(self, data_loader, result_path=None):
        n_batch = len(data_loader)
        self.eval()
        all_inputs = []
        all_preds = []
        all_targets = []
        all_types = []
        all_data_ids = []
        n_feats = self.decode_embedding.weight.data.shape[0]
        data_ph_df = data_loader.dataset.dataset.ph_df \
            if isinstance(data_loader.dataset, Subset) else data_loader.dataset.ph_df
        data_ac_df = data_loader.dataset.dataset.ac_df \
            if isinstance(data_loader.dataset, Subset) else data_loader.dataset.ac_df

        for test_data_batch in tqdm(data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                    leave=True, file=sys.stdout):
            batch_size = len(test_data_batch)
            batch_input = [d.input_feat for d in test_data_batch]
            batch_target = [d.target_feat for d in test_data_batch]
            batch_data_type = [d.data_type_id for d in test_data_batch]
            batch_data_id = [d.sample_id for d in test_data_batch]
            batch_input = torch.tensor(batch_input, dtype=torch.float32)
            batch_target = torch.tensor(batch_target, dtype=torch.float32)
            batch_data_type = torch.tensor(batch_data_type)
            # ptm input
            batch_ph_input = [d.ph_inputs for d in test_data_batch] \
                if data_ph_df is not None else None
            print(len(batch_ph_input[0]))
            batch_ac_input = [d.ac_inputs for d in test_data_batch] \
                if data_ac_df is not None else None
            if batch_ph_input is not None:
                batch_ph_input = torch.tensor(batch_ph_input, dtype=torch.float32)
            if batch_ac_input is not None:
                batch_ac_input = torch.tensor(batch_ac_input, dtype=torch.float32)

            preds, targets = self.forward(batch_input, batch_target, batch_data_type, is_train=False,
                                          ph_x=batch_ph_input, ac_x=batch_ac_input, no_type=True)
            preds = preds.cpu().detach().numpy()
            targets = targets.cpu().detach().numpy()
            for i in range(batch_size):
                all_targets.append(targets[i])
                all_preds.append(preds[i])
                all_types.append(batch_data_type[i])
                all_inputs.append(batch_input[i].cpu().detach().numpy())
                all_data_ids.append(batch_data_id[i])

        all_targets = np.hstack(all_targets)
        all_preds = np.hstack(all_preds)
        all_inputs = np.hstack(all_inputs)
        all_data_ids = np.hstack(all_data_ids)

        all_targets_array = all_targets.reshape(-1, n_feats)
        all_preds_array = all_preds.reshape(-1, n_feats)
        all_inputs_array = all_inputs.reshape(-1, n_feats)

        res_array = np.column_stack((all_targets, all_preds))
        res_df = pd.DataFrame(res_array, columns=['target', 'prediction'])
        res_df.to_csv(result_path + ".csv")
        all_diffs = (all_targets - all_preds) ** 2
        all_diffs = all_diffs.reshape(-1, n_feats)
        all_types = np.hstack(all_types)
        df = pd.DataFrame(all_diffs)
        all_type_names = ["UNKOWN" for i in range(len(all_types))]
        df['label'] = all_type_names
        grouped_means = df.groupby('label').mean()
        grouped_means_df = pd.DataFrame(grouped_means)
        grouped_means_df.columns = data_loader.dataset.feat_list
        grouped_means_df.to_csv(result_path + "_diff.csv")
        mask = all_targets != 1e-6
        targets_filtered = all_targets[mask]
        preds_filtered = all_preds[mask]
        total_r2 = r2_score(targets_filtered, preds_filtered)
        pcc = pearsonr(targets_filtered, preds_filtered)

        # Result dataframes
        all_targets_df = pd.DataFrame(all_targets_array)
        all_preds_df = pd.DataFrame(all_preds_array)
        all_inputs_df = pd.DataFrame(all_inputs_array)
        all_targets_df.columns = data_loader.dataset.feat_list
        all_preds_df.columns = data_loader.dataset.feat_list
        all_inputs_df.columns = data_loader.dataset.feat_list
        # all_targets_df.index = all_type_names
        # all_preds_df.index = all_type_names
        # all_inputs_df.index = all_type_names
        all_targets_df.index = all_data_ids
        all_preds_df.index = all_data_ids
        all_inputs_df.index = all_data_ids
        all_targets_df["TYPE"] = all_type_names
        all_inputs_df["TYPE"] = all_type_names
        all_preds_df["TYPE"] = all_type_names
        all_targets_df.to_csv(result_path + "_targets.csv")
        all_preds_df.to_csv(result_path + "_preds.csv")
        all_inputs_df.to_csv(result_path + "_inputs.csv")

        return total_r2, pcc

    def visualize_emb(self, feat_list, model_class, method="tsne", cluster_method="kmeans", num_clusters=9):
        if model_class == 1:
            class_name = "TUMOR"
        else:
            class_name = "NORMAL"
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
                    # plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster + 1}', s=6)
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
            plt.savefig(
                f"../output/model_fig/corr_model_vae_umap_embeddings_{num_clusters}_{class_name}_{method_name}.svg",
                format="svg")
        feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        feat_cluster_df.to_csv(
            f"../output/clusters/corr_model_vae_gene_clusters_{num_clusters}_{class_name}_{method_name}.csv")

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
            plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {i + 1}', s=4)

        plt.title('UMAP Visualization of Gene Embeddings')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend(title='Cluster')
        plt.savefig(f"../output/model_fig/vae_umap_spec_embeddings_{num_clusters}_{class_name}.svg", format="svg")

        feat_cluster_dict = {"Gene Name": feat_list, "Cluster": clusters}
        feat_cluster_df = pd.DataFrame.from_dict(feat_cluster_dict)
        feat_cluster_df.to_csv(f"../output/clusters/vae_gene_spec_clusters_{num_clusters}_{class_name}.csv")


class PtmCorrVAE(nn.Module):
    def __init__(self, hidden_dim, latent_dim, embedding_dim, n_feats, type_embedding_dim, n_types,
                 n_phs, n_acs, device):
        super(PtmCorrVAE, self).__init__()

        self.device = device
        # Encoder
        self.feat_fc = nn.Linear(n_feats, hidden_dim)
        self.fc1 = nn.Linear(n_feats * embedding_dim, hidden_dim)
        # self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim + type_embedding_dim, latent_dim)
        mid_dim = 2 * hidden_dim
        self.merge_hidden = nn.Sequential(nn.Linear(mid_dim, hidden_dim),
                                          nn.ReLU())
        self.mlp_protein_encoder = nn.Sequential(
            nn.Linear(n_feats, hidden_dim),
            nn.ReLU()
        )

        # Embedding
        self.gene_embedding = nn.Embedding(n_feats, embedding_dim)
        self.type_embedding = nn.Embedding(n_types, type_embedding_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim + type_embedding_dim, embedding_dim)
        self.fc4 = nn.Linear(hidden_dim, n_feats)
        self.nc_fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.ph_decode_embedding = nn.Linear(embedding_dim, n_phs)
        self.ac_decode_embedding = nn.Linear(embedding_dim, n_acs)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.logger = None

    def forward(self, x, px, data_type, is_train, ph_y, ac_y):
        x = x.to(self.device)
        px = px.to(self.device)
        data_type = data_type.to(self.device)
        ph_y = ph_y.to(self.device)
        ac_y = ac_y.to(self.device)
        mu, logvar, emb_type = self.ptm_encode(x, px, data_type)
        logvar = torch.clamp(logvar, min=-10, max=10)
        # log_var_numpy = logvar.cpu().detach().numpy()
        # mu_var_numpy = mu.cpu().detach().numpy()
        z = self.reparameterize(mu, logvar)
        recon_ph_y, ph_y, recon_ac_y, ac_y = self.ptm_decode(z, ph_y, ac_y, emb_type)
        if is_train:
            ph_vae_loss = vae_loss_function(recon_ph_y, ph_y, mu, logvar)
            ac_vae_loss = vae_loss_function(recon_ac_y, ac_y, mu, logvar)
            vae_loss = ph_vae_loss + ac_vae_loss
            return vae_loss
        else:
            return recon_ph_y, ph_y, recon_ac_y, ac_y

    def ptm_encode(self, x, px, data_type):
        h1 = torch.relu(self.feat_fc(x))
        p_h = self.mlp_protein_encoder(px)
        h1 = torch.cat((h1, p_h), 1)
        h1 = self.merge_hidden(h1)
        h1 = torch.relu(h1)
        emb_type = self.type_embedding(data_type)
        h1 = torch.cat((h1, emb_type), 1)
        return self.fc2_mu(h1), self.fc2_logvar(h1), emb_type

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        eps_numpy = eps.cpu().detach().numpy()
        return mu + eps * std

    def ptm_decode(self, z, ph_y, ac_y, emb_types):

        h3 = torch.cat((z, emb_types), 1)
        h3 = torch.relu(self.fc3(h3))
        recon_ph_y = self.ph_decode_embedding(h3)
        recon_ac_y = self.ac_decode_embedding(h3)
        return recon_ph_y, ph_y, recon_ac_y, ac_y

    # def z_forward(self, x, data_type, ph_x=None, ac_x=None):
    #     x = x.to(self.device)
    #     data_type = data_type.to(self.device)
    #     ph_x = ph_x.to(self.device)
    #     ac_x = ac_x.to(self.device)
    #     mu, logvar, emb_type = self.encode(x, data_type, ph_x=ph_x, ac_x=ac_x)
    #     logvar = torch.clamp(logvar, min=-10, max=10)
    #     z = self.reparameterize(mu, logvar)
    #     return z

    def model_training(self, train_data, n_epochs, valid_data, model_path):
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
                batch_protein_input = [d.target_feat for d in data_batch]
                batch_data_type = [d.data_type_id for d in data_batch]
                batch_input = torch.tensor(batch_input, dtype=torch.float32)
                batch_protein_input = torch.tensor(batch_protein_input, dtype=torch.float32)
                batch_data_type = torch.tensor(batch_data_type)
                # ptm input
                batch_ph_target = [d.ph_inputs for d in data_batch]
                batch_ac_target = [d.ac_inputs for d in data_batch]
                batch_ph_target = torch.tensor(batch_ph_target, dtype=torch.float32)
                batch_ac_target = torch.tensor(batch_ac_target, dtype=torch.float32)
                self.optimizer.zero_grad()
                batch_loss = self.forward(batch_input, batch_protein_input, batch_data_type, True,
                                          batch_ph_target, batch_ac_target)
                batch_loss.backward()
                self.optimizer.step()
                total_loss += batch_loss
            self.logger.info("Loss in epoch " + str(n) + ": " + str(total_loss.cpu().detach().numpy() / n_batch))
            n_valid_batch = len(valid_data)
            self.eval()
            all_ph_targets = []
            all_ph_preds = []
            all_ac_targets = []
            all_ac_preds = []
            for valid_data_batch in tqdm(valid_data, mininterval=2, desc=' -Tot it %d' % n_valid_batch,
                                         leave=True, file=sys.stdout):
                batch_size = len(valid_data_batch)
                valid_batch_input = [d.input_feat for d in valid_data_batch]
                valid_batch_protein_input = [d.target_feat for d in valid_data_batch]
                valid_batch_data_type = [d.data_type_id for d in valid_data_batch]
                valid_batch_input = torch.tensor(valid_batch_input, dtype=torch.float32)
                valid_batch_protein_input = torch.tensor(valid_batch_protein_input, dtype=torch.float32)
                valid_batch_data_type = torch.tensor(valid_batch_data_type)

                # ptm target
                valid_batch_ph_target = [d.ph_inputs for d in valid_data_batch]
                valid_batch_ac_target = [d.ac_inputs for d in valid_data_batch]
                valid_batch_ph_target = torch.tensor(valid_batch_ph_target, dtype=torch.float32)
                valid_batch_ac_input = torch.tensor(valid_batch_ac_target, dtype=torch.float32)

                ph_preds, ph_targets, ac_preds, ac_targets = self.forward(valid_batch_input, valid_batch_protein_input,
                                                                          valid_batch_data_type, False,
                                                                          valid_batch_ph_target,
                                                                          valid_batch_ac_input)
                ph_preds = ph_preds.cpu().detach().numpy()
                ph_targets = ph_targets.cpu().detach().numpy()
                ac_preds = ac_preds.cpu().detach().numpy()
                ac_targets = ac_targets.cpu().detach().numpy()
                for i in range(batch_size):
                    all_ph_targets.append(ph_targets[i])
                    all_ph_preds.append(ph_preds[i])
                    all_ac_targets.append(ac_targets[i])
                    all_ac_preds.append(ac_preds[i])
            all_ph_targets = np.hstack(all_ph_targets)
            all_ph_preds = np.hstack(all_ph_preds)
            all_ac_targets = np.hstack(all_ac_targets)
            all_ac_preds = np.hstack(all_ac_preds)
            total_ph_r2 = r2_score(all_ph_targets, all_ph_preds)
            total_ac_r2 = r2_score(all_ac_targets, all_ac_preds)
            ph_pcc = pearsonr(all_ph_targets, all_ph_preds)
            ac_pcc = pearsonr(all_ac_targets, all_ac_preds)

            self.logger.info("Total PH validation R2 score in epoch " + str(n) + ": " + str(total_ph_r2))
            self.logger.info("Total PH validation PCC in epoch " + str(n) + ": " + str(ph_pcc))
            self.logger.info("Total AC validation R2 score in epoch " + str(n) + ": " + str(total_ac_r2))
            self.logger.info("Total AC validation PCC in epoch " + str(n) + ": " + str(ac_pcc))
            if total_ph_r2 > best_result:
                best_result = total_ph_r2
                torch.save(self, model_path + "/trained_corr_ptm_vae_model.pt")

    def model_test(self, data_loader, result_path=None):
        n_batch = len(data_loader)
        data_type_dict = data_loader.dataset.dataset.data_type_dict
        id_2_type_dict = {}
        for key, value in data_type_dict.items():
            id_2_type_dict[value] = key
        self.eval()
        all_inputs = []
        all_ph_preds = []
        all_ph_targets = []
        all_ac_preds = []
        all_ac_targets = []
        all_protein_inputs = []
        all_types = []
        all_data_ids = []
        n_feats = self.gene_embedding.weight.data.shape[0]
        n_phs = self.ph_decode_embedding.weight.data.shape[0]
        n_acs = self.ac_decode_embedding.weight.data.shape[0]
        for test_data_batch in tqdm(data_loader, mininterval=2, desc=' -Tot it %d' % n_batch,
                                    leave=True, file=sys.stdout):
            batch_size = len(test_data_batch)
            batch_input = [d.input_feat for d in test_data_batch]
            batch_protein_input = [d.target_feat for d in test_data_batch]
            batch_data_type = [d.data_type_id for d in test_data_batch]
            batch_data_id = [d.sample_id for d in test_data_batch]
            batch_input = torch.tensor(batch_input, dtype=torch.float32)
            batch_protein_input = torch.tensor(batch_protein_input, dtype=torch.float32)
            batch_data_type = torch.tensor(batch_data_type)
            # ptm target
            batch_ph_target = [d.ph_inputs for d in test_data_batch]
            batch_ac_target = [d.ac_inputs for d in test_data_batch] \
                if data_loader.dataset.dataset.ac_df is not None else None
            batch_ph_target = torch.tensor(batch_ph_target, dtype=torch.float32)
            batch_ac_target = torch.tensor(batch_ac_target, dtype=torch.float32)

            ph_preds, ph_targets, ac_preds, ac_targets = self.forward(batch_input, batch_protein_input, batch_data_type,
                                                                      False, batch_ph_target, batch_ac_target)
            ph_preds = ph_preds.cpu().detach().numpy()
            ph_targets = ph_targets.cpu().detach().numpy()
            ac_preds = ac_preds.cpu().detach()
            ac_targets = ac_targets.cpu().detach()
            for i in range(batch_size):
                all_ph_targets.append(ph_targets[i])
                all_ph_preds.append(ph_preds[i])
                all_ac_targets.append(ac_targets[i])
                all_ac_preds.append(ac_preds[i])
                all_types.append(batch_data_type[i])
                all_inputs.append(batch_input[i].cpu().detach().numpy())
                all_protein_inputs.append(batch_protein_input[i].cpu().detach().numpy())
                all_data_ids.append(batch_data_id[i])

        all_ph_targets = np.hstack(all_ph_targets)
        all_ph_preds = np.hstack(all_ph_preds)
        all_ac_targets = np.hstack(all_ac_targets)
        all_ac_preds = np.hstack(all_ac_preds)
        # all_inputs = np.hstack(all_inputs)
        all_data_ids = np.hstack(all_data_ids)
        all_types = np.hstack(all_types)
        # all_ph_targets_array = all_ph_targets.reshape(-1, n_feats)
        # all_ph_preds_array = all_ph_preds.reshape(-1, n_feats)
        # all_ac_targets_array = all_ac_targets.reshape(-1, n_feats)
        # all_ph_preds_array = all_ph_preds.reshape(-1, n_feats)
        # # all_inputs_array = all_inputs.reshape(-1, n_feats)
        all_type_names = [id_2_type_dict[type_id] for type_id in all_types]

        ph_res_array = np.column_stack((all_ph_targets, all_ph_preds))
        ph_res_df = pd.DataFrame(ph_res_array, columns=['target', 'prediction'])
        ph_res_df.to_csv(result_path + "_ph.csv")
        ph_all_diffs = (all_ph_targets - all_ph_preds) ** 2
        ph_all_diffs = ph_all_diffs.reshape(-1, n_phs)

        ph_df = pd.DataFrame(ph_all_diffs)

        ph_df['label'] = all_type_names
        ph_grouped_means = ph_df.groupby('label').mean()
        ph_grouped_means_df = pd.DataFrame(ph_grouped_means)
        ph_grouped_means_df.columns = data_loader.dataset.dataset.ph_feat_list
        ph_grouped_means_df.to_csv(result_path + "_ph_diff.csv")
        ph_total_r2 = r2_score(all_ph_targets, all_ph_preds)
        ph_pcc = pearsonr(all_ph_targets, all_ph_preds)

        ac_res_array = np.column_stack((all_ac_targets, all_ac_preds))
        ac_res_df = pd.DataFrame(ac_res_array, columns=['target', 'prediction'])
        ac_res_df.to_csv(result_path + "_ac.csv")
        all_ac_diffs = (all_ac_targets - all_ac_preds) ** 2
        all_ac_diffs = all_ac_diffs.reshape(-1, n_acs)
        all_types = np.hstack(all_types)
        ac_df = pd.DataFrame(all_ac_diffs)
        all_type_names = [id_2_type_dict[type_id] for type_id in all_types]
        ac_df['label'] = all_type_names
        ac_grouped_means = ac_df.groupby('label').mean()
        ac_grouped_means_df = pd.DataFrame(ac_grouped_means)
        ac_grouped_means_df.columns = data_loader.dataset.dataset.ac_feat_list
        ac_grouped_means_df.to_csv(result_path + "ac_diff.csv")
        ac_total_r2 = r2_score(all_ac_targets, all_ac_preds)
        ac_pcc = pearsonr(all_ac_targets, all_ac_preds)

        # Result dataframes
        # all_targets_df = pd.DataFrame(all_targets_array)
        # all_preds_df = pd.DataFrame(all_preds_array)
        # all_inputs_df = pd.DataFrame(all_inputs_array)
        # all_targets_df.columns = data_loader.dataset.dataset.feat_list
        # all_preds_df.columns = data_loader.dataset.dataset.feat_list
        # all_inputs_df.columns = data_loader.dataset.dataset.feat_list
        # all_targets_df.index = all_type_names
        # all_preds_df.index = all_type_names
        # all_inputs_df.index = all_type_names
        # all_targets_df.index = all_data_ids
        # all_preds_df.index = all_data_ids
        # all_inputs_df.index = all_data_ids
        # all_targets_df["TYPE"] = all_type_names
        # all_inputs_df["TYPE"] = all_type_names
        # all_preds_df["TYPE"] = all_type_names
        # all_targets_df.to_csv(result_path + "_targets.csv")
        # all_preds_df.to_csv(result_path + "_preds.csv")
        # all_inputs_df.to_csv(result_path + "_inputs.csv")

        return ph_total_r2, ph_pcc, ac_total_r2, ac_pcc
