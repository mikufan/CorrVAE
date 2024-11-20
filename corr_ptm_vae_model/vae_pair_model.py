import utils
import torch
import torch.nn as nn
import torch.optim as optim


class VAEPairModel(nn.Module):
    def __init__(self, corr_vae_model, hidden_dim, latent_dim, embedding_dim, n_feats, type_embedding_dim, n_types,
                 device, n_ptms=0, n_acs=0, no_type=False):
        super(VAEPairModel, self).__init__()

        self.device = device

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

        self.n_feats = n_feats
        self.n_ptms = n_ptms
        self.dropout = nn.Dropout(0.3)

        # Embedding
        self.gene_embedding = nn.Embedding(n_feats, embedding_dim)
        self.type_embedding = corr_vae_model.type_embedding
        # Decoder
        self.fc3 = nn.Linear(latent_dim + type_embedding_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_feats)
        self.nc_fc4 = nn.Linear(hidden_dim, embedding_dim)
        self.decode_embedding = nn.Linear(embedding_dim, n_feats)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        # self.logger = utils.get_logger('../output/log/train_log.log')
        self.logger = None

    def forward(self, x, y, z, data_type, x_mask, y_mask, ptm_x=None, is_train=True, use_mask=False, no_type=False):
        x = x.to(self.device)
        y = y.to(self.device)
        if x_mask is not None:
            x_mask = x_mask.to(self.device)
        if y_mask is not None:
            y_mask = y_mask.to(self.device)
        if ptm_x is not None:
            ptm_x = ptm_x.to(self.device)
        data_type = data_type.to(self.device)
        emb_type = self.type_embedding(data_type)
        recon_y = self.decode(z, emb_type, no_type)
        if use_mask:
            recon_y = torch.mul(recon_y, y_mask)
        if is_train:
            vae_loss = vae_loss_function(recon_y, y, mu, logvar)
            return vae_loss
        else:
            if not return_z:
                return recon_y, y
            else:
                return z, recon_y, y


def pair_model_training(self, train_data, n_epochs, valid_data, model_path, use_mask=False, no_type=False):
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
