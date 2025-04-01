import argparse

import pandas as pd

from corr_vae_model import utils
from torch.utils.data import DataLoader, Subset
import corr_vae
import torch
import sys
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples:")
    parser.add_argument('--input', '-i', type=str, default='../data/filtered_tumor_normal.csv',
                        help='the input file')
    parser.add_argument('--target', '-t', type=str, default='../data/filtered_common_protein_tumor_normal.csv',
                        help='the target file')
    parser.add_argument('--sample_dict', '-sd', type=str, default='../data/rna_tumor_sample_dict.csv')
    parser.add_argument('--output', '-o', type=str, help='the output folder', default='output')
    # ptm_input
    parser.add_argument('--ph_input', '-pi', type=str, default='../data/filtered_common_phospho_tumor_normal.csv')
    parser.add_argument('--ac_input', '-ai', type=str, default='../data/filtered_common_tumor_acel_df.csv')
    # model
    parser.add_argument('--use_ph', '-up', action='store_true', default=False)
    parser.add_argument('--use_ac', '-ua', action='store_true', default=False)
    parser.add_argument('--random_seed', '-rs', type=int, default=0)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--embedding_dim', '-ed', type=int, default=128)
    parser.add_argument('--type_embedding_dim', '-ted', type=int, default=32)
    parser.add_argument('--hidden_dim', '-hd', type=int, default=128)
    parser.add_argument('--latent_dim', '-ld', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_epoch', '-ne', type=int, help='the number of training epochs', default=400)
    parser.add_argument('--model_output', '-mo', type=str, default='../output/model')
    parser.add_argument('--n_layer', type=int, help='the number of encoder layers', default=3)
    # config
    parser.add_argument('--test_only', action='store_true', help='only do test with trained models', default=False)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--load_model_path', '-l', default='trained_corr_vae_model.pt')
    parser.add_argument('--model_class', '-mc', type=int, default=1)
    parser.add_argument('--predict_from_normal', '-pfn', action='store_true', default=False)
    parser.add_argument('--continue_train', '-ct', action='store_true', default=False)
    parser.add_argument('--learn_linear', '-ll', action='store_true', default=False)
    parser.add_argument('--load_linear', '-lo', action='store_true', default=False)
    parser.add_argument('--ad_hoc', '-ah', action='store_true', default=False)
    parser.add_argument('--unified_train', '-ut', action='store_true', default=False)
    # clustering
    parser.add_argument('--compare_cluster', '-cc', action='store_true', default=False)
    parser.add_argument('--num_clusters', '-nc', type=int, default=9)
    parser.add_argument('--visualize', '-v', action='store_true', default=False)
    parser.add_argument('--cluster_method', '-cm', type=str, default='kmeans')

    args = parser.parse_args()
    gene_dict = "../data/gene_dict.csv"

    if not args.unified_train:
        if args.model_class == 1:
            print("Reading tumor data ...")
            model_class = "TUMOR"
        else:
            print("Reading normal data ...")
            model_class = "NORMAL"
    else:
        print("Reading tumor and normal data ...")
        model_class = "TUMOR AND NORMAL"

    # ptm_input
    if args.use_ph:
        ph_input = args.ph_input
    else:
        ph_input = None
    if args.use_ac:
        ac_input = args.ac_input
    else:
        ac_input = None

    if not args.predict_from_normal:
        model_data = utils.OmicDataset(args.input, args.target, args.sample_dict, model_class, feat_dict=gene_dict,
                                       ph_input=ph_input, ac_input=ac_input)
        model_train_dataset, model_test_dataset = utils.stratified_split(model_data, args.test_size,
                                                                         len(model_data.data_type_dict),
                                                                         random_state=42)
        if args.ad_hoc:
            model_test_dataset = model_train_dataset
    else:
        if args.model_class == 1:
            test_class = "NORMAL"
        else:
            test_class = "TUMOR"
        model_data = utils.OmicDataset(args.input, args.target, args.sample_dict, model_class, feat_dict=gene_dict,
                                       ph_input=ph_input, ac_input=args.ac_input)
        model_train_dataset, _ = utils.stratified_split(model_data, 0, len(model_data.data_type_dict), random_state=42)
        test_data = utils.OmicDataset(args.input, args.target, args.sample_dict, test_class, feat_dict=gene_dict,
                                      data_type_dict=model_data.data_type_dict, ph_input=ph_input, ac_input=ac_input)
        _, model_test_dataset = utils.stratified_split(test_data, 1, len(test_data.data_type_dict), random_state=42)
    n_feats = len(model_data[0].input_feat)
    n_types = len(model_data.data_type_dict)
    n_phs = 0
    n_acs = 0
    if model_data.ph_df is not None:
        n_phs = len(model_data.ph_df)
    if model_data.ac_df is not None:
        n_acs = len(model_data.ac_df)
    train_data = DataLoader(model_train_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=utils.collate_fn)
    test_data = DataLoader(model_test_dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=utils.collate_fn)
    if args.learn_linear:
        if not args.load_linear:
            linear_df, bias_df, r2_df, target_df = model_data.get_linear(model_train_dataset.indices)
            linear_df.to_csv(
                f"../output/corr_vae_model/corr_vae_model_{model_class}_linear.csv")
            bias_df.to_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_bias.csv")
            target_df.to_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_linear_target.csv")
            r2_df.to_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_linear_r2.csv")
        else:
            linear_df = pd.read_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_linear.csv", index_col=0)
            bias_df = pd.read_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_bias.csv", index_col=0)
        # linear_mse_df, r2, pcc = model_data.get_linear_mse(model_train_dataset.indices, linear_df, bias_df)
        linear_mse_df, r2, pcc = model_data.get_linear_mse(model_test_dataset.indices, linear_df, bias_df)
        linear_mse_df.to_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_linear_mse.csv")
        print("Linear r2: " + str(r2))
        print("Linear pcc: " + str(pcc))
        # linear_corr_df, corr = model_data.get_linear_corr(model_test_dataset.indices, initial_linear_df,
        #                                                   bias_df)
        # linear_corr_df.to_csv(f"output/corr_ptm_vae_model/corr_ptm_vae_model_{model_class}_linear_corr.csv")
        # print("Linear correlation: " + str(corr))
        sys.exit()
    if not args.test_only:
        corr_vae_model = corr_vae.CorrVAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                                          embedding_dim=args.embedding_dim, n_feats=n_feats,
                                          type_embedding_dim=args.type_embedding_dim, n_types=n_types,
                                          n_phs=n_phs, n_acs=n_acs, device=args.device)
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        corr_vae_model.to(corr_vae_model.device)
        # corr_vae_model.model_training(train_data, args.n_epoch, test_data, args.model_output)
        corr_vae_model.model_training(train_data, args.n_epoch, train_data, args.model_output)
    print('Loading trained model')
    load_model_name = args.load_model_path
    load_model = torch.load(args.model_output + "/" + load_model_name)
    if args.continue_train:
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        load_model.to(args.device)
        # load_model.model_training(train_data, args.n_epoch, test_data, args.model_output)
        load_model.model_training(train_data, args.n_epoch, train_data, args.model_output)
    if not args.unified_train:
        result_path = f"../output/corr_vae_model/corr_vae_model_{model_class}_pred"
    else:
        result_path = f"../output/corr_vae_model/corr_vae_model_TUMOR_NORMAL_pred"
    if not args.unified_train:
        total_r2, pcc = load_model.model_test(test_data, result_path=result_path)
        print("PCC is " + str(pcc))
        print("Total R2 score: " + str(total_r2))
    else:
        normal_test_indices = [idx for idx in model_test_dataset.indices
                               if model_data[idx].sample_id.endswith('.N')]
        tumor_test_indices = [idx for idx in model_test_dataset.indices
                              if not model_data[idx].sample_id.endswith('.N')]
        normal_test_data = Subset(model_data, normal_test_indices)
        tumor_test_data = Subset(model_data, tumor_test_indices)
        normal_test_dataset = DataLoader(normal_test_data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=utils.collate_fn)
        normal_result_path = result_path + "_normal_part"
        normal_total_r2, normal_pcc = load_model.model_test(normal_test_dataset, result_path=normal_result_path)
        print("PCC is " + str(normal_pcc))
        print("Total R2 score: " + str(normal_total_r2))

        tumor_test_dataset = DataLoader(tumor_test_data, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=utils.collate_fn)
        tumor_result_path = result_path + "_tumor_part"
        tumor_total_r2, tumor_pcc = load_model.model_test(tumor_test_dataset, result_path=tumor_result_path)
        print("PCC is " + str(tumor_pcc))
        print("Total R2 score: " + str(tumor_total_r2))

    if args.compare_cluster:
        if args.model_class == 1:
            compare_cluster = "../output/clusters/pair_vae_gene_clusters_9_NORMAL.csv"
        else:
            compare_cluster = "../output/clusters/pair_vae_gene_clusters_9_TUMOR.csv"
    else:
        compare_cluster = None
    if args.visualize:
        load_model.visualize_emb(model_data.feat_list, args.model_class, method="UMAP",
                                 cluster_method=args.cluster_method, num_clusters=args.num_clusters)
    if not args.unified_train:
        torch.save(load_model, args.model_output + f"/tested_corr_vae_tumor_model_{model_class}.pt")
    else:
        torch.save(load_model, args.model_output + f"/tested_corr_vae_tumor_model_TUMOR_NORMAL.pt")
