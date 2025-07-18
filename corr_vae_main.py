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
    parser.add_argument('--ac_input', '-ai', type=str, default='../data/filtered_common_acel_tumor_normal.csv')
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
    parser.add_argument('--transformer_encoder', '-te', action='store_true', default=False)
    parser.add_argument('--linformer_encoder', '-le', action='store_true', default=False)
    parser.add_argument('--contrastive_loss', '-cl', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--custom_linformer', '-cle', action='store_true', default=False)
    # parser.add_argument('--attn_output','-ao',type=str, default='../output/corr_vae_model/attn_map.csv')
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
    parser.add_argument('--no_type', '-nt', action='store_true', default=False)
    parser.add_argument('--visualize_lv', '-lv', action='store_true', default=False)

    parser.add_argument('--ood_test', '-ot', action='store_true', default=False)
    parser.add_argument('--ood_test_path', '-oot', type=str, default='../data/brain_cptac_protein.csv')
    parser.add_argument('--ood_input_path', '-ooi', type=str, default="../data/brian_cptac_data_mrna_seq_v2_rsem.txt")
    parser.add_argument('--ood_ph_input_path', '-oop', type=str,
                        default="../data/data_phosphoprotein_quantification.txt")
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
        model_class = "TUMOR_AND_NORMAL"

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
        if not args.ad_hoc:
            # model_train_dataset, model_valid_test_dataset = utils.stratified_split(model_data, args.test_size,
            #                                                                  len(model_data.data_type_dict),
            #                                                                  random_state=42)
            model_train_dataset, model_valid_dataset, model_test_dataset = utils.stratified_3_split(model_data,
                                                                                                    1 - args.test_size,
                                                                                                    0.5 * args.test_size,
                                                                                                    len(model_data.data_type_dict),
                                                                                                    random_state=42)
        else:
            model_train_dataset = model_data
            model_valid_dataset = model_train_dataset
            model_test_dataset = model_train_dataset
    else:
        if args.model_class == 1:
            test_class = "NORMAL"
        else:
            test_class = "TUMOR"
        model_data = utils.OmicDataset(args.input, args.target, args.sample_dict, model_class, feat_dict=gene_dict,
                                       ph_input=ph_input, ac_input=args.ac_input)
        model_train_dataset, _ = utils.stratified_split(model_data, 0, len(model_data.data_type_dict), random_state=42)
        model_valid_dataset, _ = utils.stratified_split(model_data, 0, len(model_data.data_type_dict), random_state=42)
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
    valid_data = DataLoader(model_valid_dataset, batch_size=args.batch_size, shuffle=True,
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
        if not args.unified_train:
            valid_linear_mse_df, valid_r2, valid_pcc = model_data.get_linear_mse(model_valid_dataset.indices, linear_df,
                                                                                 bias_df)
            valid_linear_mse_df.to_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_valid_linear_mse.csv")
            print("Linear valid r2: " + str(valid_r2))
            print("Linear valid pcc: " + str(valid_pcc))
            linear_mse_df, r2, pcc = model_data.get_linear_mse(model_test_dataset.indices, linear_df, bias_df)
            linear_mse_df.to_csv(f"../output/corr_vae_model/corr_vae_model_{model_class}_linear_mse.csv")
            print("Linear r2: " + str(r2))
            print("Linear pcc: " + str(pcc))
            sys.exit()
        else:
            normal_train_indices = [idx for idx in model_train_dataset.indices
                                    if model_data[idx].sample_id.endswith('.N')]
            tumor_train_indices = [idx for idx in model_train_dataset.indices
                                   if not model_data[idx].sample_id.endswith('.N')]
            model_data.linear_fit(tumor_train_indices, linear_df, bias_df, "tumor", "train")
            model_data.linear_fit(normal_train_indices, linear_df, bias_df, "normal", "train")
            normal_valid_indices = [idx for idx in model_valid_dataset.indices
                                    if model_data[idx].sample_id.endswith('.N')]
            tumor_valid_indices = [idx for idx in model_valid_dataset.indices
                                   if not model_data[idx].sample_id.endswith('.N')]
            # model_data.linear_fit
            valid_normal_linear_mse_df, valid_normal_r2, valid_normal_pcc = model_data.get_linear_mse(
                normal_valid_indices,
                linear_df, bias_df)
            valid_normal_linear_mse_df.to_csv(
                f"../output/corr_vae_model/corr_vae_model_{model_class}_valid_linear_mse_normal.csv")
            print("Linear valid normal r2: " + str(valid_normal_r2))
            print("Linear valid normal pcc: " + str(valid_normal_pcc))
            valid_tumor_linear_mse_df, valid_tumor_r2, valid_tumor_pcc = model_data.get_linear_mse(
                tumor_valid_indices, linear_df, bias_df)
            valid_tumor_linear_mse_df.to_csv(
                f"../output/corr_vae_model/corr_vae_model_{model_class}_valid_linear_mse_tumor.csv")
            print("Linear valid tumor r2: " + str(valid_tumor_r2))
            print("Linear valid tumor pcc: " + str(valid_tumor_pcc))

            normal_test_indices = [idx for idx in model_test_dataset.indices
                                   if model_data[idx].sample_id.endswith('.N')]
            tumor_test_indices = [idx for idx in model_test_dataset.indices
                                  if not model_data[idx].sample_id.endswith('.N')]
            test_normal_linear_mse_df, test_normal_r2, test_normal_pcc = model_data.get_linear_mse(
                normal_test_indices,
                linear_df, bias_df)
            test_normal_linear_mse_df.to_csv(
                f"../output/corr_vae_model/corr_vae_model_{model_class}_test_linear_mse_normal.csv")
            print("Linear test normal r2: " + str(test_normal_r2))
            print("Linear test normal pcc: " + str(test_normal_pcc))
            test_tumor_linear_mse_df, test_tumor_r2, test_tumor_pcc = model_data.get_linear_mse(
                tumor_test_indices, linear_df, bias_df)
            test_tumor_linear_mse_df.to_csv(
                f"../output/corr_vae_model/corr_vae_model_{model_class}_test_linear_mse_tumor.csv")
            print("Linear test tumor r2: " + str(test_tumor_r2))
            print("Linear test tumor pcc: " + str(test_tumor_pcc))
            sys.exit()

    if not args.test_only:
        corr_vae_model = corr_vae.CorrVAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                                          embedding_dim=args.embedding_dim, n_feats=n_feats,
                                          type_embedding_dim=args.type_embedding_dim, n_types=n_types,
                                          n_phs=n_phs, n_acs=n_acs, device=args.device, no_types=args.no_type,
                                          trans_encoder=args.transformer_encoder, lin_encoder=args.linformer_encoder,
                                          con_loss=args.contrastive_loss, alpha=args.alpha,
                                          cus_line=args.custom_linformer)
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        corr_vae_model.to(corr_vae_model.device)
        corr_vae_model.model_training(train_data, args.n_epoch, valid_data, args.model_output)
        # corr_vae_model.model_training(train_data, args.n_epoch, train_data, args.model_output)
    print('Loading trained model')
    load_model_name = args.load_model_path
    if not args.no_type:
        load_model_name = args.load_model_path
    else:
        load_model_name = args.load_model_path
        load_model_name = load_model_name.split(".")[0] + "_no_type.pt"
    load_model = torch.load(args.model_output + "/" + load_model_name)
    if args.continue_train:
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        load_model.to(args.device)
        load_model.model_training(train_data, args.n_epoch, valid_data, args.model_output)
        # load_model.model_training(train_data, args.n_epoch, train_data, args.model_output)
    if not args.unified_train:
        result_path = f"../output/corr_vae_model/corr_vae_model_{model_class}_pred"
    else:
        result_path = f"../output/corr_vae_model/corr_vae_model_TUMOR_NORMAL_pred"
    if not args.unified_train:
        total_r2, pcc = load_model.model_test(test_data, result_path=result_path)
        print("PCC is " + str(pcc))
        print("Total R2 score: " + str(total_r2))
    else:
        if isinstance(model_train_dataset, Subset):
            normal_train_indices = [idx for idx in model_train_dataset.indices
                                    if model_data[idx].sample_id.endswith('.N')]
            tumor_train_indices = [idx for idx in model_train_dataset.indices
                                   if not model_data[idx].sample_id.endswith('.N')]
        else:
            normal_train_indices = [idx for idx in range(len(model_train_dataset))
                                    if model_data[idx].sample_id.endswith('.N')]
            tumor_train_indices = [idx for idx in range(len(model_train_dataset))
                                   if not model_data[idx].sample_id.endswith('.N')]
        if args.visualize_lv:
            r2, pcc, scc = load_model.model_test(train_data, result_path=result_path,lv_visualize=True)

        normal_train_data = Subset(model_data, normal_train_indices)
        tumor_train_data = Subset(model_data, tumor_train_indices)
        normal_train_dataset = DataLoader(normal_train_data, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=utils.collate_fn)
        normal_train_result_path = result_path + "_train_normal_part"
        normal_train_total_r2, normal_train_pcc, normal_train_scc = load_model.model_test(normal_train_dataset,
                                                                                          result_path=normal_train_result_path)
        print("Normal train PCC is " + str(normal_train_pcc))
        print("Normal train SCC is " + str(normal_train_scc))
        print("Total train R2 score: " + str(normal_train_total_r2))

        tumor_train_dataset = DataLoader(tumor_train_data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=utils.collate_fn)
        tumor_train_result_path = result_path + "_train_tumor_part"
        tumor_train_total_r2, tumor_train_pcc, tumor_train_scc = load_model.model_test(tumor_train_dataset,
                                                                                       result_path=tumor_train_result_path)
        print("Tumor train PCC is " + str(tumor_train_pcc))
        print("Tumor train SCC is " + str(tumor_train_scc))
        print("Total train R2 score: " + str(tumor_train_total_r2))

        if isinstance(model_valid_dataset, Subset):
            normal_valid_indices = [idx for idx in model_valid_dataset.indices
                                    if model_data[idx].sample_id.endswith('.N')]
            tumor_valid_indices = [idx for idx in model_valid_dataset.indices
                                   if not model_data[idx].sample_id.endswith('.N')]
        else:
            normal_valid_indices = [idx for idx in range(len(model_valid_dataset))
                                    if model_data[idx].sample_id.endswith('.N')]
            tumor_valid_indices = [idx for idx in range(len(model_valid_dataset))
                                   if not model_data[idx].sample_id.endswith('.N')]
        normal_valid_data = Subset(model_data, normal_valid_indices)
        tumor_valid_data = Subset(model_data, tumor_valid_indices)
        normal_valid_dataset = DataLoader(normal_valid_data, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=utils.collate_fn)
        normal_valid_result_path = result_path + "_valid_normal_part"
        normal_valid_total_r2, normal_valid_pcc, normal_valid_scc = load_model.model_test(normal_valid_dataset,
                                                                                          result_path=normal_valid_result_path)
        print("Normal valid PCC is " + str(normal_valid_pcc))
        print("Normal valid SCC is " + str(normal_valid_scc))
        print("Total valid R2 score: " + str(normal_valid_total_r2))

        tumor_valid_dataset = DataLoader(tumor_valid_data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=utils.collate_fn)
        tumor_valid_result_path = result_path + "_valid_tumor_part"
        tumor_valid_total_r2, tumor_valid_pcc, tumor_valid_scc = load_model.model_test(tumor_valid_dataset,
                                                                                       result_path=tumor_valid_result_path)
        print("Tumor Valid PCC is " + str(tumor_valid_pcc))
        print("Tumor Valid SCC is " + str(tumor_valid_scc))
        print("Total valid R2 score: " + str(tumor_valid_total_r2))

        if isinstance(model_test_dataset, Subset):
            normal_test_indices = [idx for idx in model_test_dataset.indices
                                   if model_data[idx].sample_id.endswith('.N')]
            tumor_test_indices = [idx for idx in model_test_dataset.indices
                                  if not model_data[idx].sample_id.endswith('.N')]

        else:
            normal_test_indices = [idx for idx in range(len(model_test_dataset))
                                   if model_data[idx].sample_id.endswith('.N')]
            tumor_test_indices = [idx for idx in range(len(model_test_dataset))
                                  if not model_data[idx].sample_id.endswith('.N')]

        normal_test_data = Subset(model_data, normal_test_indices)
        tumor_test_data = Subset(model_data, tumor_test_indices)
        normal_test_dataset = DataLoader(normal_test_data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=utils.collate_fn)
        normal_result_path = result_path + "_normal_part"
        normal_total_r2, normal_pcc, normal_scc = load_model.model_test(normal_test_dataset,
                                                                        result_path=normal_result_path)
        print("Normal PCC is " + str(normal_pcc))
        print("Normal SCC is " + str(normal_scc))
        print("Total R2 score: " + str(normal_total_r2))

        tumor_test_dataset = DataLoader(tumor_test_data, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=utils.collate_fn)
        tumor_result_path = result_path + "_tumor_part"
        tumor_total_r2, tumor_pcc, tumor_scc = load_model.model_test(tumor_test_dataset, result_path=tumor_result_path)
        print("Tumor PCC is " + str(tumor_pcc))
        print("Tumor SCC is " + str(tumor_scc))
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
    if args.ood_test:
        if not args.use_ph:
            ood_ph_input = None
        else:
            ood_ph_input = args.ood_ph_input_path
        ood_test_data = utils.OmicDataset(args.ood_input_path, args.ood_test_path, None, "TUMOR", feat_dict=gene_dict,
                                          ph_input=ood_ph_input, ac_input=None, common_feat=model_data.feat_list,
                                          ood_data=True, common_ph_feat=model_data.ph_feat_list)
        ood_test_dataset = DataLoader(ood_test_data, batch_size=args.batch_size, shuffle=True,
                                      collate_fn=utils.collate_fn)
        ood_result_path = result_path + "_ood_test"
        tumor_total_r2, tumor_pcc = load_model.model_ood_test(ood_test_dataset, result_path=ood_result_path)
        print("PCC is " + str(tumor_pcc))
        print("Total R2 score: " + str(tumor_total_r2))
