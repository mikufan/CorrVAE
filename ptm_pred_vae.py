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
    parser.add_argument('--protein_input', '-pi', type=str, default='../data/filtered_common_protein_tumor_normal.csv',
                        help='the target file')
    parser.add_argument('--sample_dict', '-sd', type=str, default='../data/rna_tumor_sample_dict.csv')
    parser.add_argument('--output', '-o', type=str, help='the output folder', default='output')
    # ptm_input
    parser.add_argument('--ph_target', '-pt', type=str, default='../data/filtered_common_phospho_tumor_normal.csv')
    parser.add_argument('--ac_target', '-at', type=str, default='../data/filtered_common_acel_tumor_normal.csv')
    # model
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
    parser.add_argument('--load_model_path', '-l', default='trained_corr_ptm_vae_model.pt')
    parser.add_argument('--model_class', '-mc', type=int, default=1)
    parser.add_argument('--predict_from_normal', '-pfn', action='store_true', default=False)
    parser.add_argument('--continue_train', '-ct', action='store_true', default=False)
    parser.add_argument('--ad_hoc', '-ah', action='store_true', default=False)
    parser.add_argument('--unified_train', '-ut', action='store_true', default=False)

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

    model_data = utils.OmicDataset(args.input, args.protein_input, args.sample_dict, model_class, feat_dict=gene_dict,
                                   ph_input=args.ph_target, ac_input=args.ac_target)

    model_train_dataset, model_valid_dataset, model_test_dataset = utils.stratified_3_split(model_data,
                                                                                            1 - args.test_size,
                                                                                            0.5 * args.test_size,
                                                                                            len(model_data.data_type_dict),
                                                                                            random_state=42)
    n_feats = len(model_data[0].input_feat)
    n_types = len(model_data.data_type_dict)
    n_phs = len(model_data.ph_df)
    n_acs = len(model_data.ac_df)
    train_data = DataLoader(model_train_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=utils.collate_fn)
    valid_data = DataLoader(model_valid_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=utils.collate_fn)
    test_data = DataLoader(model_test_dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=utils.collate_fn)

    if not args.test_only:
        corr_vae_model = corr_vae.PtmCorrVAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
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
        normal_train_data = Subset(model_data, normal_train_indices)
        tumor_train_data = Subset(model_data, tumor_train_indices)
        normal_train_dataset = DataLoader(normal_train_data, batch_size=args.batch_size, shuffle=True,
                                          collate_fn=utils.collate_fn)
        normal_train_result_path = result_path + "_train_normal_part"
        normal_train_ph_total_r2, normal_train_ph_pcc, normal_train_ac_total_r2, normal_train_ac_pcc \
            = load_model.model_test(normal_train_dataset, result_path=normal_train_result_path)
        print("train PH PCC is " + str(normal_train_ph_pcc))
        print("Total train PH R2 score: " + str(normal_train_ph_total_r2))
        print("train AC PCC is " + str(normal_train_ph_pcc))
        print("Total train AC R2 score: " + str(normal_train_ph_total_r2))

        tumor_train_dataset = DataLoader(tumor_train_data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=utils.collate_fn)
        tumor_train_result_path = result_path + "_train_tumor_part"
        tumor_train_ph_total_r2, tumor_train_ph_pcc, tumor_train_ac_total_r2, tumor_train_ac_pcc \
            = load_model.model_test(tumor_train_dataset, result_path=tumor_train_result_path)
        print("train PH PCC is " + str(tumor_train_ph_pcc))
        print("Total train PH R2 score: " + str(tumor_train_ph_total_r2))
        print("train AC PCC is " + str(tumor_train_ac_pcc))
        print("Total train AC R2 score: " + str(tumor_train_ac_total_r2))

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
        normal_valid_ph_total_r2, normal_valid_ph_pcc, normal_valid_ac_total_r2, normal_valid_ac_pcc \
            = load_model.model_test(normal_valid_dataset, result_path=normal_valid_result_path)
        print("Valid PH PCC is " + str(normal_valid_ph_pcc))
        print("Total valid PH R2 score: " + str(normal_valid_ph_total_r2))
        print("Valid AC PCC is " + str(normal_valid_ac_pcc))
        print("Total valid AC R2 score: " + str(normal_valid_ac_total_r2))

        tumor_valid_dataset = DataLoader(tumor_valid_data, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=utils.collate_fn)
        tumor_valid_result_path = result_path + "_valid_tumor_part"
        tumor_valid_ph_total_r2, tumor_valid_ph_pcc,tumor_valid_ac_total_r2, tumor_valid_ac_pcc \
            = load_model.model_test(tumor_valid_dataset, result_path=tumor_valid_result_path)
        print("Valid PH PCC is " + str(tumor_valid_ph_pcc))
        print("Total valid PH R2 score: " + str(tumor_valid_ph_total_r2))
        print("Valid AC PCC is " + str(tumor_valid_ac_pcc))
        print("Total valid AC R2 score: " + str(tumor_valid_ac_total_r2))

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
        normal_ph_total_r2, normal_ph_pcc,normal_ac_total_r2, normal_ac_pcc \
            = load_model.model_test(normal_test_dataset, result_path=normal_result_path)
        print("PH PCC is " + str(normal_ph_pcc))
        print("Total PH R2 score: " + str(normal_ph_total_r2))
        print("AC PCC is " + str(normal_ac_pcc))
        print("Total AC R2 score: " + str(normal_ac_total_r2))

        tumor_test_dataset = DataLoader(tumor_test_data, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=utils.collate_fn)
        tumor_result_path = result_path + "_tumor_part"
        tumor_ph_total_r2, tumor_ph_pcc,tumor_ac_total_r2, tumor_ac_pcc  \
            = load_model.model_test(tumor_test_dataset, result_path=tumor_result_path)
        print("PH PCC is " + str(tumor_ph_pcc))
        print("Total R2 score: " + str(tumor_ph_total_r2))
        print("PCC is " + str(tumor_ac_pcc))
        print("Total R2 score: " + str(tumor_ac_total_r2))

    if not args.unified_train:
        torch.save(load_model, args.model_output + f"/tested_corr_ptm_vae_tumor_model_{model_class}.pt")
    else:
        torch.save(load_model, args.model_output + f"/tested_corr_ptm_vae_tumor_model_TUMOR_NORMAL.pt")
