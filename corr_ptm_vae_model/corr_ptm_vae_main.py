import argparse
from corr_ptm_vae_model import utils
from torch.utils.data import DataLoader
import corr_ptm_vae
import torch
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples:")
    parser.add_argument('--input', '-i', type=str, default='data/filtered_tumor_normal.csv',
                        help='the input file')
    parser.add_argument('--target', '-t', type=str, default='data/filtered_common_protein_tumor_normal.csv',
                        help='the target file')
    # ptm input
    parser.add_argument('--ptm_input', '-pi', type=str, default='data/filtered_common_phospho_tumor_normal.csv')
    parser.add_argument('--ac_input', '-ai', type=str, default='data/filtered_common_tumor_acel.csv')

    parser.add_argument('--sample_dict', '-sd', type=str, default='data/rna_tumor_sample_dict.csv')
    parser.add_argument('--output', '-o', type=str, help='the output folder', default='output')
    parser.add_argument('--use_ptm', '-up', action='store_true', default=False)
    parser.add_argument('--use_ac', '-ua', action='store_true', default=False)
    parser.add_argument('--use_mask', '-um', action='store_true', default=False)
    parser.add_argument('--no_zeros', '-nz', action='store_true', default=False)
    parser.add_argument('--scale_target', '-st', action='store_true',default=False)
    parser.add_argument('--scale_input', '-si', action='store_true', default=False)
    # model
    parser.add_argument('--random_seed', '-rs', type=int, default=0)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--embedding_dim', '-ed', type=int, default=128)
    parser.add_argument('--no_type', action='store_true', default=False)
    parser.add_argument('--type_embedding_dim', '-ted', type=int, default=32)
    parser.add_argument('--hidden_dim', '-hd', type=int, default=128)
    parser.add_argument('--latent_dim', '-ld', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_epoch', '-ne', type=int, help='the number of training epochs', default=400)
    parser.add_argument('--model_output', '-mo', type=str, default='output/model')
    parser.add_argument('--n_layer', type=int, help='the number of encoder layers', default=3)
    parser.add_argument('--test_only', action='store_true', help='only do test with trained models', default=False)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--load_model_path', '-l', default='trained_corr_ptm_vae_model.pt')
    parser.add_argument('--model_class', '-mc', type=int, default=1)
    parser.add_argument('--predict_de_novo', '-pdn', action='store_true', default=False)
    parser.add_argument('--predict_from_normal', '-pfn', action='store_true', default=False)
    parser.add_argument('--predict_from_tumor', '-pft', action='store_true',default=False)
    parser.add_argument('--continue_train', '-ct', action='store_true', default=False)
    parser.add_argument('--linear_layer', '-ll', action='store_true', default=False)
    # clustering
    parser.add_argument('--compare_cluster', '-cc', action='store_true', default=False)
    parser.add_argument('--compare_correlation', '-ccr', action='store_true', default=False)
    parser.add_argument('--num_clusters', '-nc', type=int, default=9)
    parser.add_argument('--visualize', '-v', action='store_true', default=False)
    parser.add_argument('--cluster_method', '-cm', type=str, default='kmeans')

    args = parser.parse_args()
    gene_dict = "data/gene_dict.csv"
    if args.model_class == 1:
        print("Reading tumor data ...")
        model_class = "TUMOR"
    else:
        print("Reading normal data ...")
        model_class = "NORMAL"

    if not args.predict_de_novo:
        model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                       args.use_ptm, args.use_ac, args.sample_dict,
                                       model_class, feat_dict=gene_dict,no_zeros=args.no_zeros,
                                       scale_target=args.scale_target)
        model_train_dataset, model_test_dataset = utils.stratified_split(model_data, args.test_size,
                                                                         len(model_data.data_type_dict),
                                                                         random_state=42)
    elif args.predict_from_normal:
        if model_class == 1:
            test_class = "NORMAL"
        else:
            test_class = "TUMOR"
        model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                       args.use_ptm, args.use_ac, args.sample_dict,
                                       model_class, feat_dict=gene_dict,no_zeros=args.no_zeros)
        model_train_dataset, _ = utils.stratified_split(model_data, 0, len(model_data.data_type_dict), random_state=42)
        test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                      args.use_ptm, args.use_ac, args.sample_dict,
                                      test_class, feat_dict=gene_dict,
                                      data_type_dict=model_data.data_type_dict)
        _, model_test_dataset = utils.stratified_split(test_data, args.test_size, len(test_data.data_type_dict),
                                                       random_state=42)
        # _, model_test_dataset = utils.stratified_split(test_data, 1, len(test_data.data_type_dict),
        #                                                random_state=42)
        # model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
        #                                args.use_ptm, args.use_ac, 'data/common_sample_dict.csv',
        #                                "NORMAL", feat_dict=gene_dict, no_zeros=args.no_zeros)
        # model_train_dataset, _ = utils.stratified_split(model_data, 0, len(model_data.data_type_dict), random_state=42)
        # test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
        #                               args.use_ptm, args.use_ac, 'data/part_sample_dict.csv',
        #                               "TUMOR", feat_dict=gene_dict)
        # _, model_test_dataset = utils.stratified_split(test_data, args.test_size, len(test_data.data_type_dict),
        #                                                random_state=42)
        # test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
        #                               args.use_ptm, args.use_ac, 'data/part_sample_dict.csv',
        #                               "NORMAL", feat_dict=gene_dict)
        _, model_test_dataset = utils.stratified_split(model_data, 1, len(model_data.data_type_dict), random_state=42)
    elif args.predict_from_tumor:
        model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                       args.use_ptm, args.use_ac, 'data/common_sample_dict.csv',
                                       "TUMOR", feat_dict=gene_dict, no_zeros=args.no_zeros)
        model_train_dataset, _ = utils.stratified_split(model_data, 0, len(model_data.data_type_dict), random_state=42)
        # test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
        #                               args.use_ptm, args.use_ac, 'data/common_sample_dict.csv',
        #                               "NORMAL", feat_dict=gene_dict)
        # _, model_test_dataset = utils.stratified_split(test_data, args.test_size, len(test_data.data_type_dict),
        #                                                random_state=42)
        test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                      args.use_ptm, args.use_ac, 'data/common_sample_dict.csv',
                                      "TUMOR", feat_dict=gene_dict)
        _, model_test_dataset = utils.stratified_split(test_data, 1, len(test_data.data_type_dict), random_state=42)
    else:
        data_type_dict = {'LSCC': 0, 'LUAD': 1, 'HNSCC': 2, 'CCRCC': 3, 'PDAC': 4, 'GBM': 0, 'HGSC': 1, 'COAD': 2,
                          'UCEC': 3, 'BRCA': 4}
        # model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
        #                                args.use_ptm, args.use_ac, 'data/part_sample_dict.csv',
        #                                "TUMOR", feat_dict=gene_dict, data_type_dict=data_type_dict)
        # model_train_dataset, _ = utils.stratified_split(model_data, 0, len(data_type_dict), random_state=42)
        # test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
        #                               args.use_ptm, args.use_ac, 'data/common_sample_dict.csv',
        #                               "TUMOR", feat_dict=gene_dict, data_type_dict=data_type_dict)
        # _, model_test_dataset = utils.stratified_split(test_data, args.test_size, len(data_type_dict), random_state=42)
        model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                       args.use_ptm, args.use_ac, 'data/common_sample_dict.csv',
                                       "TUMOR", feat_dict=gene_dict, data_type_dict=data_type_dict)
        model_train_dataset, _ = utils.stratified_split(model_data, 0, len(data_type_dict), random_state=42)
        test_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                      args.use_ptm, args.use_ac, 'data/part_sample_dict.csv',
                                      "TUMOR", feat_dict=gene_dict, data_type_dict=data_type_dict)
        _, model_test_dataset = utils.stratified_split(test_data, args.test_size, len(data_type_dict), random_state=42)

    n_feats = len(model_data[0].input_feat)
    n_types = len(model_data.data_type_dict)
    n_ptms = 0
    if model_data.ptm_df is not None:
        n_ptms = len(model_data.ptm_df)
    train_data = DataLoader(model_train_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=utils.collate_fn)
    test_data = DataLoader(model_test_dataset, batch_size=args.batch_size, shuffle=True,
                           collate_fn=utils.collate_fn)
    if not args.test_only:
        if not args.no_type:
            type_embedding_dim = args.type_embedding_dim
        else:
            type_embedding_dim = 0
        corr_ptm_vae_model = corr_ptm_vae.CorrPtmVAE(hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
                                                     embedding_dim=args.embedding_dim, n_feats=n_feats,
                                                     type_embedding_dim=args.type_embedding_dim, n_types=n_types,
                                                     device=args.device, n_ptms=n_ptms, no_type=args.no_type)
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        corr_ptm_vae_model.to(corr_ptm_vae_model.device)
        # corr_ptm_vae_model.model_training(train_data, args.n_epoch, test_data, args.model_output)
        corr_ptm_vae_model.model_training(train_data, args.n_epoch, train_data, args.model_output,
                                          use_mask=args.use_mask,
                                          no_type=args.no_type)
    print('Loading trained model')
    load_model_name = args.load_model_path
    load_model = torch.load(args.model_output + "/" + load_model_name)
    if args.continue_train:
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        load_model.to(args.device)
        # load_model.model_training(train_data, args.n_epoch, test_data, args.model_output)
        load_model.model_training(train_data, args.n_epoch, train_data, args.model_output, args.use_mask, args.no_type)
    result_path = f"output/corr_ptm_vae_model/corr_ptm_vae_model_{model_class}_pred"
    total_r2 = load_model.model_test(test_data, result_path=result_path, use_mask=args.use_mask, no_type=args.no_type)
    print("Total R2 score: " + str(total_r2))
    #

    if args.visualize:
        load_model.visualize_emb(model_data.feat_list, args.model_class, method="UMAP",
                                 cluster_method=args.cluster_method, num_clusters=args.num_clusters)
    torch.save(load_model, args.model_output + f"/tested_corr_ptm_vae_tumor_model_{model_class}.pt")
