import argparse
from corr_vae_model import utils
from torch.utils.data import DataLoader
import corr_vae
import corr_vae_pair_model
import torch
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples:")
    parser.add_argument('--input', '-i', type=str, default='../data/filtered_tumor_normal.csv',
                        help='the input file')
    parser.add_argument('--target', '-t', type=str, default='../data/filtered_common_protein_tumor_normal.csv',
                        help='the target file')
    # ptm input
    parser.add_argument('--ph_input', '-pi', type=str, default='../data/filtered_common_phospho_tumor_normal.csv')
    parser.add_argument('--ac_input', '-ai', type=str, default='../data/filtered_common_tumor_acel.csv')

    parser.add_argument('--sample_dict', '-sd', type=str, default='../data/rna_tumor_sample_dict.csv')
    parser.add_argument('--output', '-o', type=str, help='the output folder', default='output')
    parser.add_argument('--use_ph', '-up', action='store_true', default=False)
    parser.add_argument('--use_ac', '-ua', action='store_true', default=False)
    parser.add_argument('--scale_target', '-st', action='store_true', default=False)
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
    parser.add_argument('--model_output', '-mo', type=str, default='../output/model')
    parser.add_argument('--n_layer', type=int, help='the number of encoder layers', default=3)
    parser.add_argument('--test_only', action='store_true', help='only do test with trained models', default=False)
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--load_model_path', '-l', default='trained_corr_vae_model.pt')
    parser.add_argument('--trained_model_path', '-tp', default='vae_pair_model_')
    parser.add_argument('--model_class', '-mc', type=int, default=1)

    # clustering
    parser.add_argument('--compare_cluster', '-cc', action='store_true', default=False)
    parser.add_argument('--compare_correlation', '-ccr', action='store_true', default=False)
    parser.add_argument('--num_clusters', '-nc', type=int, default=9)
    parser.add_argument('--visualize', '-v', action='store_true', default=False)
    parser.add_argument('--cluster_method', '-cm', type=str, default='kmeans')

    args = parser.parse_args()
    gene_dict = "../data/gene_dict.csv"
    if args.model_class == 1:
        print("Reading tumor data ...")
        model_class = "TUMOR"
    else:
        print("Reading normal data ...")
        model_class = "NORMAL"
    if args.use_ph:
        ph_input = args.ph_input
    else:
        ph_input = None
    if args.use_ac:
        ac_input = args.ac_input
    else:
        ac_input = None

    model_data = utils.OmicDataset(args.input, args.target, args.sample_dict, model_class, feat_dict=gene_dict,
                                   ph_input=ph_input, ac_input=ac_input)
    n_phs = 0
    n_acs = 0
    if model_data.ph_df is not None:
        n_phs = len(model_data.ph_df)
    if model_data.ac_df is not None:
        n_acs = len(model_data.ac_df)
    batch_data = DataLoader(model_data, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn)
    print('Loading trained model')
    load_model_name = args.load_model_path
    load_model = torch.load(args.model_output + "/" + load_model_name)
    n_feats = len(model_data[0].input_feat)
    n_types = len(model_data.data_type_dict)
    device = "cpu"
    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        load_model.to(device)
    if not args.test_only:
        pair_model = corr_vae_pair_model.VAEPairModel(load_model, args.hidden_dim, args.latent_dim, args.embedding_dim,
                                                      n_feats, args.type_embedding_dim, args.device)
        pair_model.to(device)
        pair_model.pair_model_training(batch_data, args.n_epoch, batch_data, args.model_output, args.no_type)
    else:
        model_path = args.model_output + "/" + args.trained_model_path + model_class + ".pt"
        pair_model = torch.load(model_path)

    if args.visualize:
        pair_model.visualize_emb(model_data.feat_list, args.model_class, method="UMAP",
                                 cluster_method=args.cluster_method, num_clusters=args.num_clusters)
    torch.save(pair_model, args.model_output + f"/vae_pair_model_{model_class}.pt")
