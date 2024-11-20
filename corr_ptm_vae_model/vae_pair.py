import argparse
from corr_ptm_vae_model import utils
from torch.utils.data import DataLoader
import corr_ptm_vae
import vae_pair_model
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

    model_data = utils.OmicDataset(args.input, args.target, args.ptm_input, args.ac_input,
                                   args.use_ptm, args.use_ac, args.sample_dict,
                                   model_class, feat_dict=gene_dict,no_zeros=args.no_zeros,
                                   scale_target=args.scale_target)
    if model_data.ptm_df is not None:
        n_ptms = len(model_data.ptm_df)
    batch_data = DataLoader(model_data, batch_size=args.batch_size, shuffle=True, collate_fn=utils.collate_fn)
    print('Loading trained model')
    load_model_name = args.load_model_path
    load_model = torch.load(args.model_output + "/" + load_model_name)

    if args.device != "cpu":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        load_model.to(args.device)
    load_model.model_test(model_data, args.n_epoch, model_data, args.model_output, args.use_mask, args.no_type)
    result_path = f"output/corr_ptm_vae_model/corr_ptm_vae_model_{model_class}_pred"
    total_r2 = load_model.pair_model_test(model_data, result_path=result_path, use_mask=args.use_mask, no_type=args.no_type)
    print("Total R2 score: " + str(total_r2))
    #

    if args.visualize:
        load_model.visualize_emb(model_data.feat_list, args.model_class, method="UMAP",
                                 cluster_method=args.cluster_method, num_clusters=args.num_clusters)
    torch.save(load_model, args.model_output + f"/tested_corr_ptm_vae_pair_model_{model_class}.pt")