import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_name', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2', help='which model choose to fittune')
    parser.add_argument('-max_seq_len', type=int, default=256,
                        help='max sequence length of the model choosed')
    parser.add_argument('-dim', type=int, default=384,
                        help='embedding dimension of the model choosed')
    parser.add_argument('-article_directory', type=str,
                        default='./data/articles', help='articles directory')
    parser.add_argument('-data_vector_path', type=str,
                        default='./data.fvecs', help='data vector file path')
    parser.add_argument('-data_index_path', type=str,
                        default='./data_index.npy', help='data index file path')
    parser.add_argument('-gamma', type=float, default=0.8,
                        help='the threshold of cheat')
    parser.add_argument('-report_path', type=str,
                        default='./report.json', help='report path')
    args = parser.parse_args()

    return args
