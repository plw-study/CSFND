import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Fake News Detection with Clustering')

    # Dataset parameters
    parser.add_argument('--dataset', default='weibo', type=str,
                        help='dataset name')
    parser.add_argument('--option-file', default='./options.yaml',
                        help='options file in yaml.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for random, numpy and torch (default: 42).')
    # parser.add_argument('--lr', type=float, default=1e-4,
    #                     help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')

    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=10,
                        help='number of pre-train epochs')
    parser.add_argument('--es-patience', type=int, default=10,
                        help='patience of early stop')

    # Model parameters
    parser.add_argument('--n-clusters', type=int, default=17,
                        help='number of clusters')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='the margin for unsupervised context learning')
    parser.add_argument('--margin-class', default=0.2,
                        help='the margin for context-based triplet learning')
    parser.add_argument('--lambda-cluster', type=float, default=0.2,
                        help='the weight lambda of intra-cluster loss.')
    parser.add_argument('--lambda-triplet-class', type=float, default=0.6,
                        help='the weight lambda of context-based triplet loss.')

    return parser.parse_args()
