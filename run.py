import random
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import get_saved_model_outputs, get_cluster_ids, load_test_clu_ids, \
    calculate_cluster_centers, clustering_judge
from data_loader import load_data
from engine import train_model, test_model, unsuper_learning
from model import load_main_model, load_unsup_model
from pytorchtools_epoch import EarlyStopping
from args import parse_args
from classifiers import multi_classifiers


def run():
    args = parse_args()
    args.begin_time = time.strftime('%m%d-%H%M%S', time.localtime())
    print('===> start training at: ', args.begin_time)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load the data and configs
    with open(args.option_file, 'r') as f_opt:
        options = yaml.load(f_opt, Loader=yaml.FullLoader)
    # 根据数据集的不同，tokens个数不同，导致输入数据的维度不同
    options['flat_dim_t'] = options[args.dataset]['token_length']*768

    train_loader, valid_loader, test_loader = load_data(args.dataset, args.batch_size, options)
    # 对训练数据中文本和图像分别进行聚类，得到每个模态的聚类伪标签
    print('===> cluster text and image items and get the cluster pseudo-label.')
    text_image_ids, te_cluster_ids, im_cluster_ids = get_cluster_ids(
        args.dataset, train_loader, options, args.n_clusters)

    # Unsupervised context learning stage
    print('===> perform the Unsupervised context learning to get the context features.')
    unsup_model = load_unsup_model(args.dataset, options, args)
    unsup_checkpoint = './{}_bs{}_Clu{}_unsupervised.pt'.format(args.dataset, args.batch_size, args.n_clusters)
    if not os.path.exists(unsup_checkpoint):
        unsuper_learning(train_loader, unsup_model,
                      text_image_ids, te_cluster_ids, im_cluster_ids, args, options, unsup_checkpoint)
    else:
        print('     load unsupervised model that trained success.')
        unsup_model.load_state_dict(torch.load(unsup_checkpoint)['state_dict'])

    # 从训练好的unsup模型中，得到valid和test数据的聚类伪标签，用于主模型的训练和测试
    train_te_clu_centers, train_im_clu_centers = calculate_cluster_centers(
        unsup_model, train_loader, args.n_clusters, options['out_dim_t'],
        text_image_ids, te_cluster_ids, im_cluster_ids)
    valid_text_image_ids, valid_te_clu_ids, valid_im_clu_ids = load_test_clu_ids(
        unsup_model, valid_loader, train_te_clu_centers, train_im_clu_centers)
    test_text_image_ids, test_te_clu_ids, test_im_clu_ids = load_test_clu_ids(
        unsup_model, test_loader, train_te_clu_centers, train_im_clu_centers)
    del train_te_clu_centers, train_im_clu_centers

    print('===> training the main model.')
    final_saved_model_path = './FinalModel_{}_clu{}_lClu{}_lTri{}.pt'.format(
        args.dataset, args.n_clusters, args.lambda_cluster, args.lambda_triplet_class)
    main_model = load_main_model(args.dataset, options, args)

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizers = torch.optim.Adam(
        main_model.parameters(), lr=options[args.dataset]['learning_rate'], weight_decay=float(args.wd))

    # 用聚类度量选择主模块用的模态
    print('    => select the primary modality: ', end='')
    modality = clustering_judge(train_loader, unsup_model, text_image_ids, te_cluster_ids, im_cluster_ids, args)
    print(modality)

    early_stopping = EarlyStopping(patience=args.es_patience, verbose=False, path=final_saved_model_path)
    for epoch in range(args.epoch):
        print('  Epoch {:02d}'.format(epoch), end='| ')
        train_model(train_loader, main_model, modality, text_image_ids, te_cluster_ids, im_cluster_ids,
                    optimizers, criterion, args, epoch, unsup_model)
        test_acc = test_model(valid_loader, main_model, modality, valid_text_image_ids, valid_te_clu_ids,
                              valid_im_clu_ids, criterion, args, epoch, unsup_model)

        early_stopping(-test_acc, main_model, epoch)
        if early_stopping.early_stop:
            print('  training stop in epoch {}.'.format(epoch-args.es_patience))
            print('  the best model had been saved to ', final_saved_model_path)
            break

    main_model.load_state_dict(torch.load(final_saved_model_path)['state_dict'])

    # 使用多个二分类器，分别对每一个cluster中的news进行分类
    print('===> construct multiple binary classifiers to detect fake news within each news cluster.')
    # train_out_hidden_embs, train_out_label_embs, train_cluster_labels
    train_out_embs = get_saved_model_outputs(main_model, train_loader, modality,
                                             text_image_ids, te_cluster_ids, im_cluster_ids, unsup_model)
    test_out_embs = get_saved_model_outputs(main_model, test_loader, modality,
                                            test_text_image_ids, test_te_clu_ids, test_im_clu_ids, unsup_model)

    f = open('./{}_results.txt'.format(args.dataset), 'a+')
    fake_res, real_res = multi_classifiers(
        train_out_embs, test_out_embs, args.n_clusters, 'mlp', verbose=False)
    f.write('{:<8}_CLU{:<2d}_lambdaClu{:<.2f}_lambdaTri{:<.2f}_{:<8}|'.format(
        args.dataset, args.n_clusters, args.lambda_cluster, args.lambda_triplet_class, 'mlp'))
    # fake pre recall f1 acc, real pre recall f1 acc
    f.write('{:<.4f} & {:<.4f} & {:<.4f} & {:<.4f} & {:<.4f} & {:<.4f} & {:<.4f} & {:<.4f}\n'.format(
        fake_res[0], fake_res[1], fake_res[2], fake_res[3],
        real_res[0], real_res[1], real_res[2], real_res[3]))
    print('===> final test results: fake pre {:<.4f} recall {:<.4f} f1 {:<.4f} acc {:<.4f}.'.format(
        fake_res[0], fake_res[1], fake_res[2], fake_res[3]
    ))
    f.close()


if __name__ == '__main__':
    run()
