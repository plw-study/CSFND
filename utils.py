import numpy as np
from sklearn.cluster import KMeans
import torch
from pre_models import load_pre_models
import os
from sklearn.metrics import calinski_harabasz_score


def sklearn_kmeans(data, n_clusters):
    kmeans_model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=1)
    kmeans_model.fit(data)
    labels = kmeans_model.labels_
    # centroids = kmeans_model.cluster_centers_
    # print('---kmeans finish, cluster number: {}'.format(n_clusters))
    return labels


def torch_euclidean_dist_pairs(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def get_saved_model_outputs(model, input_data, modality, ti_ids, te_clu_ids, im_clu_ids, un_model):
    model.eval()
    un_model.eval()

    out_hidden_embs = []
    out_label_embs = []
    out_text_event_label_embs = []
    out_image_event_label_embs = []
    out_cls_emb = []

    for da in input_data:
        input_text, input_image, input_label, input_ti_id, _ = da

        te_input_cluster_ids = np.array([])
        im_input_cluster_ids = np.array([])
        for i in input_ti_id:
            te_input_cluster_ids = np.append(te_input_cluster_ids, te_clu_ids[np.where(ti_ids == i)])
            im_input_cluster_ids = np.append(im_input_cluster_ids, im_clu_ids[np.where(ti_ids == i)])
        te_input_cluster_ids = torch.tensor(te_input_cluster_ids).cuda()
        im_input_cluster_ids = torch.tensor(im_input_cluster_ids).cuda()

        with torch.no_grad():
            con_rep_t, con_rep_v, = un_model(input_text, input_image)
            _, _, _, _, cs_rep_t, cs_rep_v, mm_fused_out, cls_emb = model(
                input_text, input_image, te_input_cluster_ids, im_input_cluster_ids, con_rep_t, con_rep_v)

        out_hidden_embs.extend(mm_fused_out.detach().cpu().numpy())
        out_label_embs.extend(input_label.detach().cpu().numpy())
        out_cls_emb.extend(cls_emb.detach().cpu().numpy())
        out_text_event_label_embs.extend(te_input_cluster_ids.detach().cpu().numpy())
        out_image_event_label_embs.extend(im_input_cluster_ids.detach().cpu().numpy())

    out_hidden_embs = np.array(out_hidden_embs)
    out_label_embs = np.array(out_label_embs)
    out_cls_emb = np.array(out_cls_emb)
    out_text_event_label_embs = np.array(out_text_event_label_embs)
    out_image_event_label_embs = np.array(out_image_event_label_embs)

    if modality == 'text':
        cluster_labels = out_text_event_label_embs
    elif modality == 'image':
        cluster_labels = out_image_event_label_embs

    return out_hidden_embs, out_label_embs, out_cls_emb, cluster_labels


def get_cluster_ids(dataset, train_data, opts, n_clusters):
    text_clu_file_name = './data_files/{}/Clu{}_TEXT_cluids.npy'.format(dataset, n_clusters)
    image_clu_file_name = './data_files/{}/Clu{}_IMAGE_cluids.npy'.format(dataset, n_clusters)
    ti_file_name = './data_files/{}/Clu{}_TEXT_IMAGE_ids.npy'.format(dataset, n_clusters)

    if os.path.exists(text_clu_file_name):
        te_im_id_embs = np.load(ti_file_name)
        text_cluster_ids = np.load(text_clu_file_name)
        image_cluster_ids = np.load(image_clu_file_name)
    else:
        bertModel = load_pre_models(opts[dataset]['bert_model'], opts[dataset]['fine_tune_bert'], 'text')
        vggModel = load_pre_models(opts[dataset]['vgg_model'], opts[dataset]['fine_tune_vgg'], 'image')
        te_im_id_embs = []
        cls_embeddings = []
        image_token_embeddings = []

        for one_data in train_data:
            input_text, input_image, _, te_im_ids, _ = one_data

            with torch.no_grad():
                im_token_emb = vggModel(input_image)
                cls_emb, _ = bertModel(input_text)  # cls_emb: [bs, dim]  token_emb: [bs, n_token, dim]

            image_token_embeddings.extend(im_token_emb.cpu().numpy())
            cls_embeddings.extend(cls_emb.cpu().numpy())
            te_im_id_embs.extend(te_im_ids)

        text_cluster_ids = sklearn_kmeans(np.array(cls_embeddings), n_clusters=n_clusters)
        image_cluster_ids = sklearn_kmeans(np.array(image_token_embeddings), n_clusters=n_clusters)

        del cls_embeddings
        del image_token_embeddings
        te_im_id_embs = np.array(te_im_id_embs)
        text_cluster_ids = np.array(text_cluster_ids)
        image_cluster_ids = np.array(image_cluster_ids)
        np.save(text_clu_file_name, text_cluster_ids)
        np.save(image_clu_file_name, image_cluster_ids)
        np.save(ti_file_name, te_im_id_embs)

    assert len(te_im_id_embs) == len(text_cluster_ids)
    assert len(te_im_id_embs) == len(image_cluster_ids)
    # print('   cluster number is {}.'.format(n_clusters))
    # te_cluster_nums = np.bincount(text_cluster_ids, minlength=n_clusters)
    # im_cluster_nums = np.bincount(image_cluster_ids, minlength=n_clusters)
    # print('    text data numbers in each cluster:', te_cluster_nums)
    # print('    image data numbers in each cluster:', im_cluster_nums)
    return te_im_id_embs, text_cluster_ids, image_cluster_ids


def clustering_judge(all_loader, model, ti_ids, te_clu_ids, im_clu_ids, args):
    model.eval()
    text_mlp2_embs = []
    image_mlp2_embs = []

    for batch_data in all_loader:
        input_text, input_image, input_label, input_ti_id, _ = batch_data

        te_input_cluster_ids = np.array([])
        im_input_cluster_ids = np.array([])
        for i in input_ti_id:
            te_input_cluster_ids = np.append(te_input_cluster_ids, te_clu_ids[np.where(ti_ids == i)])
            im_input_cluster_ids = np.append(im_input_cluster_ids, im_clu_ids[np.where(ti_ids == i)])
        te_input_cluster_ids = torch.tensor(te_input_cluster_ids).cuda()
        im_input_cluster_ids = torch.tensor(im_input_cluster_ids).cuda()
        assert len(te_input_cluster_ids) == len(input_ti_id)
        assert len(im_input_cluster_ids) == len(input_ti_id)

        with torch.no_grad():
            mlp2_t, mlp2_v = model(input_text, input_image)

        text_mlp2_embs.extend(mlp2_t.detach().cpu().numpy())
        image_mlp2_embs.extend(mlp2_v.detach().cpu().numpy())

    text_mlp2_embs = np.array(text_mlp2_embs)
    image_mlp2_embs = np.array(image_mlp2_embs)

    text_cluster_ids = sklearn_kmeans(text_mlp2_embs, n_clusters=args.n_clusters)
    image_cluster_ids = sklearn_kmeans(image_mlp2_embs, n_clusters=args.n_clusters)

    text_ch = calinski_harabasz_score(text_mlp2_embs, text_cluster_ids)
    image_ch = calinski_harabasz_score(image_mlp2_embs, image_cluster_ids)
    # print('   the ch  score of text and image, higher is better:', text_ch, image_ch)

    if text_ch > image_ch:
        return 'text'
    else:
        return 'image'


def calculate_cluster_centers(model, train_loader, n_clusters, dim, ti_ids, te_clu_ids, im_clu_ids):
    model.eval()
    cluster_centers_t = torch.zeros((n_clusters, dim)).cuda()
    cluster_centers_v = torch.zeros((n_clusters, dim)).cuda()
    cluster_data_count_t = torch.zeros((n_clusters, 1)).cuda()
    cluster_data_count_v = torch.zeros((n_clusters, 1)).cuda()

    for batch_data in train_loader:
        input_text, input_image, _, input_ti_id, _ = batch_data

        te_input_cluster_ids = np.array([])
        im_input_cluster_ids = np.array([])
        for i in input_ti_id:
            te_input_cluster_ids = np.append(te_input_cluster_ids, te_clu_ids[np.where(ti_ids == i)])
            im_input_cluster_ids = np.append(im_input_cluster_ids, im_clu_ids[np.where(ti_ids == i)])
        assert len(te_input_cluster_ids) == len(input_ti_id)
        assert len(im_input_cluster_ids) == len(input_ti_id)

        with torch.no_grad():
            mlp2_t, mlp2_v = model(input_text, input_image)

        for one_id, one_mlp in zip(te_input_cluster_ids, mlp2_t):
            cluster_data_count_t[int(one_id)] += 1.0
            cluster_centers_t[int(one_id)] += one_mlp

        for one_id, one_mlp in zip(im_input_cluster_ids, mlp2_v):
            cluster_data_count_v[int(one_id)] += 1.0
            cluster_centers_v[int(one_id)] += one_mlp

    return cluster_centers_t/cluster_data_count_t, cluster_centers_v/cluster_data_count_v


def load_test_clu_ids(model, test_loader, tr_te_centers, tr_im_centers):
    model.eval()
    test_te_clu_ids = []
    test_im_clu_ids = []
    test_ti_ids = []

    for batch_data in test_loader:
        input_text, input_image, _, input_ti_id, _ = batch_data

        with torch.no_grad():
            mlp2_t, mlp2_v = model(input_text, input_image)

        dist_t = torch_euclidean_dist_pairs(mlp2_t, tr_te_centers)
        dist_v = torch_euclidean_dist_pairs(mlp2_v, tr_im_centers)
        batch_te_clu_ids = torch.min(dist_t, dim=1)[1]
        batch_im_clu_ids = torch.min(dist_v, dim=1)[1]

        test_te_clu_ids.extend(batch_te_clu_ids.detach().cpu().numpy())
        test_im_clu_ids.extend(batch_im_clu_ids.detach().cpu().numpy())
        test_ti_ids.extend(input_ti_id)

    return np.array(test_ti_ids), np.array(test_te_clu_ids), np.array(test_im_clu_ids)

