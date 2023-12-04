import numpy as np
import torch
from sklearn.metrics import accuracy_score


def train_model(train_data, model, modality, ti_ids, te_clu_ids, im_clu_ids, optimi, cls_cri, args, epoch, un_model):
    model.train()
    context_triplet_all_loss, intra_cluster_all_loss, predict_all_loss, train_all_loss = 0, 0, 0, 0
    skl_acc = 0.0
    train_num = 1
    un_model.eval()

    for step, train_d in enumerate(train_data):

        input_text, input_image, input_label, input_ti_id, _ = train_d
        if len(input_label) < args.batch_size:
            continue

        te_input_cluster_ids = np.array([])
        im_input_cluster_ids = np.array([])
        for i in input_ti_id:
            te_input_cluster_ids = np.append(te_input_cluster_ids, te_clu_ids[np.where(ti_ids == i)])
            im_input_cluster_ids = np.append(im_input_cluster_ids, im_clu_ids[np.where(ti_ids == i)])
        te_input_cluster_ids = torch.tensor(te_input_cluster_ids).cuda()
        im_input_cluster_ids = torch.tensor(im_input_cluster_ids).cuda()
        if len(te_input_cluster_ids) != len(input_ti_id):
            print(len(te_clu_ids), len(im_clu_ids))
            print(step, len(te_input_cluster_ids), len(input_ti_id), len(input_text))
            raise ValueError()
        assert len(te_input_cluster_ids) == len(input_ti_id)
        assert len(im_input_cluster_ids) == len(input_ti_id)

        with torch.no_grad():
            contextual_rep_t, contextual_rep_v = un_model(input_text, input_image)

        _, _, _, _, _, _, gru_fused, cls_emb = model(
            input_text, input_image, te_input_cluster_ids, im_input_cluster_ids, contextual_rep_t, contextual_rep_v)

        pre_labels = cls_emb.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_label.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_label.size(0)
        train_num += input_label.size(0)

        if modality == 'text':
            context_triplet_loss = triplet_class_loss(gru_fused, args.margin_class, te_input_cluster_ids, input_label, epoch)
            intra_cluster_loss = cluster_intra_inter_distance_loss(gru_fused, te_input_cluster_ids)
        elif modality == 'image':
            context_triplet_loss = triplet_class_loss(gru_fused, args.margin_class, im_input_cluster_ids, input_label, epoch)
            intra_cluster_loss = cluster_intra_inter_distance_loss(gru_fused, im_input_cluster_ids)
        predict_loss = cls_cri(cls_emb, input_label.float())
        if context_triplet_loss is None:
            continue
        train_loss = context_triplet_loss * args.lambda_triplet_class + \
                     intra_cluster_loss * args.lambda_cluster + \
                     predict_loss

        optimi.zero_grad()
        train_loss.backward()
        optimi.step()
        adjust_learning_rate(optimi, epoch)

        context_triplet_all_loss += context_triplet_loss.item() * args.lambda_triplet_class
        intra_cluster_all_loss += intra_cluster_loss.item() * args.lambda_cluster
        predict_all_loss += predict_loss.item()
        train_all_loss += train_loss.item()

    print('train loss {:<5f}, con {:<5f}, intra {:<5f}, pred {:<5f}, acc {:<5f}'.format(
        train_all_loss, context_triplet_all_loss, intra_cluster_all_loss, predict_all_loss, skl_acc/train_num
    ), end='|')

    return train_all_loss / train_num, skl_acc / train_num


def test_model(test_data, model, modality, ti_ids, te_clu_ids, im_clu_ids, cls_cri, args, epoch, un_model):
    model.eval()
    intra_cluster_all_loss, context_triplet_all_loss, predict_all_loss, test_all_loss = 0, 0, 0, 0
    skl_acc = 0
    test_num = 1
    un_model.eval()

    for step, test_da in enumerate(test_data):
        input_text, input_image, input_label, input_ti_id, _ = test_da

        te_input_cluster_ids = np.array([])
        im_input_cluster_ids = np.array([])
        for i in input_ti_id:
            te_input_cluster_ids = np.append(te_input_cluster_ids, te_clu_ids[np.where(ti_ids == i)])
            im_input_cluster_ids = np.append(im_input_cluster_ids, im_clu_ids[np.where(ti_ids == i)])
        te_input_cluster_ids = torch.tensor(te_input_cluster_ids).cuda()
        im_input_cluster_ids = torch.tensor(im_input_cluster_ids).cuda()

        with torch.no_grad():
            contextual_rep_t, contextual_rep_v = un_model(input_text, input_image)
            _, _, _, _, _, _, gru_fused, cls_emb = model(
                input_text, input_image, te_input_cluster_ids, im_input_cluster_ids, contextual_rep_t, contextual_rep_v)

        pre_labels = cls_emb.clone().detach()
        pre_labels[pre_labels > 0.5] = 1
        pre_labels[pre_labels <= 0.5] = 0
        acc = accuracy_score(input_label.detach().cpu(), pre_labels.cpu())
        skl_acc += acc * input_label.size(0)
        test_num += input_label.size(0)

        if modality == 'text':
            context_triplet_loss = triplet_class_loss(gru_fused, args.margin_class, te_input_cluster_ids, input_label, epoch)
            intra_cluster_loss = cluster_intra_inter_distance_loss(gru_fused, te_input_cluster_ids)
        elif modality == 'image':
            context_triplet_loss = triplet_class_loss(gru_fused, args.margin_class, im_input_cluster_ids, input_label, epoch)
            intra_cluster_loss = cluster_intra_inter_distance_loss(gru_fused, im_input_cluster_ids)
        predict_loss = cls_cri(cls_emb, input_label.float())

        if context_triplet_loss is None:
            continue

        test_loss = context_triplet_loss * args.lambda_triplet_class + \
                     intra_cluster_loss * args.lambda_cluster + \
                     predict_loss

        context_triplet_all_loss += context_triplet_loss * args.lambda_triplet_class
        intra_cluster_all_loss += intra_cluster_loss * args.lambda_cluster
        predict_all_loss += predict_loss
        test_all_loss += test_loss

    print('valid loss {:<5f}, con {:<5f}, intra {:<5f}, pred {:<5f}, acc {:<5f}'.format(
        test_all_loss, context_triplet_all_loss, intra_cluster_all_loss, predict_all_loss, skl_acc / test_num
    ))
    return skl_acc/test_num


def unsuper_learning(all_loader, model, ti_ids, te_clu_ids, im_clu_ids, args, options, save_model_path):
    """
    用train数据对triplet cluster loss部分做训练，没有使用谣言真假标签
    """
    model.train()
    optimizers = torch.optim.Adam(model.parameters(), lr=options[args.dataset]['learning_rate'],
                                  weight_decay=float(args.wd))

    best_loss = 1e+5
    counter = 0

    for epoch in range(100):
        text_loss, image_loss, loss_item = 0.0, 0.0, 0.0

        for batch_data in all_loader:
            input_text, input_image, input_label, input_ti_id, _ = batch_data
            if len(input_label) < args.batch_size:
                continue

            te_input_cluster_ids = np.array([])
            im_input_cluster_ids = np.array([])
            for i in input_ti_id:
                te_input_cluster_ids = np.append(te_input_cluster_ids, te_clu_ids[np.where(ti_ids == i)])
                im_input_cluster_ids = np.append(im_input_cluster_ids, im_clu_ids[np.where(ti_ids == i)])
            te_input_cluster_ids = torch.tensor(te_input_cluster_ids).cuda()
            im_input_cluster_ids = torch.tensor(im_input_cluster_ids).cuda()
            if len(te_input_cluster_ids) != len(input_ti_id):
                print(len(te_clu_ids), len(im_clu_ids))
                print(len(te_input_cluster_ids), len(input_ti_id), len(input_text))
                raise ValueError()
            assert len(te_input_cluster_ids) == len(input_ti_id)
            assert len(im_input_cluster_ids) == len(input_ti_id)

            mlp2_t, mlp2_v = model(input_text, input_image)

            sem_tri_loss1 = semantic_triplet_learning(mlp2_t, None, None, args.margin, te_input_cluster_ids, epoch)
            sem_tri_loss2 = semantic_triplet_learning(mlp2_v, None, None, args.margin, im_input_cluster_ids, epoch)
            if sem_tri_loss1 is None:
                continue
            if sem_tri_loss2 is None:
                continue
            loss = sem_tri_loss1 + sem_tri_loss2

            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            adjust_learning_rate(optimizers, epoch)

            text_loss += sem_tri_loss1.item()
            image_loss += sem_tri_loss2.item()
            loss_item += loss.item()

        if loss_item == 0:
            print('     epoch {:<2d}| no valid triplet samples, skip.'.format(epoch))
            continue

        print('     epoch {:<2d}| loss {:<6f}, text {:<6f} image {:<6f} '.format(
            epoch, loss_item, text_loss, image_loss), end='|')

        if epoch < 10:
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, save_model_path)
            print('.')
            continue

        if loss_item < best_loss:
            counter = 0
            best_loss = loss_item
            print('a lower loss, save model to:', save_model_path)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, save_model_path)
        else:
            print('a larger loss, continue.')
            counter += 1

        if counter >= 10:
            print('early stop with patience 10 at epoch ', epoch-10)
            return

    print('      not early stop, finish 100 epoch.')
    print('     pretrained model saved to: ', save_model_path)
    return


def semantic_triplet_learning(t_emb, t_tokens, t_cls, margin, clu_ids, epoch):
    """
    用cluster label指导triplet训练，
    具体而言，batch中cluster内的节点拉近，cluster间的节点拉远。
    :param t_emb:
    :param t_tokens:
    :param t_cls:
    :param margin:
    :return:
    """
    batch_size = t_emb.size(0)
    dist = torch.pow(t_emb, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
    dist = dist + dist.t()
    dist.addmm_(t_emb, t_emb.t(), beta=1, alpha=-2)
    # eps = 1e-4 / t_emb.size(1)
    # dist = (dist + eps).sqrt()
    dist = dist.clamp(min=1e-12).sqrt()

    mask = clu_ids.expand(batch_size, batch_size).eq(clu_ids.expand(batch_size, batch_size).t())
    tri_loss = semi_hard_samples(dist, mask, batch_size, margin, epoch)

    return tri_loss


def semi_hard_samples(dist, mask, batch_size, margin, epoch):
    """
    采样semi-hard neg样本。具体方式为对一个batch中所有可能的a，p，n组合进行判断:
    1. 如果 dist(a, n) - dist(a, p) < 0,  是harder样本，在训练后期处理;
    2. 如果 0 < dist(a,n) - dist(a, p) < margin, 是semi-hard样本，在训练前期处理;
    3. 如果 dist(a,n) - dist(a, p) > margin, 是easy样本，无需处理；
    """
    # sampled_triplet_num = 0
    dist_ap_sub_an = []
    for i in range(batch_size):
        all_neg_dist = dist[i][mask[i] == 0]
        all_pos_dist = dist[i][mask[i]]
        an_sub_ap_dist = all_neg_dist.unsqueeze(1) - all_pos_dist
        if epoch < 10:  # semi-hard negs
            semi_hard_dist = an_sub_ap_dist[((0 < an_sub_ap_dist) & (an_sub_ap_dist < margin))]
        else:  # semi-hard + hardest negs
            semi_hard_dist = an_sub_ap_dist[an_sub_ap_dist < margin]
        # sampled_triplet_num += len(semi_hard_dist)
        if len(semi_hard_dist) == 0:  # 当前数据点没有符合要求的anchor，pos和neg三元组样本组合
            continue
        # dist_ap_sub_an.append(semi_hard_dist)
        diss_diff = torch.clamp(-1.0 * semi_hard_dist + margin, min=0.0)
        dist_ap_sub_an.append(torch.mean(diss_diff).unsqueeze(0))
    if len(dist_ap_sub_an) == 0:  # 当前batch中没有符合要求的anchor，pos和neg三元组样本组合
        return None
    dist_ap_sub_an = torch.cat(dist_ap_sub_an)
    loss1 = torch.mean(dist_ap_sub_an)
    # if epoch > 10:
    #     print('    sampled triplet num is :',  sampled_triplet_num)
    return loss1


def triplet_class_loss(t_emb, margin, clu_ids, labels, epoch):
    """
    使用三元组学习区分cluster内的real 和 fake 样本；
    """
    batch_size = t_emb.size(0)
    single_label = labels[:, 0]

    dist = torch.pow(t_emb, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
    dist = dist + dist.t()
    dist.addmm_(t_emb, t_emb.t(), beta=1, alpha=-2)
    # eps = 1e-4 / t_emb.size(1)
    # dist = (dist + eps).sqrt()
    dist = dist.clamp(min=1e-12).sqrt()

    cluster_mask = clu_ids.expand(batch_size, batch_size).eq(clu_ids.expand(batch_size, batch_size).t())
    rumor_mask = single_label.expand(batch_size, batch_size).eq(single_label.expand(batch_size, batch_size).t())

    diss_diff = []
    for i in range(batch_size):
        # the pos in same cluster
        dist_pos = dist[i][cluster_mask[i] & rumor_mask[i]]
        if len(dist_pos) == 0:
            continue
        # the neg in same cluster
        dist_neg = dist[i][cluster_mask[i] & (~ rumor_mask[i])]
        if len(dist_neg) == 0:
            continue
        # select the semi_hard + harder apn triplets
        dist_an_sub_ap = dist_neg.unsqueeze(1) - dist_pos
        semi_harder_points = dist_an_sub_ap[dist_an_sub_ap < margin]
        if len(semi_harder_points) == 0:
            continue
        diss_max = torch.clamp(-1.0 * semi_harder_points + margin, min=0.0)
        diss_diff.append(torch.mean(diss_max).unsqueeze(0))
    if len(diss_diff) == 0:
        return None
    diss_diff = torch.cat(diss_diff)
    return torch.mean(diss_diff)


def cluster_intra_inter_distance_loss(t_emb, clu_ids):
    """
    这个loss用于计算类内距离和类间距离，保持类关系明确
    """
    batch_size = t_emb.size(0)
    cluster_mask = clu_ids.expand(batch_size, batch_size).eq(clu_ids.expand(batch_size, batch_size).t())

    dist = torch.pow(t_emb, 2).sum(dim=1, keepdim=True).expand(batch_size, batch_size)
    dist = dist + dist.t()
    dist.addmm_(t_emb, t_emb.t(), beta=1, alpha=-2)
    # eps = 1e-4 / t_emb.size(1)
    # dist = (dist + eps).sqrt()
    dist = dist.clamp(min=1e-12).sqrt()

    intra_dis = []
    for i in range(batch_size):
        one_intra_dis = torch.mean(dist[i][cluster_mask[i]])
        intra_dis.append(one_intra_dis.unsqueeze(0))
    intra_dis = torch.cat(intra_dis)
    # dist.masked_fill_(cluster_mask == 0, torch.tensor(0))
    # intra_dis = torch.mean(dist, dim=-1)
    # intra_loss = torch.mean(intra_dis)
    return torch.mean(intra_dis)


def adjust_learning_rate(optimizer, epoch):
    """
    Updates the learning rate given the learning rate decay.
    """
    if epoch > 20:
        for group in optimizer.param_groups:
            group['lr'] = 0.5 * group['lr']
    # if epoch < 30:
    #     for group in optimizer.param_groups:
    #         group['lr'] = 1e-4
    #         # if 'step' not in group:
    #         #     group['step'] = 0
    #         # group['step'] += 1
    #
    #         # group['lr'] = args.lr / (1 + group['step'] * args.lr_decay)
    # elif epoch < 60:
    #     for group in optimizer.param_groups:
    #         group['lr'] = 1e-5
    # else:
    #     for group in optimizer.param_groups:
    #         group['lr'] = 1e-6
