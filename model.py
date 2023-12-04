import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pre_models import TextModel, ImageModel
from attention import NetShareFusion, PerceptronAttention


class HiModel(nn.Module):
    def __init__(self, dataset, opt, args):
        super(HiModel, self).__init__()
        self.opt = opt
        self.args = args
        self.class_num = 2
        self.text_if_training = self.opt[dataset]['fine_tune_bert']
        self.image_if_training = self.opt[dataset]['fine_tune_vgg']
        self.textModel = TextModel(self.opt[dataset]['bert_model'], if_training=self.opt[dataset]['fine_tune_bert'])
        self.imageModel = ImageModel(self.opt[dataset]['vgg_model'], if_training=self.opt[dataset]['fine_tune_vgg'])

        self.attention_t = PerceptronAttention(self.opt['out_dim_t'], self.opt['out_dim_t'])
        self.attention_v = PerceptronAttention(self.opt['out_dim_v'], self.opt['out_dim_v'])

        self.gru_t = nn.GRUCell(self.opt['out_dim_t'], self.opt['mid_dim_t'])
        self.gru_v = nn.GRUCell(self.opt['out_dim_v'], self.opt['mid_dim_v'])
        self.cross_attention = NetShareFusion(model_dim=self.opt['mid_dim_v'])

        # Linear for semantic emb
        self.linear_t2 = nn.Linear(self.opt['flat_dim_t'], self.opt['mid_dim_t'])
        self.bn_layer_t2 = nn.BatchNorm1d(self.opt['mid_dim_t'], affine=True, track_running_stats=True)
        self.linear_v2 = nn.Linear(self.opt['flat_dim_v'], self.opt['mid_dim_v'])
        self.bn_layer_v2 = nn.BatchNorm1d(self.opt['mid_dim_v'], affine=True, track_running_stats=True)

        # Classifier
        self.cls_linear_fused = nn.Linear(self.opt['mid_dim_t'], self.class_num)

    def attention_cal(self, tri_feats, emb_feats, clu_ids, modal):
        """
        对于每一个数据，用他同cluster的16个其他数据的features计算他们的关系，也就是attention值
        不足16个的用数据0补齐，超过的随机舍弃或者去掉最远的。
        :param tri_feats: [batch_size, tri_dim] cluster context embs
        :param emb_feats: [batch_size, hid_dim] use cred emb to weight values
        :param clu_ids: [batch_size, 1] indicating the cluster number the datas belonging to.
        :return:
        """
        batch_size = tri_feats.size(0)
        nei_number = int(math.sqrt(self.args.batch_size))

        mask = clu_ids.expand(batch_size, batch_size).eq(clu_ids.expand(batch_size, batch_size).t())
        eye_mask = torch.eye(batch_size) > .5
        if torch.cuda.is_available():
            eye_mask = eye_mask.cuda()
        mask = mask.masked_fill_(eye_mask, 0)

        select_nei_features = torch.zeros(batch_size, nei_number, tri_feats.size(1)).cuda()  # [batch_size, nei_number, feature_dim]
        for i in range(batch_size):
            nei_features = tri_feats[mask[i], :]
            nei_count = len(nei_features)
            if nei_count <= nei_number:
                select_nei_features[i, :nei_count, :] = nei_features
            else:
                # 超出nei_number个同类数据，随机选择若干数据
                random_nei = np.random.choice(nei_count, nei_number, replace=False)
                select_nei_features[i, :, :] = nei_features[random_nei, :]

        if modal == 'text':
            attention_value = self.attention_t(tri_feats, select_nei_features)  # [batch_size, nei_number]
        elif modal == 'image':
            attention_value = self.attention_v(tri_feats, select_nei_features)
        else:
            raise ValueError('Error! Modal for attention must be text or image.')
        clu_features = torch.bmm(attention_value.unsqueeze(1), select_nei_features).squeeze()  # [batch_size, out_feature]
        return clu_features, attention_value

    def forward(self, input_t, input_v, te_clu_ids, im_clu_ids, t_tri_en, v_tri_en):
        batch_size_t = input_t.size(0)
        if self.text_if_training:
            _, input_t = self.textModel(input_t)
        else:
            with torch.no_grad():
                _, input_t = self.textModel(input_t)  # t_pre_token: [batch_size, MAX_LENGTH, dim]
        input_t = torch.reshape(input_t, (batch_size_t, -1))  # [batch_size, MAX_LENGTH * dim]

        if self.image_if_training:
            input_v = self.imageModel(input_v)
        else:
            with torch.no_grad():
                input_v = self.imageModel(input_v)

        te_clu_emb, _ = self.attention_cal(t_tri_en, None, te_clu_ids, 'text')
        im_clu_emb, _ = self.attention_cal(v_tri_en, None, im_clu_ids, 'image')

        input_t = self.linear_t2(input_t)
        input_t = self.bn_layer_t2(input_t)
        input_t = F.relu(input_t)

        input_v = self.linear_v2(input_v)
        input_v = self.bn_layer_v2(input_v)
        input_v = F.relu(input_v)

        gru_out_t = self.gru_t(te_clu_emb, input_t)
        gru_out_v = self.gru_v(im_clu_emb, input_v)

        mm_fused = self.cross_attention(gru_out_v, gru_out_t, attn_mask=None)

        cls_emb = self.cls_linear_fused(mm_fused)
        cls_emb = F.softmax(cls_emb, dim=1)

        return t_tri_en, v_tri_en, input_t, input_v, gru_out_t, gru_out_v, mm_fused, cls_emb


class UnsupModel(nn.Module):
    def __init__(self, dataset, opt, args):
        super(UnsupModel, self).__init__()
        self.opt = opt
        self.args = args
        self.class_num = 2
        self.text_if_training = self.opt[dataset]['fine_tune_bert']
        self.image_if_training = self.opt[dataset]['fine_tune_vgg']
        self.textModel = TextModel(self.opt[dataset]['bert_model'], if_training=self.opt[dataset]['fine_tune_bert'])
        self.imageModel = ImageModel(self.opt[dataset]['vgg_model'], if_training=self.opt[dataset]['fine_tune_vgg'])

        self.linear_t = nn.Linear(self.opt['bert_dim_t'], self.opt['hidden_dim_t'])
        self.bn_layer_t = nn.BatchNorm1d(self.opt['hidden_dim_t'], affine=True, track_running_stats=True)
        self.linear_v = nn.Linear(self.opt['flat_dim_v'], self.opt['hidden_dim_v'])
        self.bn_layer_v = nn.BatchNorm1d(self.opt['hidden_dim_v'], affine=True, track_running_stats=True)

        self.linear_t1 = nn.Linear(self.opt['hidden_dim_t'], self.opt['out_dim_t'])
        self.bn_layer_t1 = nn.BatchNorm1d(self.opt['out_dim_t'], affine=True, track_running_stats=True)
        self.linear_v1 = nn.Linear(self.opt['hidden_dim_v'], self.opt['out_dim_v'])
        self.bn_layer_v1 = nn.BatchNorm1d(self.opt['out_dim_v'], affine=True, track_running_stats=True)

    def triplet_encoder(self, en_t):
        en_t = self.linear_t(en_t)
        en_t = self.bn_layer_t(en_t)
        en_t = F.relu(en_t)

        en_t = self.linear_t1(en_t)
        en_t = self.bn_layer_t1(en_t)
        en_t = F.relu(en_t)

        return en_t

    def triplet_encoder_image(self, en_v):
        en_v = self.linear_v(en_v)
        en_v = self.bn_layer_v(en_v)
        en_v = F.relu(en_v)

        en_v = self.linear_v1(en_v)
        en_v = self.bn_layer_v1(en_v)
        en_v = F.relu(en_v)

        return en_v

    def forward(self, input_t, input_v):
        batch_size_t = input_t.size(0)
        if self.text_if_training:
            _, input_t = self.textModel(input_t)
        else:
            with torch.no_grad():
                _, input_t = self.textModel(input_t)
        # t_pre_mean = torch.mean(t_pre_token, dim=1)  # [batch_size, dim]
        input_t = torch.sum(input_t, dim=1)  # [batch_size, dim]
        # t_pre_flat = torch.reshape(t_pre_token, (batch_size_t, -1))  # [batch_size, MAX_LENGTH * dim]

        if self.image_if_training:
            input_v = self.imageModel(input_v)
        else:
            with torch.no_grad():
                input_v = self.imageModel(input_v)

        input_t = self.triplet_encoder(input_t)
        input_v = self.triplet_encoder_image(input_v)

        return input_t, input_v


def load_main_model(dataset, options, args):
    model = HiModel(dataset, options, args)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_unsup_model(dataset, options, args):
    model = UnsupModel(dataset, options, args)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

