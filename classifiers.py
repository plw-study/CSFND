import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix


def multi_classifiers(train_embs, test_embs, cluster_num, svm_type='linear', verbose=False):
    train_hidden_embs, train_label_embs, train_cls_embs, train_cluster_labels = train_embs
    test_hidden_embs, test_label_embs, test_cls_embs, test_cluster_labels = test_embs
    input_dim = train_hidden_embs.shape[1]

    tn_list, tp_list, fn_list, fp_list = [], [], [], []
    real_tn_list, real_tp_list, real_fn_list, real_fp_list = [], [], [], []

    for cluster_i in range(cluster_num):
        if verbose:
            print('--- clu {:2d}'.format(cluster_i), end='|')
        tr_mask = np.where(train_cluster_labels == cluster_i)
        te_mask = np.where(test_cluster_labels == cluster_i)
        tr_hid_emb = train_hidden_embs[tr_mask]
        te_hid_emb = test_hidden_embs[te_mask]

        if len(te_hid_emb) > 0:  # 有测试数据，就需要有测试结果

            if len(tr_hid_emb) > 0:  # 有训练数据，就需要训练一个二分类器模型

                tr_label = train_label_embs[tr_mask]
                te_label = test_label_embs[te_mask]
                if verbose:
                    print('{:<4d} fake, {:<4d} real in {:<4d} train| {:<4d} fake, {:<4d} real in {:<4d} test'.format(
                        np.sum(tr_label[:, 0] == 0), np.sum(tr_label[:, 0] == 1), len(tr_label),
                        np.sum(te_label[:, 0] == 0), np.sum(te_label[:, 0] == 1), len(te_label)), end='\n')

                if svm_type == 'mlp':
                    y_pre, y_true = mlp(tr_hid_emb, tr_label, te_hid_emb, te_label, input_dim)
                    y_pre = y_pre[:, 0]
                    te_label = y_true[:, 0]
                else:
                    tr_label = train_label_embs[tr_mask][:, 0]
                    te_label = test_label_embs[te_mask][:, 0]
                    if svm_type == 'bnb':
                        y_pre = naive_bayes(tr_hid_emb, tr_label, te_hid_emb)
                    elif svm_type == 'knn':
                        y_pre = knn(tr_hid_emb, tr_label, te_hid_emb)
                    else:
                        if (0 in tr_label) and (1 in tr_label):
                            y_pre = svm(tr_hid_emb, tr_label, te_hid_emb, svm_type)
                        else:
                            # 训练集中所有数据都是同一个类
                            y_pre = np.array([tr_label[0] for i in te_label])

            else:  # 没有训练数据，用原模型的最后一层mlp的2位输出结果作为detection结果
                y_pre = test_cls_embs[te_mask]
                y_pre[y_pre > 0.5] = 1
                y_pre[y_pre <= 0.5] = 0
                y_pre = y_pre[:, 0]
                te_label = test_label_embs[te_mask][:, 0]
                if verbose:
                    print('{:<4d} fake {:<4d} real in {:<4d} train| {:<4d} fake {:<4d} real in {:<4d} test'.format(
                        0, 0, 0,
                        np.sum(te_label == 0), np.sum(te_label == 1), len(te_label)), end='\n')

            # 1: positive => real pre, recall, f1
            real_tn, real_fp, real_fn, real_tp = confusion_matrix(te_label, y_pre, labels=[0, 1]).ravel()
            # 0: positive => fake pre, recall, f1
            tn, fp, fn, tp = confusion_matrix(te_label, y_pre, labels=[1, 0]).ravel()
            # pres, recalls, f1s, supports = precision_recall_fscore_support(
            #     te_label, y_pre, labels=[0, 1], average=None)
            if verbose:
                print('          fake pos:tn {:<3d} fp {:<3d} fn {:<3d} tp {:<3d}'.format(tn, fp, fn, tp), end='|')
                print('real pos:tn {:<3d} fp {:<3d} fn {:<3d} tp {:<3d}'.format(real_tn, real_fp, real_fn, real_tp), end='|')

            tn_list.append(tn)
            fp_list.append(fp)
            fn_list.append(fn)
            tp_list.append(tp)

            real_tn_list.append(real_tn)
            real_fp_list.append(real_fp)
            real_fn_list.append(real_fn)
            real_tp_list.append(real_tp)

            # if tp == 0:
            #     pre = 0
            #     recall = 0
            #     f1 = 0
            # else:
            #     pre = tp*1.0/(tp+fp)
            #     recall = tp*1.0/(tp+fn)
            #     f1 = 2 * pre * recall / (pre + recall)
            # acc = (tp+tn)*1.0/(tn+tp+fn+fp)
            pre, recall, f1, acc = pre_recall_f1_compute(tn, fp, fn, tp)
            real_pre, real_recall, real_f1, real_acc = pre_recall_f1_compute(real_tn, real_fp, real_fn, real_tp)

            if verbose:
                print('fake pre {:<4f} recal {:<4f} f1 {:<4f} acc {:<4f}'.format(
                    pre, recall, f1, acc), end='|')
                print('real pre {:<4f} recal {:<4f} f1 {:<4f} acc {:<4f}'.format(
                    real_pre, real_recall, real_f1, real_acc), end='\n')

        else:
            if verbose:
                print('This cluster has no test data.')
            # 训练数据有，测试数据没有，不用处理
            continue

    overall_tn = np.array(tn_list).sum()
    overall_fp = np.array(fp_list).sum()
    overall_fn = np.array(fn_list).sum()
    overall_tp = np.array(tp_list).sum()

    real_overall_tn = np.array(real_tn_list).sum()
    real_overall_fp = np.array(real_fp_list).sum()
    real_overall_fn = np.array(real_fn_list).sum()
    real_overall_tp = np.array(real_tp_list).sum()

    overall_pre, overall_recall, overall_f1, overall_acc = pre_recall_f1_compute(
        overall_tn, overall_fp, overall_fn, overall_tp)
    print('--- overall fake pre {:<4f} recal {:<4f} f1 {:<4f} acc {:<4f}'.format(
        overall_pre, overall_recall, overall_f1, overall_acc), end='|')

    real_overall_pre, real_overall_recall, real_overall_f1, real_overall_acc = pre_recall_f1_compute(
        real_overall_tn, real_overall_fp, real_overall_fn, real_overall_tp)
    print('overall real pre {:<4f} recal {:<4f} f1 {:<4f} acc {:<4f}|{:<10s}'.format(
        real_overall_pre, real_overall_recall, real_overall_f1, real_overall_acc, svm_type), end='\n')

    return (overall_pre, overall_recall, overall_f1, overall_acc), \
           (real_overall_pre, real_overall_recall, real_overall_f1, real_overall_acc)


def pre_recall_f1_compute(tn, fp, fn, tp):
    if tp == 0:
        pre = 0
        recall = 0
        f1 = 0
    else:
        pre = tp * 1.0 / (tp + fp)
        recall = tp * 1.0 / (tp + fn)
        f1 = 2 * pre * recall / (pre + recall)
    acc = (tp + tn) * 1.0 / (tn + tp + fn + fp)

    return pre, recall, f1, acc


def svm(train_data, train_label, test_data, svm_type):
    if svm_type not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
        raise ValueError('ERROR! SVM type is invalid.')
    svm_cls = SVC(kernel=svm_type, max_iter=-1)
    svm_cls.fit(train_data, train_label)
    predicts = svm_cls.predict(test_data)
    # print(predicts)
    return predicts


def mlp(train_data, train_label, test_data, test_label, input_dim):
    train_data = PreData(train_data, train_label)
    test_data = PreData(test_data, test_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=64,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=64,
                                               shuffle=True)

    mlp1 = MLPClassifier(input_dim, 2).cuda()  # 70*768
    best_test_acc, pre_labels, ground_labels = mlp1.run(301, train_loader, test_loader)
    return pre_labels, ground_labels


def knn(train_data, train_label, test_data):
    knn_cls = KNeighborsClassifier(n_neighbors=2)
    knn_cls.fit(train_data, train_label)
    return knn_cls.predict(test_data)


def naive_bayes(train_data, train_label, test_data, nb_type='multinomial'):
    nb_cls = BernoulliNB()
    nb_cls.fit(train_data, train_label)
    return nb_cls.predict(test_data)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLPClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=True)
        # self.linear2 = nn.Linear(hidden_dim, 2, bias=True)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.001)  # params for best weibo
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.0001)

    def acc(self, y_true, y_pre):
        # y_pre = torch.softmax(y_pre, dim=1)
        y_pre[y_pre > 0.5] = 1
        y_pre[y_pre <= 0.5] = 0
        acc = accuracy_score(y_true, y_pre)
        return acc

    def train_model(self, train_data):
        self.train()
        train_loss = 0.0
        train_acc = 0.0
        train_num = 0
        for tr_da, tr_la in train_data:
            train_num += len(tr_la)
            # y = torch.tanh(self.linear1(tr_da))
            y = torch.softmax(torch.tanh(self.linear1(tr_da)), dim=1)
            # y = torch.softmax(self.linear2(y), dim=1)
            loss = self.criterion(y, tr_la.float())
            train_acc += self.acc(tr_la.clone().detach().cpu(), y.clone().detach().cpu()) * len(tr_la)
            train_loss += loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return train_loss, train_acc/train_num

    def valid(self, test_data):
        self.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_num = 0
        pre_labels = []
        ground_labels = []
        for te_da, te_la in test_data:
            test_num += len(te_la)
            # y = torch.tanh(self.linear1(te_da))
            y = torch.softmax(torch.tanh(self.linear1(te_da)), dim=1)
            # y = torch.softmax(self.linear2(y), dim=1)
            loss = self.criterion(y, te_la)
            test_acc += self.acc(te_la.clone().detach().cpu(), y.clone().detach().cpu()) * len(te_la)
            test_loss += loss
            pre_labels.extend(y.clone().detach().cpu().numpy())
            ground_labels.extend(te_la.clone().detach().cpu().numpy())
        pre_labels = np.array(pre_labels)
        ground_labels = np.array(ground_labels)
        return test_loss, test_acc/test_num, pre_labels, ground_labels

    def run(self, epoch, train_data, test_data):
        best_test_acc = -1
        best_pre_labels = []
        accor_ground_labels = []

        for ep in range(epoch):
            train_loss, train_acc = self.train_model(train_data)
            test_loss, test_acc, pre_labels, ground_labels = self.valid(test_data)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_pre_labels = pre_labels
                accor_ground_labels = ground_labels
            # if ep % 20 == 0:
            #     print('   Epoch {:<3} | train loss {:<4f} acc {:<4f}, test loss {:<4f} acc {:<4f}, '.format(
            #         ep, train_loss, train_acc, test_loss, test_acc))
        best_pre_labels[best_pre_labels > 0.5] = 1
        best_pre_labels[best_pre_labels <= 0.5] = 0
        return best_test_acc, best_pre_labels, accor_ground_labels


class PreData(data.Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels
        self.length = len(labels)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        one_datas = torch.tensor(self.datas[item])
        one_labels = torch.FloatTensor(self.labels[item])
        if torch.cuda.is_available():
            one_datas = one_datas.cuda()
            one_labels = one_labels.cuda()
        return one_datas, one_labels