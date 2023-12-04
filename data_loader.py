import os
import torch
import torch.utils.data as data
from text_process import dataset_filter
from transformers import BertTokenizer


class PreDataset(data.Dataset):
    def __init__(self, dataset, data_type, options):
        self.data_type = data_type
        print('===> process {} data...'.format(dataset))
        if not os.path.exists('./data_files/{}'.format(dataset, data_type)):
            os.makedirs('./data_files/{}'.format(dataset))

        if not os.path.exists('./data_files/{}/{}_text_ids.pt'.format(dataset, data_type)):
            self.text_lists, self.image_lists, self.labels, self.text_image_ids = dataset_filter(dataset, data_type)
            tokenizer = BertTokenizer.from_pretrained(options[dataset]['bert_model'], local_files_only=True)
            encodes = tokenizer(self.text_lists, padding='max_length', truncation=True,
                                max_length=options[dataset]['token_length']+2,
                                add_special_tokens=True)
            self.text_ids = torch.tensor(encodes["input_ids"])
            self.length = len(self.labels)
            self.labels = torch.tensor(self.labels)

            torch.save(self.text_ids, './data_files/{}/{}_text_ids.pt'.format(dataset, data_type))
            torch.save(self.image_lists, './data_files/{}/{}_image_lists.pt'.format(dataset, data_type))
            torch.save(self.labels, './data_files/{}/{}_labels.pt'.format(dataset, data_type))
            torch.save(self.text_image_ids, './data_files/{}/{}_text_image_ids.pt'.format(dataset, data_type))
        else:
            self.text_ids = torch.load('./data_files/{}/{}_text_ids.pt'.format(dataset, data_type))
            self.image_lists = torch.load('./data_files/{}/{}_image_lists.pt'.format(dataset, data_type))
            self.labels = torch.load('./data_files/{}/{}_labels.pt'.format(dataset, data_type))
            self.text_image_ids = torch.load('./data_files/{}/{}_text_image_ids.pt'.format(dataset, data_type))
            self.length = len(self.labels)

        fake_news = (self.labels[:, 0] == 0).sum()
        real_news = (self.labels[:, 0] == 1).sum()
        print('    load {} {:<5} {} data from files,'.format(dataset, self.length, data_type), end='')
        print(' include {} fake and {} real news.'.format(fake_news.item(), real_news.item()))

    def __getitem__(self, item):
        text_tensor = self.text_ids[item]  # int tensor
        image_tensor = self.image_lists[item]  # float tensor
        label_tensor = self.labels[item]  # int
        ti_id_tensor = self.text_image_ids[item]  # str

        if torch.cuda.is_available():
            text_tensor = text_tensor.cuda()
            image_tensor = image_tensor.cuda()
            label_tensor = label_tensor.cuda()

        return text_tensor, image_tensor, label_tensor, ti_id_tensor, self.data_type

    def __len__(self):
        return self.length


def load_data(dataset, batch_size, options):

    train_data = PreDataset(dataset, 'train', options)
    test_data = PreDataset(dataset, 'test', options)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=batch_size,
                                               shuffle=False)

    return train_loader, test_loader, test_loader






