import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from transformers import BertModel, BertTokenizer


class TextModel(nn.Module):
    def __init__(self, model_name, if_training=False):
        super(TextModel, self).__init__()
        self.textModel = BertModel.from_pretrained(model_name, local_files_only=True)
        self.if_training = if_training

        if self.if_training:
            self.train()
        else:
            self.eval()

    def forward(self, input_text):
        # input_text: [batch_size, token_num, dim]
        last_hidden_states = self.textModel(input_text)[0]
        cls_emb = last_hidden_states[:, 0, :]  # cls_emb: [batch_size, dim]
        # remove [CLS] and [SEP] tag
        tokens_emb = last_hidden_states[:, 1:-1, :]  # tokens_emb: [batch_size, MAX_LENGTH, dim]
        return cls_emb, tokens_emb


class ImageModel(nn.Module):
    def __init__(self, model_name, if_training=False):
        super(ImageModel, self).__init__()
        if model_name == 'vgg19':
            # self.imageModel = models.vgg19(pretrained=True)
            pre_file = torch.load('/home/hibird/plw/models/vgg19-dcbb9e9d.pth')
            self.imageModel = models.vgg19(pretrained=False)
            self.imageModel.load_state_dict(pre_file)

            # new_classifier = torch.nn.Sequential(*list(self.imageModel.children())[-1][:6])
            new_classifier = self.imageModel.classifier[:6]
            self.imageModel.classifier = new_classifier
        else:
            raise ValueError('Error! ImageModel supports only vgg19.')

        if if_training:
            self.train()
        else:
            self.eval()

    def forward(self, input_image):
        """
        :param input_image: [batch_size, ?] after Image and transforms
        :return:
        """
        return self.imageModel(input_image).data


def load_pre_models(model_name, pretraining, modality):
    if modality == 'text':
        model = TextModel(model_name, pretraining)
    elif modality == 'image':
        model = ImageModel(model_name, pretraining)
    else:
        raise ValueError('ERROR! modality must be text or image.')
    if torch.cuda.is_available():
        model = model.cuda()
    return model
