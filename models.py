
import torch

import torch.nn as nn



def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.05, 0.05)
    elif isinstance(m, nn.LSTM):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                i, f, g, o = p.chunk(4)
                nn.init.xavier_uniform_(i)
                nn.init.xavier_uniform_(f)
                nn.init.xavier_uniform_(g)
                nn.init.xavier_uniform_(o)
            elif 'weight_hh' in n:
                i, f, g, o = p.chunk(4)
                nn.init.orthogonal_(i)
                nn.init.orthogonal_(f)
                nn.init.orthogonal_(g)
                nn.init.orthogonal_(o)
            elif 'bias' in n:
                i, f, g, o = p.chunk(4)
                nn.init.zeros_(i)
                nn.init.ones_(f)
                nn.init.zeros_(g)
                nn.init.zeros_(o)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        try:
            nn.init.zeros_(m.bias)
        except AttributeError:
            pass




class SimpleNonLinear(nn.Module):
    def __init__(self, params):
        super().__init__()

        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']

        self.layer_1 = nn.Linear(input_dim, 128)
        self.layer_2 = nn.Linear(128, 64)
        self.layer_3 = nn.Linear(64, 32)
        # self.layer_4 = nn.Linear(64, 32)
        self.layer_out = nn.Linear(32, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(32)
        # self.batchnorm4 = nn.BatchNorm1d(32)


    def forward(self, params):
        x = params['input']


        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # x = self.layer_4(x)
        # x = self.batchnorm4(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        # x = self.layer_2(x)
        # # x = self.batchnorm2(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        #
        # x = self.layer_3(x)
        # # x = self.batchnorm3(x)
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.layer_out(x)

        output = {
            'prediction': x,
            'adv_output': None,
            'hidden': x,  # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
            # 'second_adv_output': second_adv_output

        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.layer_1, self.layer_2, self.layer_3, self.layer_out])

    def reset(self):
        self.layer_1.apply(initialize_parameters)  # don't know, if this is needed.
        self.layer_2.apply(initialize_parameters)  # don't know, if this is needed.
        self.layer_3.apply(initialize_parameters)  # don't know, if this is needed.
        # self.layer_4.apply(initialize_parameters)  # don't know, if this is needed.
        # self.layer_5.apply(initialize_parameters)  # don't know, if this is needed.
        # self.layer_6.apply(initialize_parameters)  # don't know, if this is needed.
        self.layer_out.apply(initialize_parameters)  # don't know, if this is needed.

class SimpleLinear(nn.Module):
    def __init__(self, params):
        super().__init__()
        input_dim = params['model_arch']['encoder']['input_dim']
        output_dim = params['model_arch']['encoder']['output_dim']
        self.encoder = nn.Linear(input_dim, output_dim)
        self.encoder.apply(initialize_parameters)

    def forward(self, params):
        text = params['input']
        prediction = self.encoder(text)

        output = {
            'prediction': prediction,
            'adv_output': None,
            'hidden': prediction, # just for compatabilit
            'classifier_hiddens': None,
            'adv_hiddens': None
            # 'second_adv_output': second_adv_output

        }

        return output

    @property
    def layers(self):
        return torch.nn.ModuleList([self.encoder])

    def reset(self):
        self.encoder.apply(initialize_parameters)  # don't know, if this is needed.