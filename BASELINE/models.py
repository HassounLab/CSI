# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)



import sys
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import dgl
#from torch.profiler import profile, record_function, ProfilerActivity

from dgllife.model import GCN, GAT
from utils import array_to_bitstring

class PROT_CNN(nn.Module):
    def __init__(self, hp, input_dim):
        super(PROT_CNN, self).__init__()
        self.embed = nn.Embedding(input_dim, hp['prot_embedding_dim'], padding_idx=input_dim - 1)
        self.conv_prot = nn.Conv1d(in_channels=hp['cnn_prot_length']+2, out_channels=hp['cnn_out_channels'], 
                                   kernel_size=hp['cnn_kernel_size'])
        self.out_channels = hp['cnn_out_channels']
        self.fc_prot = nn.Linear(93*self.out_channels, hp['prot_embedding_dim'])
        self.fc_prot_embed = nn.Linear(93, hp['prot_embedding_dim'])
        
    def forward(self, x, lengths):
        h = self.embed(x)
        h = self.conv_prot(h)
        h = h.view(-1, self.out_channels*93)
        # final projection
        z = self.fc_prot(h)

        return h, z

    def load_weights(self, pretrained_model):
        # load pretrained model weights
        state_dict = self.state_dict()
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): key = key[7:]
            if key in state_dict and value.shape == state_dict[key].shape: state_dict[key] = value
        self.load_state_dict(state_dict)
        
    def em(self, h, lengths, cpu=False):
        # get representations with different lengths from the collated single matrix
        e = [None] * len(lengths)
        for i in range(len(lengths)):
            if cpu: e[i] = h[i].cpu()
            else:   e[i] = h[i]
        return e


class MOL_GNN2(nn.Module):
    def __init__(self, hp, in_dim):
        super(MOL_GNN2, self).__init__()
        dropout = [hp['gnn_dropout'] for _ in range(len(hp['gnn_channels']))]
        batchnorm = [True for _ in range(len(hp['gnn_channels']))]
        gnn_map = {
            "gcn": GCN(in_dim, hp['gnn_channels'], batchnorm = batchnorm, dropout = dropout),
            "gat": GAT(in_dim, hp['gnn_channels'], hp['attn_heads'])
        }
        self.GNN = gnn_map[hp['gnn_type']]
        self.pool = dgl.nn.pytorch.glob.MaxPooling()
        self.fc1_graph = nn.Linear(hp['gnn_channels'][len(hp['gnn_channels']) - 1], hp['gnn_hidden_dim'] * 2)
        self.fc2_graph = nn.Linear(hp['gnn_hidden_dim'] * 2, hp['prot_embedding_dim'])
            
        self.fc1_prot = nn.Linear(hp['prot_embedding_dim'], hp['gnn_hidden_dim'] * 2)
        self.fc2_prot = nn.Linear(hp['gnn_hidden_dim'] * 2, hp['prot_embedding_dim'])

        self.fc_comp = nn.Linear(hp['gnn_channels'][len(hp['gnn_channels']) - 1], hp['prot_embedding_dim'])
        self.dropout = nn.Dropout(hp['fc_dropout'])
        self.relu = nn.ReLU()
        self.W_out1 = nn.Linear(2 * hp['prot_embedding_dim'], hp['gnn_hidden_dim'] * 2)
        self.W_out2 = nn.Linear(hp['gnn_hidden_dim'] * 2, hp['gnn_hidden_dim'])
        self.W_out3 = nn.Linear(hp['gnn_hidden_dim'], hp['num_classes'])
        self.embedding_dim = hp['prot_embedding_dim']
        self.out_channels = hp['cnn_out_channels']
        self.fc_prot = nn.Linear(hp['prot_embedding_dim'], hp['prot_embedding_dim'])
        self.prot_pool = F.max_pool2d
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W_attention_prot = nn.Linear(hp['prot_embedding_dim'], hp['prot_embedding_dim'])
        self.W_attention_comp = nn.Linear(hp['prot_embedding_dim'], hp['prot_embedding_dim'])
        
    def graphembedding(self, g, f):
        f = self.GNN(g, f)
        h = self.pool(g, f)
        return h, f
    
    #@profile
    def cross_attention(self, comp_pooled, comp_gnn, comp_graph, prot_pooled, prot_list):
        return comp_pooled, prot_pooled
                

    def forward(self, g, f, prot_bat, prot_list, c1=None, c2=None):
        comp_pooled, f = self.graphembedding(g, f)
            
        prot_bat_final = prot_bat

        comp, prot_bat_final = self.cross_attention(comp_pooled, f, g, prot_bat_final, prot_list)
        h = self.relu(self.fc1_graph(comp))
        h = self.dropout(h)
        h = self.fc2_graph(h)
        comp = self.dropout(h)

        y_cat = torch.cat((comp, prot_bat_final), 1) 
            
        h = self.W_out1(y_cat)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.W_out2(h)
        h = self.relu(h)
        h = self.dropout(h)
        
        z_interaction = self.W_out3(h)


        return z_interaction, comp, prot_bat_final

    def load_weights(self, pretrained_model):
        # load pretrained_model weights
        state_dict = {}
        loaded_dict = torch.load(pretrained_model, map_location=torch.device('cpu'))
        if loaded_dict.get('model_state_dict', None) is not None: #hack for dgl saved models
            loaded_dict = loaded_dict['model_state_dict']
        for key, value in loaded_dict.items():
            if key.startswith("module"): state_dict[key[7:]] = value
            else: state_dict[key] = value
        self.load_state_dict(state_dict)
