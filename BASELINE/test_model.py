# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)

import torch
import pandas as pd
import numpy as np
import pickle
from dataset import create_train_val_dataset, create_test_dataset, collate_gnn_cpi
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgllife.utils import EarlyStopping
from tqdm import tqdm
from torch.distributions import Normal
import yaml
import os
import time
import matplotlib.pyplot as plt
from utils import Print, load_models, get_cindex, create_posneg_distrib, report_metric
from dataset import Protein
from models import MOL_GNN2, PROT_CNN
from sklearn.metrics import auc, roc_auc_score, r2_score, average_precision_score
from collections import defaultdict

def test(prot_model, mol_model, model_name, hp, output, **kwargs):
    for key, value in kwargs.items():
        hp[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)

    if os.path.exists(hp['data_dir']+"molgraph_dict.pkl"):
        with open(os.path.join(hp['data_dir'], "molgraph_dict.pkl"), 'rb') as f:
            molgraph_dict = pickle.load(f)
    else:
        molgraph_dict = {}

        
    sample_g = molgraph_dict[list(molgraph_dict.keys())[0]]
    x_size = sample_g.ndata['h'].shape[1]
    e_size = sample_g.edata['e'].shape[1]
    virtual_node = hp['num_virtual_nodes'] > 0
    
    encoder = Protein()
    prot_input_dim = len(encoder)
    
    models_list = []
    
    if prot_model == None:
        Print("Creating Base Protein LM Model...", output)
        prot_model = PROT_CNN(hp, prot_input_dim)
        prot_model = prot_model.to(device)

    if mol_model == None:  
        Print("Creating Molecule GNN Model...", output)
        mol_model = MOL_GNN2(hp, x_size)
        mol_model = mol_model.to(device)
    
    models_list.append([prot_model, "", True, True, False])
    models_list.append([mol_model, "mol", False, False, False])
        
    if model_name == None:
        load_models(hp, models_list, device, output)

    for model, idx, frz, _, _ in models_list: model.eval()
    
    Print('Creating Dataset...', output)
    test_ds = create_test_dataset(hp, encoder, device, molgraph_dict, output) 
    
    test_params = {'batch_size': hp['batch_size_val'],
                 'shuffle': False}
    collate_fn = collate_gnn_cpi(molgraph_dict)
    
    test_dl = DataLoader(test_ds, collate_fn=collate_fn, **test_params)

    if model_name == None:
        model_name = hp['gnn_type'] + "_" + str(int(round(time.time() * 1000)))
        os.mkdir("results/" + model_name)
    loss_func = torch.nn.CrossEntropyLoss()
    metrics = [get_cindex, r2_score]
            
    test_loss = 0
    test_ap = 0
    predlist = torch.Tensor()
    labellist = torch.Tensor()
    for batch_id, (g, p_block, lengths, y) in enumerate(tqdm(test_dl, total=int(len(test_dl)), leave=False)):
        g = g.to(torch.device(device))
        p_block = p_block.to(torch.device(device))
        lengths = lengths.to(torch.device(device))
        y = y.to(torch.device(device))[:,0]
        with torch.no_grad():
            z0, r0 = prot_model(p_block, lengths)
            prediction, _, _ = mol_model(g, g.ndata['h'], r0, z0)
            
        loss = loss_func(prediction, y)
        logits = F.softmax(prediction, 1)
        logits = logits[:,1]
        logits = logits.cpu()

        y = y.cpu()

        test_loss += loss.detach().item()
        prediction = prediction[:,1]
        prediction = prediction.cpu()
        predlist = torch.cat([predlist, prediction])
        labellist = torch.cat([labellist, y])
        
    test_loss /= (batch_id + 1)
    
    test_ap = average_precision_score(labellist, predlist)
    metrics = report_metric(len(list(set(test_ds.compounds))), len(list(set(test_ds.orig_seq))), labellist, predlist, np.array(test_ds.orig_inter, dtype=int))
    inline_log = 'Test_loss: {:.3f}, Test_ap: {:.3f}, Test_rpec: {:.3f}'.format(test_loss, test_ap, metrics['rprecision'][0])
    Print(inline_log, output)
    inline_log = 'Test_Comp_map: {:.3f}, Test_Comp_rpec: {:.3f}, Test_cmap_3: {:.3f}, Test_Comp_prec_1: {:.3f}'.format(
        metrics['compound_map'][0], metrics['compound_rprecision'][0], metrics['compound_map_3'][0], metrics['compound_precision_1'][0])
    Print(inline_log, output)
    inline_log = 'Test_Seq_map: {:.3f}, Test_seq_rprec: {:.3f}, Test_smap_3: {:.3f}, Test_seq_prec_1: {:.3f}'.format(
        metrics['enzyme_map'][0], metrics['enzyme_rprecision'][0], metrics['enzyme_map_3'][0], metrics['enzyme_precision_1'][0])
    Print(inline_log, output)
    model_time = str(int(round(time.time() * 1000)))
    outfile = 'results/'+model_time+'_posneg_hist.png'
    create_posneg_distrib(predlist, labellist, outfile)
    Print("created pos neg distrib in {}".format(outfile), output)
    #torch.save(predlist, "predlist.pt")
    #torch.save(labellist, "labellist.pt")
    return test_loss, test_ap



if __name__ == "__main__":
    import sys
    
    with open('default.yaml') as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)
        
    if len(sys.argv) > 1:
        for i in range(len(sys.argv) - 1):
            key, value_raw = sys.argv[i+1].split("=")
            print(str(key) + ": " + value_raw)
            try:
                hp[key] = int(value_raw)
            except ValueError:
                try:
                    hp[key] = float(value_raw)
                except ValueError:
                    hp[key] = value_raw

    logfile = hp['logfile']
    output = open(logfile, "a")
        
    test_loss, test_ap = test(None, None, None, hp, output)
    Print('Testing done', output)
    
