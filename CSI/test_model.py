# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)

import torch
import pandas as pd
import numpy as np
import pickle
from dataset import create_contr_dataset, create_test_dataset, collate_contr_views, collate_gnn_cpi
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
from models import PROT_CNN, MEI_V1, MEI_V2, CONTR_FINAL, PROT_FINAL_V3, PROT_CNN2
from sklearn.metrics import auc, roc_auc_score, r2_score, average_precision_score
from collections import defaultdict

def test(prot_model, prot_model_v2, prot_model_v3, prot_model_final_v3, model_v1, model_v2, model_v3, 
         pred_model, final_prot_model, final_mei_model, model_name, hp, output, **kwargs):
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
    
    if prot_model_v3 is not None or hp.get("pretrained_v3_model", None) is not None:
        num_view = 3
    else:
        num_view = 2

    if hp.get("contr_views", None) is not None:
        contr_views = hp.get("contr_views")
        num_view = len(contr_views)
    else:
        if num_view == 2:
            contr_views = [0,1]
        else:
            contr_views = [0,1,2]
        
    if prot_model == None:
        Print("Creating Base Protein LM Model...", output)
        prot_model = PROT_CNN(hp, prot_input_dim)
        prot_model_v2 = PROT_CNN(hp, prot_input_dim)
        prot_model_v3 = PROT_CNN2(hp, prot_input_dim)
        prot_model = prot_model.to(device)
        prot_model_v2= prot_model_v2.to(device)
        prot_model_v3= prot_model_v3.to(device)

    if model_v1 == None:  
        Print("Creating Molecule GNN Model...", output)
        model_v1 = MEI_V1(hp, x_size)
        model_v1 = model_v1.to(device)

    if model_v2 == None:
        model_v2 = MEI_V2(hp, x_size)
        model_v2 = model_v2.to(device)
        
    if pred_model == None:  
        pred_model = CONTR_FINAL(hp, num_view, hp['full_model'])    
        pred_model = pred_model.to(device)

    if prot_model_final_v3 == None:
        prot_model_final_v3 = PROT_FINAL_V3(hp)
        prot_model_final_v3 = prot_model_final_v3.to(device)

    if final_prot_model == None:
        final_prot_model = PROT_CNN(hp, prot_input_dim)
        models_list.append([final_prot_model, "final_prot", False, False, False])

    final_prot_model = final_prot_model.to(device)

    if final_mei_model == None:
        final_mei_model = MEI_V1(hp, x_size)
        final_mei_model = final_mei_model.to(device)
        models_list.append([final_mei_model, "final_mei", False, False, False])
    
    models_list.append([prot_model, "", True, True, False])
    if prot_model_v2 is not None:
        models_list.append([prot_model_v2, "prot_v2", True, True, False])
    if 2 in contr_views:
        if prot_model_v3 is not None:
            models_list.append([prot_model_v3, "prot_v3", True, True, False])
    models_list.append([model_v1, "mei_v1", True, False, False])
    models_list.append([model_v2, "mei_v2", True, False, False])
    models_list.append([pred_model, "pred", True, False, False])   
    if 2 in contr_views:
        models_list.append([prot_model_final_v3, "pf_v3", True, True, False])
        
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
            
    test_loss = 0
    test_ap = 0
    predlist = torch.Tensor()
    labellist = torch.Tensor()
    for batch_id, (g, p_block, lengths, y) in enumerate(tqdm(test_dl, total=int(len(test_dl)), leave=False)):
        f_view_l = []
        g = g.to(torch.device(device))
        p_block = p_block.to(torch.device(device))
        lengths = lengths.to(torch.device(device))
        y = y.to(torch.device(device))[:,0]
        with torch.no_grad():
            z0, r0 = prot_model(p_block, lengths)            
            if 0 in contr_views:
                f_mei, _ = model_v1(g, g.ndata['h'], r0, z0)
                f_view_l.append(f_mei)
            else:
                f_mei = []
            
            if 1 in contr_views:
                f_cc, _ = model_v2(g, g.ndata['h'], g, g.ndata['h'])
                f_view_l.append(f_cc)
            else:
                f_cc = []
            
            if 2 in contr_views:
                _, f_fasta = prot_model_v3(p_block, lengths, p_block, lengths)
                f_fasta = prot_model_final_v3(f_fasta)
                f_view_l.append(f_fasta)
            else:
                f_fasta = []
            
            if num_view == 2:
                f_view_l.append([])
            f_view_l.append([])            
            
                
            prediction = pred_model(*f_view_l)

            
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
        
    test_loss, test_ap = test(None, None, None, None, None, None, None, None, None, None, None, hp, output)
    Print('Testing done', output)
    
