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
from models import PROT_CNN, MEI_V1, MEI_V2, CONTR_FINAL, PROT_FINAL_V3, CONTR_FINAL_2, PROT_CNN2
from sklearn.metrics import auc, roc_auc_score, r2_score, average_precision_score
from collections import defaultdict

def test_comb(prot_model_1, prot_model_v2_1, prot_model_v3_1, prot_model_final_v3_1, model_v1_1, model_v2_1, model_v3_1, 
    prot_model_2, prot_model_v2_2, prot_model_v3_2, prot_model_final_v3_2, model_v1_2, model_v2_2, model_v3_2, 
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
    
    models_list, models_list_1, models_list_2 = [], [], []
    

    num_view_1 = 2
    with open('model1.yaml') as f:
        hp1 = yaml.load(f, Loader=yaml.FullLoader)
    if hp1.get("contr_views", None) is not None:
        contr_views_1 = hp1.get("contr_views")
        num_view_1 = len(contr_views_1)
    else:
        if num_view_1 == 2:
            contr_views_1 = [0,1]
        else:
            contr_views_1 = [0,1,2]
        
    if prot_model_1 == None:
        Print("Creating Base Protein LM Model...", output)
        prot_model_1 = PROT_CNN(hp, prot_input_dim)
        prot_model_v2_1 = PROT_CNN(hp, prot_input_dim)
        prot_model_v3_1 = PROT_CNN2(hp, prot_input_dim)
        prot_model_1 = prot_model_1.to(device)
        prot_model_v2_1= prot_model_v2_1.to(device)
        prot_model_v3_1= prot_model_v3_1.to(device)

    if model_v1_1 == None:  
        Print("Creating Molecule GNN Model...", output)
        model_v1_1 = MEI_V1(hp, x_size)
        model_v1_1 = model_v1_1.to(device)

    if model_v2_1 == None:
        model_v2_1 = MEI_V2(hp, x_size)
        model_v2_1 = model_v2_1.to(device)
            
    if prot_model_final_v3_1 == None:
        prot_model_final_v3_1 = PROT_FINAL_V3(hp)
        prot_model_final_v3_1 = prot_model_final_v3_1.to(device)

    models_list_1.append([prot_model_1, "", True, True, False])
    if prot_model_v2_1 is not None:
        models_list_1.append([prot_model_v2_1, "prot_v2", True, True, False])
    if 2 in contr_views_1:
        if prot_model_v3_1 is not None:
            models_list_1.append([prot_model_v3_1, "prot_v3", True, True, False])
    models_list_1.append([model_v1_1, "mei_v1", True, False, False])
    models_list_1.append([model_v2_1, "mei_v2", True, False, False])
    if 2 in contr_views_1:
        models_list_1.append([prot_model_final_v3_1, "pf_v3", True, True, False])

    num_view_2 = 2
    with open('model2.yaml') as f:
        hp2 = yaml.load(f, Loader=yaml.FullLoader)
    if hp2.get("contr_views", None) is not None:
        contr_views_2 = hp2.get("contr_views")
        num_view_2 = len(contr_views_2)
    else:
        if num_view_2 == 2:
            contr_views_2 = [0,1]
        else:
            contr_views_2 = [0,1,2]
        
    if prot_model_2 == None:
        Print("Creating Base Protein LM Model...", output)
        prot_model_2 = PROT_CNN(hp, prot_input_dim)
        prot_model_v2_2 = PROT_CNN(hp, prot_input_dim)
        prot_model_v3_2 = PROT_CNN2(hp, prot_input_dim)
        prot_model_2 = prot_model_2.to(device)
        prot_model_v2_2= prot_model_v2_2.to(device)
        prot_model_v3_2= prot_model_v3_2.to(device)

    if model_v1_2 == None:  
        Print("Creating Molecule GNN Model...", output)
        model_v1_2 = MEI_V1(hp, x_size)
        model_v1_2 = model_v1_2.to(device)

    if model_v2_2 == None:
        model_v2_2 = MEI_V2(hp, x_size)
        model_v2_2 = model_v2_2.to(device)
            
    if prot_model_final_v3_2 == None:
        prot_model_final_v3_2 = PROT_FINAL_V3(hp)
        prot_model_final_v3_2 = prot_model_final_v3_2.to(device)

    models_list_2.append([prot_model_2, "", True, True, False])
    if prot_model_v2_2 is not None:
        models_list_2.append([prot_model_v2_2, "prot_v2", True, True, False])
    if 2 in contr_views_2:
        if prot_model_v3_2 is not None:
            models_list_2.append([prot_model_v3_2, "prot_v3", True, True, False])
    models_list_2.append([model_v1_2, "mei_v1", True, False, False])
    models_list_2.append([model_v2_2, "mei_v2", True, False, False])
    if 2 in contr_views_2:
        models_list_2.append([prot_model_final_v3_2, "pf_v3", True, True, False])
        
    if pred_model == None:  
        pred_model = CONTR_FINAL_2(hp, num_view_1, num_view_2, hp['full_model'])    
        pred_model = pred_model.to(device)

    models_list.append([pred_model, "pred", True, False, False])   

    if final_prot_model == None:
        final_prot_model = PROT_CNN(hp, prot_input_dim)
        models_list.append([final_prot_model, "final_prot", False, False, False])

        final_prot_model = final_prot_model.to(device)

    if final_mei_model == None:
        final_mei_model = MEI_V1(hp, x_size)
        final_mei_model = final_mei_model.to(device)
        models_list.append([final_mei_model, "final_mei", False, False, False])
    
        
    if model_name == None:
        load_models(hp1, models_list_1, device, output)
        load_models(hp2, models_list_2, device, output)
        load_models(hp, models_list, device, output)

    for model, idx, frz, _, _ in models_list_1+models_list_2+models_list: model.eval()
    
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
            z0, r0 = prot_model_1(p_block, lengths)            
            if 0 in contr_views_1:
                f_mei, _ = model_v1_1(g, g.ndata['h'], r0, z0)
                f_view_l.append(f_mei)
            else:
                f_mei = []
            
            if 1 in contr_views_1:
                f_cc, _ = model_v2_1(g, g.ndata['h'], g, g.ndata['h'])
                f_view_l.append(f_cc)
            else:
                f_cc = []
            
            if 2 in contr_views_1:
                _, f_fasta = prot_model_v3_1(p_block, lengths, p_block, lengths)
                f_fasta = prot_model_final_v3_1(f_fasta)
                f_view_l.append(f_fasta)
            else:
                f_fasta = []
            
            if num_view_1 == 2:
                f_view_l.append([])
            if num_view_1 == 1:
                f_view_l.append([])
                f_view_l.append([])
                
            z0, r0 = prot_model_2(p_block, lengths)
            if 0 in contr_views_2:
                f_mei, _ = model_v1_2(g, g.ndata['h'], r0, z0)
                f_view_l.append(f_mei)
            else:
                f_mei = []
            
            if 1 in contr_views_2:
                f_cc, _ = model_v2_2(g, g.ndata['h'], g, g.ndata['h'])
                f_view_l.append(f_cc)
            else:
                f_cc = []
            
            if 2 in contr_views_2:
                _, f_fasta = prot_model_v3_2(p_block, lengths, p_block, lengths)
                f_fasta = prot_model_final_v3_2(f_fasta)
                f_view_l.append(f_fasta)
            else:
                f_fasta = []
            
            if num_view_2 == 2:
                f_view_l.append([])
            if num_view_2 == 1:
                f_view_l.append([])
                f_view_l.append([])
                
            f_view_l.append([])            
            
                
            prediction = pred_model(f_view_l)

            
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
        
    test_loss, test_ap = test_comb(None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, hp, output)
    Print('Testing done', output)
    
