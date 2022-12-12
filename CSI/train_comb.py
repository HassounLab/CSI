# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)

import torch
import pandas as pd
import numpy as np
import pickle
from dataset import create_contr_dataset, create_test_dataset, create_train_val_dataset, collate_contr_views, collate_gnn_cpi
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgllife.utils import EarlyStopping
from tqdm import tqdm
from torch.distributions import Normal
import yaml
import os
import time
import matplotlib.pyplot as plt
from utils import Print, load_models, get_gpu_memory, get_cindex, save_model, print_hp, set_seeds, contrastive_loss, contrastive_loss3, contrastive_loss_softmax, save_all_models, DebugAnomaly, MyEarlyStopping
from test_model_comb import test_comb
from dataset import Protein
from models import PROT_CNN, MEI_V1, MEI_V2, CONTR_FINAL, PROT_FINAL_V3, CONTR_FINAL_2, PROT_CNN2
from sklearn.metrics import auc, roc_auc_score, roc_curve, average_precision_score
from collections import defaultdict
import gc, sys, GPUtil, psutil

def train(hp, output, **kwargs):
    if len(kwargs) > 1:
        Print("Training with default with the following/above modifications:", output)
    for key, value in kwargs.items():
        hp[key] = value
        Print(str(key) + ": " + str(value), output)
    encoder = Protein()
    mode = 'both'
    prot_input_dim = len(encoder)
    
    if os.path.exists(hp['data_dir']+"molgraph_dict.pkl"):
        with open(os.path.join(hp['data_dir'], "molgraph_dict.pkl"), 'rb') as f:
            molgraph_dict = pickle.load(f)
    else:
        molgraph_dict = {}
        
        
    models_list_1 = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    models_list_2 = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print("available GPU memory = {}".format(get_gpu_memory()), output)
    print('Creating Dataset...')
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
            
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)

    
    sample_g = molgraph_dict[list(molgraph_dict.keys())[0]]
    x_size = sample_g.ndata['h'].shape[1]
    e_size = sample_g.edata['e'].shape[1]
    virtual_node = hp['num_virtual_nodes'] > 0
    
    
    # Create model
        
    if not os.path.exists("early_stopping/"):
        os.mkdir("early_stopping/")
    if not os.path.exists("results/"):
        os.mkdir("results/")
    
    
    Print("Creating Base Protein LM Model...", output)
        
    # Create model
    prot_model_1 = PROT_CNN(hp, prot_input_dim)
    models_list_1.append([prot_model_1, "", False, False, False])
    prot_model_v2_1 = None
    if 2 in contr_views_1:
        prot_model_v3_1 = PROT_CNN2(hp, prot_input_dim)
        models_list_1.append([prot_model_v3_1, "prot_v3", False, False, False])
    else:
        prot_model_v3_1 = None

    prot_model_1 = prot_model_1.to(device)
    if 2 in contr_views_1:        
        prot_model_v3_1 = prot_model_v3_1.to(device)
        
        prot_model_final_v3_1 = PROT_FINAL_V3(hp)
        prot_model_final_v3_1 = prot_model_final_v3_1.to(device)
        models_list_1.append([prot_model_final_v3_1, "pf_v3", False, True, False])
    else:
        prot_model_final_v3_1 = None
    
    Print("Creating Molecule GNN Model...", output)

    if 0 in contr_views_1:
        model_v1_1 = MEI_V1(hp, x_size)
        model_v1_1 = model_v1_1.to(device)
        models_list_1.append([model_v1_1, "mei_v1", False, False, False])
    else:
        model_v1_1 = None
    
    if 1 in contr_views_1:
        model_v2_1 = MEI_V2(hp, x_size)
        model_v2_1 = model_v2_1.to(device)
        models_list_1.append([model_v2_1, "mei_v2", False, False, False])
    else:
        model_v2_1 = None
    model_v3_1 = None

    
    load_models(hp1, models_list_1, device, output)

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
                    
    # Create model
    prot_model_2 = PROT_CNN(hp, prot_input_dim)
    models_list_2.append([prot_model_2, "", False, False, False])
    prot_model_v2_2 = None
    if 2 in contr_views_2:
        prot_model_v3_2 = PROT_CNN2(hp, prot_input_dim)
        models_list_2.append([prot_model_v3_2, "prot_v3", False, False, False])
    else:
        prot_model_v3_2 = None

    prot_model_2 = prot_model_1.to(device)
    if 2 in contr_views_2:        
        prot_model_v3_2 = prot_model_v3_2.to(device)
        
        prot_model_final_v3_2 = PROT_FINAL_V3(hp)
        prot_model_final_v3_2 = prot_model_final_v3_2.to(device)
        models_list_2.append([prot_model_final_v3_2, "pf_v3", False, True, False])
    else:
        prot_model_final_v3_2 = None
    
    Print("Creating Molecule GNN Model...", output)

    if 0 in contr_views_2:
        model_v1_2 = MEI_V1(hp, x_size)
        model_v1_2 = model_v1_2.to(device)
        models_list_2.append([model_v1_2, "mei_v1", False, False, False])
    else:
        model_v1_2 = None
    
    if 1 in contr_views_2:
        model_v2_2 = MEI_V2(hp, x_size)
        model_v2_2 = model_v2_2.to(device)
        models_list_2.append([model_v2_2, "mei_v2", False, False, False])
    else:
        model_v2_2 = None
    model_v3_2 = None

    
    load_models(hp2, models_list_2, device, output)
    
    model_time = str(int(round(time.time() * 1000)))
    model_name = hp['gnn_type'] + "_" + model_time
    os.mkdir("results/" + model_name)
    
    Print("available GPU memory = {}".format(get_gpu_memory()), output)


    print('Creating Final Training Dataset...')
    Print("available GPU memory = {}".format(get_gpu_memory()), output)
    
    gc.collect()
    Print("available GPU memory = {}".format(get_gpu_memory()), output)
    
    train_ds, val_ds = create_train_val_dataset(mode, hp, encoder, device, molgraph_dict, output) 
    
    dl_params = {'batch_size': hp['batch_size_train'],
                 'shuffle': True}
    val_params = {'batch_size': hp['batch_size_val'],
                 'shuffle': True}
    
    collate_fn = collate_gnn_cpi(molgraph_dict)
    train_dl = DataLoader(train_ds, collate_fn=collate_fn, **dl_params)
    val_dl = DataLoader(val_ds, collate_fn=collate_fn, **val_params)    

    Print("Creating Final Prediction Model...", output)
    
    final_models_list, models_list = [], []
    pred_model = CONTR_FINAL_2(hp, num_view_1, num_view_2, hp['full_model'])    
    pred_model = pred_model.to(device)
    models_list.append([pred_model, "pred", False, False, False])   
    final_models_list.append([pred_model, "pred", False, False, False])   

    final_prot_model = None
    final_mei_model = None
        
    for model in models_list_1+models_list_2+models_list:
        idx = model[1]
        if idx != "pred" and idx != "final_prot" and idx != "final_mei":
            model[2] = True
                       
    load_models(hp, final_models_list, device, output)

    params = []
    for model, idx, frz, _, _ in models_list:
        if frz: continue
        else:             params += [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([{'params':params,    'lr':hp['pred_lr']}])
    #optimizer = torch.optim.Adam([{'params':params,    'lr':hp['pred_lr'], 'weight_decay': hp['l2']}])

    train_losses = []
    val_losses = []

    stopper = MyEarlyStopping(models_list, model_time, hp, output,
        mode='lower', patience = hp['early_stopping_patience'], 
        filename = "early_stopping/pred_"+ str(int(round(time.time() * 1000))))
    
    Print("Starting Final Training...", output)
    neg_ratio = hp.get("neg_to_pos", 1.0)
    train_wt = torch.Tensor([1.0, float(neg_ratio)])
    train_wt = train_wt.to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=train_wt)
    val_loss_func = torch.nn.CrossEntropyLoss()
        
    for epoch in range(hp['num_epoch']):
        train_loss = 0
        for model, idx, frz, _, _ in models_list_1+models_list_2+models_list: model.train()

        for batch_id, (g, p_block, lengths, y) in enumerate(tqdm(train_dl, total=int(len(train_dl)), leave=False)):

            f_view_l = []
            g = g.to(torch.device(device))
            p_block = p_block.to(torch.device(device))
            lengths = lengths.to(torch.device(device))
            y = y.to(torch.device(device))[:,0]
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
        
            loss.backward()
            
            for model, _, _, clip_grad, _ in models_list:
                if clip_grad: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            optimizer.zero_grad()            


            train_loss += loss.detach().item()
            
        train_loss /= (batch_id + 1)
        #print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cpu_time_total"))
        
        val_loss = 0
        val_ap = 0.0
        predlist = torch.Tensor()
        labellist = torch.Tensor()
        for model, idx, frz, _, _ in models_list_1+models_list_2+models_list: model.eval()

        for batch_id, (g, p_block, lengths, y) in enumerate(val_dl):
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
                
                
            loss = val_loss_func(prediction, y)
            logits = F.softmax(prediction, 1)
            logits = logits[:,1]
            logits = logits.cpu()
            y = y.cpu()
            prediction = prediction[:,1]
            prediction = prediction.cpu()
            predlist = torch.cat([predlist, prediction])
            labellist = torch.cat([labellist, y])

            val_loss += loss.detach().item()
            
        val_loss /= (batch_id + 1)
        val_ap = average_precision_score(labellist, predlist)
        inline_log = 'Epoch {} / {}, train_loss: {:.4f}, val_loss: {:.4f}, ap: {:.4f}'.format(
            epoch + 1, hp['num_epoch'], train_loss, val_loss, val_ap
        )
        Print(inline_log, output)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        early_stop = stopper.step(val_loss)
        if early_stop:
            saved_model_name = "Saved early stopping model in " + stopper.filename
            Print(saved_model_name, output)
            break
    
    save_all_models(hp, models_list, model_time + "_last", output)
        
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.ylim([0, 2])
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Loss')
    plt.savefig("results/" + model_name + "/training_loss.png")  
    loss_graph_file = "created loss graph in results/" + model_name +"/training_loss.png"
    Print(loss_graph_file, output)
    plt.close()
    return prot_model_1, prot_model_v2_1, prot_model_v3_1, prot_model_final_v3_1, model_v1_1, model_v2_1, model_v3_1, prot_model_2, prot_model_v2_2, prot_model_v3_2, prot_model_final_v3_2, model_v1_2, model_v2_2, model_v3_2, pred_model, final_prot_model, final_mei_model, train_loss, val_loss, val_ap, model_name


if __name__ == "__main__":
    import sys
    set_seeds(2021)
    
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
    print_hp(hp, output)
    
    prot_model_1, prot_model_v2_1, prot_model_v3_1, prot_model_final_v3_1, model_v1_1, model_v2_1, model_v3_1, prot_model_2, prot_model_v2_2, prot_model_v3_2, prot_model_final_v3_2, model_v1_2, model_v2_2, model_v3_2, pred_model, final_prot_model, final_mei_model, train_loss, val_loss, val_ap, model_name = train(hp, output)
    test_loss, test_ap = test_comb(prot_model_1, prot_model_v2_1, prot_model_v3_1, prot_model_final_v3_1, model_v1_1, model_v2_1, model_v3_1, 
    prot_model_2, prot_model_v2_2, prot_model_v3_2, prot_model_final_v3_2, model_v1_2, model_v2_2, model_v3_2, pred_model, final_prot_model, final_mei_model, model_name, hp, output)
    Print('Training done', output)
    
