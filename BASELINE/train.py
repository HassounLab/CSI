# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)

import torch
import pickle
from dataset import create_train_val_dataset, create_test_dataset, collate_gnn_cpi
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dgllife.utils import EarlyStopping
from tqdm import tqdm
import yaml
import os
import time
import matplotlib.pyplot as plt
from utils import Print, load_models, get_cindex, save_model, print_hp
from test_model import test
from dataset import Protein
from models import PROT_CNN, MOL_GNN2
from sklearn.metrics import average_precision_score
from collections import defaultdict

#from torch.profiler import profile, record_function, ProfilerActivity

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
    
    models_list = [] # list of lists [model, idx, flag_frz, flag_clip_grad, flag_clip_weight]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    print('Creating Dataset...')
    train_ds, val_ds = create_train_val_dataset(mode, hp, encoder, device, molgraph_dict, output) 
    
    dl_params = {'batch_size': hp['batch_size_train'],
                 'shuffle': True}
    val_params = {'batch_size': hp['batch_size_val'],
                 'shuffle': True}
    
    collate_fn = collate_gnn_cpi(molgraph_dict)
    train_dl = DataLoader(train_ds, collate_fn=collate_fn, **dl_params)
    val_dl = DataLoader(val_ds, collate_fn=collate_fn, **val_params)
    
    sample_g = molgraph_dict[list(molgraph_dict.keys())[0]]
    x_size = sample_g.ndata['h'].shape[1]
    
    
    # Create model
        
    if not os.path.exists("early_stopping/"):
        os.mkdir("early_stopping/")
    if not os.path.exists("results/"):
        os.mkdir("results/")
        
    Print("Creating Base Protein LM Model...", output)
        
    # Create model
    prot_model = PROT_CNN(hp, prot_input_dim)
    models_list.append([prot_model, "", False, False, False])

            
    prot_model = prot_model.to(device)
    
    Print("Creating Molecule GNN Model...", output)
    
    mol_model = MOL_GNN2(hp, x_size)
    models_list.append([mol_model, "mol", False, False, False])

    load_models(hp, models_list, device, output)

    model_name = hp['gnn_type'] + "_" + str(int(round(time.time() * 1000)))
    os.mkdir("results/" + model_name)
    
    neg_ratio = hp.get("neg_to_pos", 1.0)
    train_wt = torch.Tensor([1.0, float(neg_ratio)])
    train_wt = train_wt.to(device)
    loss_func = torch.nn.CrossEntropyLoss(weight=train_wt)
    val_loss_func = torch.nn.CrossEntropyLoss()
    params, pr_params = [], []
    for model, idx, frz, _, _ in models_list:
        if frz: continue
        elif idx != "mol": params    += [p for p in model.parameters() if p.requires_grad]
        else:             pr_params += [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam([{'params':params,    'lr':hp['prot_lr'],      'weight_decay': hp['l2']   },
                                  {'params':pr_params, 'lr':hp['mol_lr'],   'weight_decay': hp['l2']   }])
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25)
    stopper = EarlyStopping(
        mode='lower', patience = hp['early_stopping_patience'], 
        filename = "early_stopping/mol_"+ str(int(round(time.time() * 1000))))
    prot_stopper = EarlyStopping(
        mode='lower', patience = hp['early_stopping_patience'], 
        filename = "early_stopping/prot_"+ str(int(round(time.time() * 1000))))
    
    train_losses = []
    val_losses = []
    Print("Starting Training...", output)
        
    for epoch in range(hp['num_epoch']):
        train_loss = 0
        for model, idx, frz, _, _ in models_list: model.train()

        for batch_id, (g, p_block, lengths, y) in enumerate(tqdm(train_dl, total=int(len(train_dl)), leave=False)):

            g = g.to(torch.device(device))
            p_block = p_block.to(torch.device(device))
            lengths = lengths.to(torch.device(device))
            y = y.to(torch.device(device))[:,0]
            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            #with torch.autograd.profiler.profile() as prof:
            z0, r0 = prot_model(p_block, lengths)
                
            prediction, _, _ = mol_model(g, g.ndata['h'], r0, z0)
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
        for model, idx, frz, _, _ in models_list: model.eval()

        for batch_id, (g, p_block, lengths, y) in enumerate(val_dl):
            g = g.to(torch.device(device))
            p_block = p_block.to(torch.device(device))
            lengths = lengths.to(torch.device(device))
            y = y.to(torch.device(device))[:,0]
        
            with torch.no_grad():
                z0, r0 = prot_model(p_block, lengths)
                #z0_list = prot_model.em(z0, lengths, cpu=True)
                #z0_list = [t.to(device) if type(t) is torch.Tensor else t for t in z0_list]
                prediction, _, _ = mol_model(g, g.ndata['h'], r0, z0)
                
            loss = val_loss_func(prediction, y)
            logits = F.softmax(prediction, 1)
            logits = logits[:,1]
            logits = logits.cpu()
            y = y.cpu()
            #ap = average_precision_score(y, logits)
            prediction = prediction[:,1]
            prediction = prediction.cpu()
            predlist = torch.cat([predlist, prediction])
            labellist = torch.cat([labellist, y])
            if batch_id % 10 == 0: 
                Print('# val_loss {:.1%} loss={:.4f}'.format(
                    batch_id / len(val_dl), loss.item()), output)

            val_loss += loss.detach().item()
            #val_ap += ap
            
        val_loss /= (batch_id + 1)
        #val_ap /= (batch_id + 1)
        val_ap = average_precision_score(labellist, predlist)
        inline_log = 'Epoch {} / {}, train_loss: {:.4f}, val_loss: {:.4f}, ap: {:.4f}'.format(
            epoch + 1, hp['num_epoch'], train_loss, val_loss, val_ap
        )
        Print(inline_log, output)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        #scheduler.step()
        early_stop = prot_stopper.step(val_loss, prot_model)
        early_stop = stopper.step(val_loss, mol_model)
        if early_stop:
            saved_model_name = "Saved early stopping model in " + stopper.filename
            Print(saved_model_name, output)
            break
    
        
    plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.ylim([0, 2])
    plt.legend(['train', 'val'], loc='upper left')
    plt.title('Loss')
    plt.savefig("results/" + model_name + "/training_loss.png")
    loss_graph_file = "created loss graph in results/" + model_name +"/training_loss.png"
    Print(loss_graph_file, output)
            
    return prot_model, mol_model, val_ap, train_loss, val_loss, model_name


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
    print_hp(hp, output)
    
    prot_model, mol_model, val_ap, train_loss, val_loss, model_name = train(hp, output)
    test_loss, test_ap = test(prot_model, mol_model, model_name, hp, output)
    Print('Training done', output)
    
