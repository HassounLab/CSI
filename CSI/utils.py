# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)

""" Utility functions """

import os
import sys
import math
import random
import numpy as np
from datetime import datetime
from sklearn.metrics import auc, roc_auc_score, r2_score, average_precision_score
from scipy.stats import pearsonr, spearmanr, t
import matplotlib.pyplot as plt
import subprocess as sp

import torch
import torch.nn as nn
import colorama
import torch
import torch.nn.functional as F
from dgllife.utils import EarlyStopping

import pdb
import traceback
from colorama import Fore, Back, Style
from torch import autograd
colorama.init()

contr_softmax_loss = nn.CrossEntropyLoss()

class DebugAnomaly (autograd.detect_anomaly):
  def __init__(self):
    super(DebugAnomaly, self).__init__()
  def __enter__(self):
    super(DebugAnomaly, self).__enter__()
    return self
  def __exit__(self, type, value, trace):
    super(DebugAnomaly, self).__exit__()
    if isinstance(value, RuntimeError):
      traceback.print_tb(trace)
      halt(str(value))


def halt(msg):
  print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
  print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
  print (Fore.RED + "┃        Debug Anomaly 00000004, 0000AAC0             ┃")
  print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
  print(Style.RESET_ALL)
  print (msg)
  pdb.set_trace()


def Print(string, output, newline=False):
    """ print to stdout and a file (if given) """
    time = datetime.now()
    print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print('\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string]), file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def remove_file(filename):
    if os.path.exists(filename):
        os.remove(filename)
        
def set_output(args, string, test=False, embedding=False):
    """ set output configurations """
    output, save_prefix, index = sys.stdout, None, ""
    if args["output_path"] is not None:
        if not test and not embedding:
            if not os.path.exists(args["output_path"] + "/weights/"):
                os.makedirs(args["output_path"] + "/weights/", exist_ok=True)
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/weights/" + index

        elif not embedding:
            if not os.path.exists(args["output_path"]):
                os.makedirs(args["output_path"], exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/weights/"

        else:
            if not os.path.exists(args["output_path"] + "/embeddings/"):
                os.makedirs(args["output_path"] + "/embeddings/", exist_ok=True)
            if args["pretrained_model"] is not None:
                index += os.path.splitext(args["pretrained_model"])[0].split("/")[-1] + "_"
            if args["output_index"] is not None:
                index += args["output_index"] + "_"
            output = open(args["output_path"] + "/" + index + string + ".txt", "a")
            save_prefix = args["output_path"] + "/embeddings/"
            
    return output, save_prefix


def load_models(hp, models, device, output):
    """ load models if pretrained_models are available """
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_enz_model" if idx == "" else "pretrained_%s_model" % idx
        idx_path = hp.get(idx)
        if idx_path is not None:
            Print('loading %s weights from %s' % (idx, hp[idx]), output)
            models[m][0].load_weights(hp[idx])

        models[m][0] = models[m][0].to(device)

def save_all_models(hp, models, model_time, output):
    for m in range(len(models)):
        model, idx = models[m][0], models[m][1]
        idx = "pretrained_enz_model" if idx == "" else "pretrained_%s_model" % idx
        filename = hp['data_dir'] + idx + '_' + model_time + '.pt'
        save_model(model, filename, output)
        
def save_model(model, filename, output):
    Print('saving %s weights to %s' % (model.__class__.__name__, filename), output)
    torch.save(model.state_dict(), filename)


            
def evaluate_result(dict, metric):
    """ calculate the evaluation metric from the given results """
    result = None
    if metric   == "acc"  : result = (dict["correct"]) / (dict["n"]+ np.finfo(float).eps)
    elif metric == "pr"   : result = dict["tp"] / (dict["tp"] + dict["fp"]+ np.finfo(float).eps)
    elif metric == "re"   : result = dict["tp"] / (dict["tp"] + dict["fn"]+ np.finfo(float).eps)
    elif metric == "sp"   : result = dict["tn"] / (dict["tn"] + dict["fp"]+ np.finfo(float).eps)
    elif metric == "f1"   : result = 2 * dict["tp"] / (2 * dict["tp"] + dict["fp"] + dict["fn"]+ np.finfo(float).eps)
    elif metric == "mcc"  : result = ((dict["tp"] * dict["tn"] - dict["fp"] * dict["fn"]) /
                                    (math.sqrt((dict["tp"] + dict["fp"]) * (dict["tp"] + dict["fn"]) *
                                               (dict["tn"] + dict["fp"]) * (dict["tn"] + dict["fn"])) + np.finfo(float).eps))

    if result is None:
        if "labels" in dict and len(dict["labels"].shape) == 2: dict["labels"] = dict["labels"][:, 0]
        if "logits" in dict and len(dict["logits"].shape) == 2: dict["logits"] = dict["logits"][:, 0]
    if metric   == "auc"  : result = auc(dict["labels"], dict["logits"])
    elif metric == "aupr" : result = roc_auc_score(dict["labels"], dict["logits"])
    elif metric == "ci"   : result = ci(dict["labels"], dict["logits"])
    elif metric == "r2"   : result = r2_score(dict["labels"], dict["logits"])
    elif metric == "r"    : result = pearsonr(dict["labels"], dict["logits"])[0]
    elif metric == "rho"  : result = spearmanr(dict["labels"], dict["logits"])[0]
    return result


def steiger_test(xy, xz, yz, n):
    """ One-tailed Steiger's test for the statistic significance between two dependent correlation coefficients """
    # ab: correlation coefficient between a and b (xy, xz, yz)
    # n: number of elements in x, y and z
    d = xy - xz
    determin = 1 - xy ** 2 - xz ** 2 - yz ** 2 + 2 * xy * xz * yz
    av = (xy + xz)/2
    cube = (1 - yz) * (1 - yz) * (1 - yz)
    e = (n - 1) * (1 + yz)/(((2 * (n - 1)/(n - 3)) * determin + (av ** 2) * cube))
    if e < 0:
        return np.nan, np.nan
    t2 = d * np.sqrt(e)
    p = 1 - t.cdf(abs(t2), n - 2)

    return t2, p

def ci(y,f):
    y = y.numpy()
    f = f.numpy()
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci

def get_cindex(Y, P):
    Y = Y.numpy()
    P = P.numpy()

    summ = 0
    pair = 0
    
    for i in range(1, len(Y)):
        for j in range(0, i):
            if i is not j:
                if(Y[i] > Y[j]):
                    pair +=1
                    summ +=  1* (P[i] > P[j]) + 0.5 * (P[i] == P[j])
        
            
    if pair != 0:
        return summ/pair
    else:
        return 0
    
def nan_hook(self, inp, output):
            if type(output) == torch.nn.utils.rnn.PackedSequence:
               outputs = [output.data]
            elif not isinstance(output, tuple):
                if type(output[0]) == torch.nn.utils.rnn.PackedSequence:
                    outputs = [output.data]
                else:
                    outputs = [output]
            else:
                if type(output[0]) == torch.nn.utils.rnn.PackedSequence:
                    outputs = [output[0].data]
                else:
                    outputs = output

            for i, out in enumerate(outputs):
                nan_mask = torch.isnan(out)
                if nan_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])
                inf_mask = torch.isinf(out)
                if inf_mask.any():
                    print("In", self.__class__.__name__)
                    raise RuntimeError(f"Found INF in output {i} at indices: ", inf_mask.nonzero(), "where:", out[inf_mask.nonzero()[:, 0].unique(sorted=True)])


def print_hp(hp, output):
    hplist = []
    hplist.append("Values of Hyperparameters:")
    for key, value in hp.items():
        hplist.append("{} : {}".format(key, value))
    for str in hplist:
        Print(str, output)
        
def array_to_bitstring(array):
    return ''.join(['{:02d}'.format(int(x)) for x in array])

def unique_tensor(tensor_list):
    #returns list of unique tensors from input list of tensors
    ret_list = []
    for i, t in enumerate(tensor_list):
        found = False
        for tt in ret_list:
            if torch.equal(tt, t): 
                found = True
                break
        if not found:
            ret_list.append(t)
    
    return ret_list

def r_squared_error(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)

def gen_at_k_metrics(self, labels, logits, k):
    ylist = [(logits[i], labels[i]) for i, _ in enumerate(labels)]
    ylist.sort(key=lambda x: x[0], reverse=True)
    
    # Number of relevant items
    n_rel = sum((true_r == 1) for (_, true_r) in ylist)

    # Number of recommended items in top k
    n_rec_k = k

    # Number of relevant and recommended items in top k
    n_rel_and_rec_k = sum((true_r == 1) for (_, true_r) in ylist[:k])
    n_r = sum((true_r == 1) for (_, true_r) in ylist[:n_rel])
    
    # Precision@K: Proportion of recommended items that are relevant
    # When n_rec_k is 0, Precision is undefined. We here set it to 0.

    prec_at_k = torch.true_divide(n_rel_and_rec_k, n_rec_k) if n_rec_k != 0 else 0
    # Recall@K: Proportion of relevant items that are recommended
    # When n_rel is 0, Recall is undefined. We here set it to 0.

    recall_at_k = torch.true_divide(n_rel_and_rec_k, n_rel) if n_rel != 0 else 0
    r_precision = torch.true_divide(n_r, n_rel) if n_rel != 0 else 0
    
    return prec_at_k, recall_at_k, r_precision

def get_k(y_obs,y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs,y_pred):
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down= sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def get_rm2(ys_orig,ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

def create_batch_views(v):
    bsz = v.shape[0]
    feat_sz = v.shape[1]
    device = v.device
    
    alt_v = torch.zeros((bsz, bsz, feat_sz), device=device)
    for idx in range(bsz):
        alt_v[idx] = v.clone().detach()
        tmp = alt_v[idx, 0, :]
        alt_v[idx, 0, :] = alt_v[idx, idx, :]
        alt_v[idx, idx, :] = tmp
        
    return alt_v

def contrastive_loss(v1, v2, tau=1.0):
    
    v1_norm = torch.norm(v1, dim=1, keepdim=True)
    v2_norm = torch.norm(v2, dim=1, keepdim=True)
    
    v2T = torch.transpose(v2, 0, 1)
    
    inner_prod = torch.matmul(v1, v2T)
    
    v2_normT = torch.transpose(v2_norm, 0, 1)
    
    norm_mat = torch.matmul(v1_norm, v2_normT)
    
    loss_mat = torch.div(inner_prod, norm_mat)
    
    loss_mat = loss_mat * (1/tau)
    
    loss_mat = torch.exp(loss_mat)
    
    numerator = torch.diagonal(loss_mat)
    numerator = torch.unsqueeze(numerator, 0)
    
    Lv1_v2_denom = torch.sum(loss_mat, dim=1, keepdim=True)
    Lv1_v2_denom = torch.transpose(Lv1_v2_denom, 0, 1)
    #Lv1_v2_denom = Lv1_v2_denom - numerator
    
    Lv2_v1_denom = torch.sum(loss_mat, dim=0, keepdim=True)
    #Lv2_v1_denom = Lv2_v1_denom - numerator
    
    Lv1_v2 = torch.div(numerator, Lv1_v2_denom)
    
    Lv1_v2 = -1 * torch.log(Lv1_v2)
    Lv1_v2 = torch.mean(Lv1_v2)
    
    Lv2_v1 = torch.div(numerator, Lv2_v1_denom)
    
    Lv2_v1 = -1 * torch.log(Lv2_v1)
    Lv2_v1 = torch.mean(Lv2_v1)
    
    return Lv1_v2 + Lv2_v1

def contrastive_loss3(v1, v2, v3, tau=1.0):
    Lv1v2 = contrastive_loss(v1, v2, tau)
    Lv2v3 = contrastive_loss(v2, v3, tau)
    Lv3v1 = contrastive_loss(v3, v1, tau)
    
    return Lv1v2 + Lv2v3 + Lv3v1

def contrastive_loss_softmax(v1, v2, tau=1.0):
    batchSize = v1.shape[0]
    inputSize = v1.shape[1]
    alt_v1 = create_batch_views(v1)
    alt_v2 = create_batch_views(v2)
    Lv1_v2 = torch.bmm(alt_v2, v1.view(batchSize, inputSize, 1))
    Lv2_v1 = torch.bmm(alt_v1, v2.view(batchSize, inputSize, 1))
    Lv1_v2 = torch.div(Lv1_v2, tau)
    Lv2_v1 = torch.div(Lv2_v1, tau)
    Lv1_v2 = Lv1_v2.contiguous()
    Lv2_v1 = Lv2_v1.contiguous()
    
    Lv1_v2 = Lv1_v2.squeeze()
    Lv2_v1 = Lv2_v1.squeeze()
    
    label = torch.zeros([batchSize]).cuda().long()

    Lv1_v2 = contr_softmax_loss(Lv1_v2, label)
    Lv2_v1 = contr_softmax_loss(Lv2_v1, label)
    
    return Lv1_v2 + Lv2_v1

def create_posneg_distrib(predlist, labellist, outfile):
    logits = F.softmax(predlist)
    logits = logits/max(logits)
    posidx = torch.nonzero(labellist == 1)
    negidx = torch.nonzero(labellist == 0)
    poslogits = [p.item() for p in logits[posidx]]
    neglogits = [n.item() for n in logits[negidx]]
    #bins = np.linspace(0, 1, 100)
    bins = np.linspace(min(min(poslogits), min(neglogits)), max(max(poslogits), max(neglogits)), 100)
    plt.figure()
    plt.hist(poslogits, bins, alpha=0.5, label='pos', log=True, color='g')
    plt.hist(neglogits, bins, alpha=0.5, label='neg', log=True, color='r')
    plt.xlabel("Normalized prediction probabilities")
    plt.ylabel("Number of interactions - log scale")
    plt.savefig(outfile)
    
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

class MyEarlyStopping(EarlyStopping):
    def __init__(self, models_list, model_time, hp, output, mode='higher', patience=10, filename=None, metric=None):
        super(MyEarlyStopping, self).__init__(mode=mode, patience=patience, filename=filename, metric=metric)
        self.models_list = models_list
        self.model_time = model_time
        self.hp = hp
        self.output = output

    def step(self, score):
        """Update based on a new score.

        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.

        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            save_all_models(self.hp, self.models_list, self.model_time + "_best", self.output)
        elif self._check(score, self.best_score):
            self.best_score = score
            save_all_models(self.hp, self.models_list, self.model_time + "_best", self.output)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            save_all_models(self.hp, self.models_list, self.model_time + "_current", self.output)

        return self.early_stop
        
def report_metric(num_compound, num_enzyme, true_interaction, pred_interaction, inter_data):
    metric = {}

    # compute map
    def map(n, dim, k=None):
        rst = []
        for i in range(n):
            indices = inter_data[:, dim] == i
            if indices.sum() == 0: continue
            x = true_interaction[indices]
            y = pred_interaction[indices]
            if k is not None:
                y_sorted_indices = np.argsort(-y)[:k]
                x = x[y_sorted_indices]
                y = y[y_sorted_indices]
                if x.sum() == 0:
                    rst.append(0)
                    continue
            if x.sum() == 0:
                rst.append(0)
            else:
                rst.append(average_precision_score(y_true=x, y_score=y))
        rst = (np.mean(rst), np.std(rst) / np.sqrt(len(rst)))
        return rst
    metric['compound_map'] = map(num_compound, 0, k=None)
    metric['enzyme_map'] = map(num_enzyme, 1, k=None)

    metric['compound_map_3'] = map(num_compound, 0, k=3)
    metric['enzyme_map_3'] = map(num_enzyme, 1, k=3)

    # compute r precision and precision@k(1, 3)
    def precision(n, k=None, dim=None):
        def h(x, y):
            m_true = int(x.sum()) if k is None else k
            if m_true == 0: return -1

            xy = np.vstack([x, y]).T
            xy_sorted_indices = np.argsort(-xy[:, 1])
            xy = xy[xy_sorted_indices]

            z = xy[:m_true, 0].sum() / m_true

            return z

        rst = []
        if dim is None:
            x = true_interaction
            y = pred_interaction
            z = h(x, y)
            if z != -1:
                rst.append(z)
        else:
            for i in range(n):
                indices = inter_data[:, dim] == i
                if indices.sum() == 0: continue
                x = true_interaction[indices]
                y = pred_interaction[indices]
                z = h(x, y)
                if z != -1:
                    rst.append(z)
        if k is None and dim is None:
            rst = [np.mean(rst)]
        else:
            rst = (np.mean(rst), np.std(rst) / np.sqrt(len(rst)))
        return rst

    metric['compound_rprecision'] = precision(num_compound, k=None, dim=0)
    metric['enzyme_rprecision'] = precision(num_enzyme, k=None, dim=1)
    metric['rprecision'] = precision(num_enzyme, k=None, dim=None)

    metric['compound_precision_1'] = precision(num_compound, k=1, dim=0)
    metric['enzyme_precision_1'] = precision(num_enzyme, k=1, dim=1)

    return metric
