# Author - Apurva Kalia, Tufts University (apurva.kalia@tufts.edu)

# Load Packages
import torch
import pandas as pd
import numpy as np
import pickle
import rdkit.Chem as Chem
import dgllife.utils as chemutils
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import dgl
import yaml
from utils import Print
import os
from collections import defaultdict
import json
import math
from sklearn.preprocessing import QuantileTransformer

class Protein:
    """ protein sequence encoder """
    def __init__(self):
        chars = b'ARNDCQEGHILKMFPSTWYVXOUBZ'
        encoding = np.arange(len(chars))
        encoding[21:] = [11, 4, 20, 20]  # encode 'OUBZ' as synonyms
        encoding += 1                    # leave 0 for padding tokens
        self.chars = np.frombuffer(chars, dtype=np.uint8)
        self.encoding = np.zeros(256, dtype=np.uint8) + 21
        self.encoding[self.chars] = encoding
        self.size = encoding.max() + 1

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        return chr(self.chars[i])

    def encode(self, x, reverse_complement=False):
        """ encode a byte string into alphabet indices """
        if not reverse_complement:
            x = np.frombuffer(x, dtype=np.uint8)
            string = self.encoding[x]
        else:
            x = np.frombuffer(x, dtype=np.uint8)[::-1]
            string = self.encoding_rc[x]
        return string

    def decode(self, x, reverse_complement=False):
        """ decode index array, x, to byte string of this alphabet """
        if not reverse_complement:
            string = self.chars[x-1]
        else:
            string = self.chars_rc[x[::-1]-1]
        return string.tobytes()
    
def atom_bond_type_one_hot(atom):
    bs = atom.GetBonds()
    bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
    return [any(bt[:, i]) for i in range(bt.shape[1])]

def split_sequence(sequence, prot_ngram_dict, ngram):
    sequence = '-' + sequence + '='
    words = [prot_ngram_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def get_atom_featurizer(feature_mode, element_list):
    atom_mass_fun = chemutils.ConcatFeaturizer(
        [chemutils.atom_mass]
    )
    
    def atom_type_one_hot(atom):
        return chemutils.atom_type_one_hot(
            atom, allowable_set = element_list, encode_unknown = True
        )
    
    if feature_mode == 'light':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot
        ])
    elif feature_mode == 'full':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot, 
            atom_bond_type_one_hot,
            chemutils.atom_degree_one_hot, 
            chemutils.atom_total_degree_one_hot,
            chemutils.atom_explicit_valence_one_hot,
            chemutils.atom_implicit_valence_one_hot,
            chemutils.atom_hybridization_one_hot,
            chemutils.atom_total_num_H_one_hot,
            chemutils.atom_formal_charge_one_hot,
            chemutils.atom_num_radical_electrons_one_hot,
            chemutils.atom_is_aromatic_one_hot,
            chemutils.atom_is_in_ring_one_hot,
            chemutils.atom_chiral_tag_one_hot
        ])
    elif feature_mode == 'medium':
        atom_featurizer_funs = chemutils.ConcatFeaturizer([
            chemutils.atom_mass,
            atom_type_one_hot, 
            atom_bond_type_one_hot,
            chemutils.atom_total_degree_one_hot,
            chemutils.atom_total_num_H_one_hot,
            chemutils.atom_is_aromatic_one_hot,
            chemutils.atom_is_in_ring_one_hot,
        ])

    return chemutils.BaseAtomFeaturizer(
        {"h": atom_featurizer_funs, 
        "m": atom_mass_fun}
    )

def get_bond_featurizer(feature_mode, self_loop):
    if feature_mode == 'light':
        return chemutils.BaseBondFeaturizer(
            featurizer_funcs = {'e': chemutils.ConcatFeaturizer([
                chemutils.bond_type_one_hot
            ])}, self_loop = self_loop
        )
    elif feature_mode == 'full':
        return chemutils.CanonicalBondFeaturizer(
            bond_data_field='e', self_loop = self_loop
        )

def create_atoms(mol):
    # NOTE: my error handling
    try:
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    except Exception as e:
        print("Error creating atoms: {}".format(str(e)))
        return None
    return np.array(atoms)

def parse_cpi_stream(f, comment=b'#'):
    """ parse fasta stream for (compound, sequence, label) data """
    compound, sequence, label = None, [], []
    for line in f:
        if line.startswith(comment): continue
        line = line.strip().split()
        compound, sequence = line[0], line[1]
        try:
            label = line[2]
        except:
            print("CPI line did not have a third element; setting -999 as label")
            label = -999
        if '.' in compound.decode():
            #print("found period in smile {}".format(compound.decode()))
            continue # ignore compounds with '.' in them
        if compound is not None:
            yield compound, sequence, label

def seq_cat(length, prot):
    x = np.zeros(length)
    for i, ch in enumerate(prot[:length]): 
        x[i] = prot[i]
    return x  
            
def load_data(datafile, hp, encoder, molgraph_dict, device, sanity_check=False):
    comp_dict = defaultdict(lambda: len(comp_dict))
    seq_dict = defaultdict(lambda: len(seq_dict))
    with open(datafile, 'rb') as f:
        graphs, compounds, adjacencies, sequences, labels = [], [], [], [], []
        orig_seq, orig_inter_l = [], []
        for compound, sequence, label in parse_cpi_stream(f):
            if hp['prot_min_len'] != -1 and len(sequence) <  hp['prot_min_len']: continue
            if hp['prot_max_len'] != -1 and len(sequence) >  hp['prot_max_len']: continue
            if hp['prot_truncate'] != -1 and len(sequence) > hp['prot_truncate']: sequence = sequence[:hp['prot_truncate']]
            if hp['sanity_check'] and len(sequences) == 100: break
            mol = Chem.MolFromSmiles(compound.decode())
            atoms = create_atoms(mol)
            
            if atoms is None:
                print("no atom failure in compound  {}".format(compound.decode()))
                continue

            if len(atoms) == 1:
                #print("found unit length compound {}".format(compound.decode()))
                continue
            
            atom = [a in hp['element_list'] for a in atoms]            
            if not all(atom):
                continue

            smile = compound.decode()
            if smile not in molgraph_dict.keys():
                node_featurizer = get_atom_featurizer(hp['atom_feature'], hp['element_list']) 
                edge_featurizer = get_bond_featurizer(hp['bond_feature'], hp['self_loop'])
                
                g = chemutils.mol_to_bigraph(mol, node_featurizer=node_featurizer, edge_featurizer=edge_featurizer, add_self_loop = hp['self_loop'],
                                     num_virtual_nodes = hp['num_virtual_nodes'])
                if g is None:
                    print("could not create graph for {}".format(smile))
                    continue
                molgraph_dict[smile] = g.to(device)
            compounds.append(smile)
            orig_seq.append(sequence)
            orig_inter_l.append((comp_dict[smile], seq_dict[sequence], label))
            sequence = encoder.encode(sequence.upper())
            sequence = seq_cat(hp['cnn_prot_length'], sequence)
            sequences.append(sequence)
            labels.append(np.array([float(label)]))
    
    sequences = [torch.from_numpy(sequence).long().to(device) for sequence in sequences]
    labels = torch.from_numpy(np.stack(labels)).long().to(device)
    return compounds, sequences, orig_seq, labels, orig_inter_l, molgraph_dict


class CPIDataset(Dataset):
    def __init__(self, compounds, sequences, orig_seq, labels, hp, encoder, orig_inter, mol_dict=None):
        self.compounds = compounds
        self.sequences = sequences
        self.orig_seq = orig_seq
        self.labels = labels
        self.num_alphabets = len(encoder)
        self.orig_inter = orig_inter
        self.mol_dict = mol_dict
        if mol_dict is not None:
            self.use_mol_dict = True
        else:
            self.use_mol_dict = False
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        x = self.sequences[idx]
        tokens = torch.zeros(len(x) + 2, dtype=torch.long) + (self.num_alphabets - 1)
        tokens[1:len(x) + 1] = x
        return self.compounds[idx], tokens, self.labels[idx]
    
class collate_gnn_cpi(object):
    # class to hold and batch graph objects
    def __init__(self, *params):
        self.molgraph_dict = params[0]
    def __call__(self, batch):
        retval = list(collate_cpi_sequences(batch))
        compounds = retval[0]
        batched_g = [self.molgraph_dict[c] for c in compounds]
        batched_g = dgl.batch(batched_g)
        ret_tuple = tuple(retval[1:])
        ret_tuple = batched_g, *ret_tuple
        return ret_tuple

def collate_cpi_sequences(args):
    """ collate sequences with different lengths into a single matrix; use 0 for [START / STOP/ PADDING] tokens """
    comp = [a[0] for a in args]
    x = [a[1] for a in args]
    y = [a[2] for a in args]
    
    lengths = np.array([len(seq) for seq in x])
    b, l = len(x), max(lengths)
    x_block = x[0].new_zeros((b, l))
    y_block = torch.stack(y, 0)
    
    for i in range(b):
        x_block[i, 1:len(x[i])-1] = x[i][1:-1]
    lengths = torch.from_numpy(lengths)

    return comp, x_block, lengths, y_block


def split_dataset(dataset, ratio=0.8):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2
    
def create_train_val_dataset(mode, hp, encoder, device, molgraph_dict, output):
    
    compounds, seq, orig_seq, labels, orig_inter, molgraph_dict = load_data(hp['train_data'], hp, 
                                                      encoder, molgraph_dict, device, hp['sanity_check'])

    if mode == 'both': #this means both train and valid data set files are supplied
        train_comp, train_seq, train_orig_seq, train_labels, train_orig_inter = compounds, seq, orig_seq, labels, orig_inter
        valid_comp, valid_seq, valid_orig_seq, valid_labels, valid_orig_inter, molgraph_dict = load_data(hp['valid_data'], hp, 
                                                                       encoder, molgraph_dict, device, hp['sanity_check'])

    else: # this means that train set will be divided into train and validation
        dataset = list(map(list, zip(compounds, seq, labels)))

        dataset_train, dataset_valid = split_dataset(dataset)
        train_comp, train_seq, train_labels = list(zip(*dataset_train))
        valid_comp, valid_seq, valid_labels = list(zip(*dataset_valid))
    
    Print("Training Set with {} examples".format(len(train_seq)), output)
    Print("Validation Set with {} examples".format(len(valid_seq)), output)
    
    train_ds = CPIDataset(train_comp, train_seq, train_orig_seq, train_labels, hp, encoder, train_orig_inter, molgraph_dict)
    val_ds = CPIDataset(valid_comp, valid_seq, valid_orig_inter, valid_labels, hp, encoder, valid_orig_inter, molgraph_dict)
    with open(os.path.join(hp['data_dir'], 'molgraph_dict.pkl'), 'wb') as f:
        pickle.dump(molgraph_dict, f)
    return train_ds, val_ds



def create_test_dataset(hp, encoder, device, molgraph_dict, output):

    test_compounds, test_seq, test_orig_seq, test_labels, test_orig_inter, molgraph_dict = load_data(hp['test_data'], hp, 
                                                      encoder, molgraph_dict, device, hp['sanity_check'])
        
    Print("Test Set with {} examples from {}".format(len(test_seq), hp['test_data']), output)
    test_ds = CPIDataset(test_compounds, test_seq, test_orig_seq, test_labels, hp, encoder, test_orig_inter, molgraph_dict)

    return test_ds

    
if __name__ == "__main__":
    print('Testing pytorch dataset definition...')
    with open('default.yaml') as f:
        hp = yaml.load(f, Loader=yaml.FullLoader)
    
    molgraph_dict = {}
    encoder = Protein()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logfile = hp['logfile']
    output = open(logfile, "a")

    train_ds, val_ds = create_train_val_dataset('both', hp, encoder, device, molgraph_dict, output)
    #load_data(hp['train_data'], hp, encoder, molgraph_dict, device, sanity_check=False)
    print('Success!')



