# CSI
# Contrastive Data Stratification for Interaction Prediction
# Environment and packages:
All python packages needed are listed in the file package_list.txt

# dataset:
This directory contains all the datasets. There is a directory for each of the datasets. Within each directory
are files for the contrastive dataset with stratification by specific keys as explained in the paper. The format
of the contrastive dataset files is:
[view number] [object1] [object2] [key]
The training, dev and test files for final predictor training are also available in each dataset dir. The format
of these files is:
[SMILES string] [FASTA string] [Interaction label]
All files are gzip'ed files because of their size. Please unzip them before using.

# BASELINE:
This directory is for the baseline models. All options to train and test the model are specified in default.yaml
file. One version of the pre-trained model is also included. These are specified in the default.yaml file in the
pretrained* variables.

To try the model run the command "python test_model.py". This will test the pretrained model on the KEGG
test dataset. If you want to test the baseline model on your own data, create your test in the format specified
above and change the test_data variable in the default.yaml file to point to your test data

To train the baseline model, comment out the pretrained* variables in the default.yaml file and run the command
"python train.py".

It is strongly recommended to run on a GPU since these models are fairly large

# CSI:
This is the CSI model. As with baseline, all options to train and test the model are specified in the defult.yaml
file. In addition to this, there are also two other option files - model1.yaml (for phase 1A) and model2.yaml
(for phase 1B). One version of the pretrained phase 1A model and phase 1B model and the final predictor are
included.

To try the model run the command "python test_model_comb.py". This will test the pretrained model on the KEGG
dataset. To test out the pretrained CSI model on your own data, follow the same steps as outlined for the
BASELINE model.

To train the model for phase 1A or phase 1B, set the required contrastive dataset in the train_contr_data
variable in the defauly.yaml file and the reqired training, dev and test data for the final predictor in the
train_data, valid_data and test_data variables. Comment out any pretrained* variables and run the command
"python tain.py".
