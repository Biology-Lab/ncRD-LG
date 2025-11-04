import sys
sys.path.append('../code')

from collections import defaultdict
from re import split
from random import shuffle, choice
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
import torch


class ModelConf(object):
    def __init__(self, file):
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, key):
        return self.config[key]

    def contain(self, key):
        return key in self.config
    
    def get(self, key, default=None):
        """Get configuration value, return default if not found"""
        return self.config.get(key, default)

    def read_configuration(self, file):
        with open(file) as f:
            for ind, line in enumerate(f):
                line = line.strip()
                # Skip empty lines and comment lines
                if line != '' and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)  # Only split first = sign
                        self.config[key.strip()] = value.strip()
                    else:
                        print(f"Warning: Line {ind+1} format incorrect, skipping: {line}")


class OptionConf(object):
    def __init__(self, content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i, drug in enumerate(self.line):
            if (drug.startswith('-') or drug.startswith('--')) and not drug[1:].isdigit():
                ind = i
                if i < len(self.line) - 1:
                    while drug.startswith('-') and not drug[1:].isdigit():
                        ind = i + 1
                        if ind < len(self.line) - 1:
                            drug = self.line[ind]
                        else:
                            break
                    if ind < len(self.line):
                        self.options[drug] = self.line[ind + 1]

    def __getitem__(self, key):
        return self.options[key]

    def get(self, key, default=None):
        return self.options.get(key, default)


class Interaction(object):
    def __init__(self, conf, training_set, test_set):
        # miRNA version interaction data initialization (main code)
        self.config = conf
        self.training_set = training_set
        self.test_set = test_set
        self.lncRNA_num = 0
        self.drug_num = 0
        self.lncRNA = {}
        self.drug = {}
        
        # Load miRNA data
        self.load_miRNA_data()
        
        # lncRNA version interaction data initialization (commented for future activation)
        # def __init__(self, conf, training_set, test_set):
        #     """lncRNA version interaction data initialization"""
        #     self.config = conf
        #     self.training_set = training_set
        #     self.test_set = test_set
        #     self.lncRNA_num = 0
        #     self.drug_num = 0
        #     self.lncRNA = {}
        #     self.drug = {}
        #     
        #     # Load lncRNA data
        #     self.load_lncRNA_data()

    def load_miRNA_data(self):
        """Load miRNA data"""
        # miRNA version data loading logic (main code)
        print("Loading miRNA data...")
        
        # Extract all miRNAs and drugs from training and test sets
        all_lncRNAs = set()
        all_drugs = set()
        
        for lncRNA_id, drug_id, _ in self.training_set:
            all_lncRNAs.add(lncRNA_id)
            all_drugs.add(drug_id)
        
        for lncRNA_id, drug_id, _ in self.test_set:
            all_lncRNAs.add(lncRNA_id)
            all_drugs.add(drug_id)
        
        # Create ID mapping
        self.lncRNA = {lncRNA_id: idx for idx, lncRNA_id in enumerate(all_lncRNAs)}
        self.drug = {drug_id: idx for idx, drug_id in enumerate(all_drugs)}
        
        self.lncRNA_num = len(self.lncRNA)
        self.drug_num = len(self.drug)
        
        print(f"Loading completed: {self.lncRNA_num} miRNAs, {self.drug_num} drugs")
        
        # lncRNA version data loading logic (commented for future activation)
        # def load_lncRNA_data(self):
        #     """Load lncRNA data"""
        #     print("Loading lncRNA data...")
        #     
        #     all_lncRNAs = set()
        #     all_drugs = set()
        #     
        #     for lncRNA_id, drug_id, _ in self.training_set:
        #         all_lncRNAs.add(lncRNA_id)
        #         all_drugs.add(drug_id)
        #     
        #     for lncRNA_id, drug_id, _ in self.test_set:
        #         all_lncRNAs.add(lncRNA_id)
        #         all_drugs.add(drug_id)
        #     
        #     self.lncRNA = {lncRNA_id: idx for idx, lncRNA_id in enumerate(all_lncRNAs)}
        #     self.drug = {drug_id: idx for idx, drug_id in enumerate(all_drugs)}
        #     
        #     self.lncRNA_num = len(self.lncRNA)
        #     self.drug_num = len(self.drug)
        #     
        #     print(f"Loading completed: {self.lncRNA_num} lncRNAs, {self.drug_num} drugs")


class LGCN_Encoder(object):
    def __init__(self, data, emb_size, n_layers):
        # miRNA version GCN encoder initialization (main code)
        self.data = data
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.embedding_dict = {}
        self.optimizer = None
        
        # Initialize embeddings
        self.init_embeddings()
        
        # Create optimizer
        self.create_optimizer()
        
        # lncRNA version GCN encoder initialization (commented for future activation)
        # def __init__(self, data, emb_size, n_layers):
        #     """lncRNA version GCN encoder initialization"""
        #     self.data = data
        #     self.emb_size = emb_size
        #     self.n_layers = n_layers
        #     self.embedding_dict = {}
        #     self.optimizer = None
        #     
        #     self.init_embeddings()
        #     self.create_optimizer()

    def init_embeddings(self):
        """Initialize embeddings"""
        # miRNA version embedding initialization (main code)
        # Initialize miRNA embeddings
        self.embedding_dict['lncRNA_emb'] = torch.nn.Parameter(
            torch.randn(self.data.lncRNA_num, self.emb_size)
        )
        
        # Initialize drug embeddings
        self.embedding_dict['drug_emb'] = torch.nn.Parameter(
            torch.randn(self.data.drug_num, self.emb_size)
        )
        
        # lncRNA version embedding initialization (commented for future activation)
        # def init_embeddings_lnc(self):
        #     """lncRNA version embedding initialization"""
        #     self.embedding_dict['lncRNA_emb'] = torch.nn.Parameter(
        #         torch.randn(self.data.lncRNA_num, self.emb_size)
        #     )
        #     self.embedding_dict['drug_emb'] = torch.nn.Parameter(
        #         torch.randn(self.data.drug_num, self.emb_size)
        #     )

    def create_optimizer(self):
        """Create optimizer"""
        # miRNA version optimizer creation (main code)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        # lncRNA version optimizer creation (commented for future activation)
        # def create_optimizer_lnc(self):
        #     """lncRNA version optimizer creation"""
        #     self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self):
        """Forward pass"""
        # miRNA version forward pass logic (main code)
        # Get embeddings
        lncRNA_emb = self.embedding_dict['lncRNA_emb']
        drug_emb = self.embedding_dict['drug_emb']
        
        # Apply GCN layers
        for layer in range(self.n_layers):
            lncRNA_emb = self.gcn_layer(lncRNA_emb, 'lncRNA')
            drug_emb = self.gcn_layer(drug_emb, 'drug')
        
        # Return concatenated embeddings
        return torch.cat([lncRNA_emb, drug_emb], dim=0)
        
        # lncRNA version forward pass logic (commented for future activation)
        # def forward_lnc(self):
        #     """lncRNA version forward pass"""
        #     lncRNA_emb = self.embedding_dict['lncRNA_emb']
        #     drug_emb = self.embedding_dict['drug_emb']
        #     
        #     for layer in range(self.n_layers):
        #         lncRNA_emb = self.gcn_layer(lncRNA_emb, 'lncRNA')
        #         drug_emb = self.gcn_layer(drug_emb, 'drug')
        #     
        #     return torch.cat([lncRNA_emb, drug_emb], dim=0)

    def gcn_layer(self, emb, node_type):
        """GCN layer"""
        # miRNA version GCN layer logic (main code)
        # Simplified GCN layer implementation
        # Should implement actual GCN layer with adjacency matrix and message passing
        # For simplicity, return original embedding
        return emb
        
        # lncRNA version GCN layer logic (commented for future activation)
        # def gcn_layer_lnc(self, emb, node_type):
        #     """lncRNA version GCN layer"""
        #     return emb

    def parameters(self):
        """Return model parameters"""
        # miRNA version parameter return logic (main code)
        return [self.embedding_dict['lncRNA_emb'], self.embedding_dict['drug_emb']]
        
        # lncRNA version parameter return logic (commented for future activation)
        # def parameters_lnc(self):
        #     """lncRNA version parameter return"""
        #     return [self.embedding_dict['lncRNA_emb'], self.embedding_dict['drug_emb']]

    def state_dict(self):
        """Return model state dictionary"""
        # miRNA version state dictionary return logic (main code)
        return {
            'lncRNA_emb': self.embedding_dict['lncRNA_emb'].state_dict(),
            'drug_emb': self.embedding_dict['drug_emb'].state_dict()
        }
        
        # lncRNA version state dictionary return logic (commented for future activation)
        # def state_dict_lnc(self):
        #     """lncRNA version state dictionary return"""
        #     return {
        #         'lncRNA_emb': self.embedding_dict['lncRNA_emb'].state_dict(),
        #         'drug_emb': self.embedding_dict['drug_emb'].state_dict()
        #     }


def load_config(config_file):
    """Load configuration file"""
    # miRNA version configuration loading (main code)
    config = ModelConf(config_file)
    return config
    
    # lncRNA version configuration loading (commented for future activation)
    # def load_config_lnc(config_file):
    #     """lncRNA version configuration loading"""
    #     config = ModelConf(config_file)
    #     return config


def load_data(config):
    """Load data"""
    # miRNA version data loading (main code)
    print("Loading miRNA data...")
    
    # Get data path from configuration
    data_path = config.get('data_path', '../../Dataset/')
    
    # Load training data
    training_set = load_training_data(data_path)
    
    # Load test data
    test_set = load_test_data(data_path)
    
    # Create interaction object
    interaction = Interaction(config, training_set, test_set)
    
    return interaction
    
    # lncRNA version data loading (commented for future activation)
    # def load_data_lnc(config):
    #     """lncRNA version data loading"""
    #     print("Loading lncRNA data...")
    #     
    #     data_path = config.get('data_path', '../../Dataset/')
    #     
    #     training_set = load_training_data_lnc(data_path)
    #     test_set = load_test_data_lnc(data_path)
    #     
    #     interaction = Interaction(config, training_set, test_set)
    #     
    #     return interaction


def load_training_data(data_path):
    """Load training data"""
    # miRNA version training data loading (main code)
    training_data = []
    
    # Load data from MiPair.csv
    import pandas as pd
    try:
        df = pd.read_csv(f'{data_path}MiPair.csv')
        for _, row in df.iterrows():
            lncRNA_id = row['miRNA_Name']
            drug_id = row['Drug_Name']
            label = 1  # Positive sample
            training_data.append((lncRNA_id, drug_id, label))
    except Exception as e:
        print(f"Failed to load training data: {e}")
    
    return training_data
    
    # lncRNA version training data loading (commented for future activation)
    # def load_training_data_lnc(data_path):
    #     """lncRNA version training data loading"""
    #     training_data = []
    #     
    #     import pandas as pd
    #     try:
    #         df = pd.read_csv(f'{data_path}LncPair.csv')
    #         for _, row in df.iterrows():
    #             lncRNA_id = row['lncRNA_Name']
    #             drug_id = row['Drug_Name']
    #             label = 1  # Positive sample
    #             training_data.append((lncRNA_id, drug_id, label))
    #     except Exception as e:
    #         print(f"Failed to load lncRNA training data: {e}")
    #     
    #     return training_data


def load_test_data(data_path):
    """Load test data"""
    # miRNA version test data loading (main code)
    test_data = []
    
    # Load data from MiPair.csv (simplified processing, should load from test set)
    import pandas as pd
    try:
        df = pd.read_csv(f'{data_path}MiPair.csv')
        # Take first 20% as test data
        test_size = int(len(df) * 0.2)
        test_df = df.head(test_size)
        for _, row in test_df.iterrows():
            lncRNA_id = row['miRNA_Name']
            drug_id = row['Drug_Name']
            label = 1  # Positive sample
            test_data.append((lncRNA_id, drug_id, label))
    except Exception as e:
        print(f"Failed to load test data: {e}")
    
    return test_data
    
    # lncRNA version test data loading (commented for future activation)
    # def load_test_data_lnc(data_path):
    #     """lncRNA version test data loading"""
    #     test_data = []
    #     
    #     import pandas as pd
    #     try:
    #         df = pd.read_csv(f'{data_path}LncPair.csv')
    #         test_size = int(len(df) * 0.2)
    #         test_df = df.head(test_size)
    #         for _, row in test_df.iterrows():
    #             lncRNA_id = row['lncRNA_Name']
    #             drug_id = row['Drug_Name']
    #             label = 1  # Positive sample
    #             test_data.append((lncRNA_id, drug_id, label))
    #     except Exception as e:
    #         print(f"Failed to load lncRNA test data: {e}")
    #     
    #     return test_data


def prepare_training_data(data):
    """Prepare training data"""
    # miRNA version training data preparation (main code)
    training_data = []
    
    # Extract data from training set
    for lncRNA_id, drug_id, label in data.training_set:
        training_data.append((lncRNA_id, drug_id, label))
    
    return training_data
    
    # lncRNA version training data preparation (commented for future activation)
    # def prepare_training_data_lnc(data):
    #     """lncRNA version training data preparation"""
    #     training_data = []
    #     
    #     for lncRNA_id, drug_id, label in data.training_set:
    #         training_data.append((lncRNA_id, drug_id, label))
    #     
    #     return training_data


def get_test_data(data):
    """Get test data"""
    # miRNA version test data retrieval (main code)
    test_data = []
    
    # Extract data from test set
    for lncRNA_id, drug_id, label in data.test_set:
        test_data.append((lncRNA_id, drug_id, label))
    
    return test_data
    
    # lncRNA version test data retrieval (commented for future activation)
    # def get_test_data_lnc(data):
    #     """lncRNA version test data retrieval"""
    #     test_data = []
    #     
    #     for lncRNA_id, drug_id, label in data.test_set:
    #         test_data.append((lncRNA_id, drug_id, label))
    #     
    #     return test_data