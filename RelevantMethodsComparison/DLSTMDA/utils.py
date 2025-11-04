import os
import numpy as np
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp',dataset='',
                 xd=None, xt=None, y=None,z=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        self.dataset = dataset
        self.xd = xd
        self.xt = xt
        self.y = y
        self.z = z
        self.smile_graph = smile_graph
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y, z, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    
    @property
    def processed_paths(self):
        return [os.path.join(self.processed_dir, self.dataset + '.pt')]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        pass

    def process(self, xd, xt, y, z, smile_graph):
        # miRNA version data processing logic (main code)
        assert (len(xd) == len(xt)), "The number of drugs and targets must be equal"
        data_list = []
        data_len = len(xd)
        print('Creating miRNA dataset...')
        
        for i in range(data_len):
            print('Converting to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.smile = torch.LongTensor([z[i]])
            data_list.append(GCNData)
        
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # lncRNA version data processing logic (commented for future activation)
        # def process_lnc(self, xd, xt, y, z, smile_graph):
        #     """lncRNA version data processing"""
        #     assert (len(xd) == len(xt)), "The number of drugs and targets must be equal"
        #     data_list = []
        #     data_len = len(xd)
        #     print('Creating lncRNA dataset...')
        #     
        #     for i in range(data_len):
        #         print('Converting to graph: {}/{}'.format(i+1, data_len))
        #         smiles = xd[i]
        #         target = xt[i]
        #         labels = y[i]
        #         # convert SMILES to molecular representation using rdkit
        #         c_size, features, edge_index = smile_graph[smiles]
        #         # make the graph ready for PyTorch Geometrics GCN algorithms:
        #         GCNData = DATA.Data(x=torch.Tensor(features),
        #                             edge_index=torch.LongTensor(edge_index).transpose(1, 0),
        #                             y=torch.FloatTensor([labels]))
        #         GCNData.target = torch.LongTensor([target])
        #         GCNData.smile = torch.LongTensor([z[i]])
        #         data_list.append(GCNData)
        #     
        #     if self.pre_filter is not None:
        #         data_list = [data for data in data_list if self.pre_filter(data)]
        # 
        #     if self.pre_transform is not None:
        #         data_list = [self.pre_transform(data) for data in data_list]
        #     print('Graph construction done. Saving to file.')
        #     data, slices = self.collate(data_list)
        #     torch.save((data, slices), self.processed_paths[0])

def load_data():
    """Load miRNA data"""
    # miRNA version data loading logic (main code)
    print('Loading miRNA data...')
    
    # Load preprocessed data
    processed_data_file_train = 'processed/train0.pt'
    processed_data_file_test = 'processed/test0.pt'
    
    if os.path.exists(processed_data_file_train) and os.path.exists(processed_data_file_test):
        print('Loading preprocessed data...')
        train_data = TestbedDataset(root='./', dataset='train0')
        test_data = TestbedDataset(root='./', dataset='test0')
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        return train_loader, test_loader
    else:
        print('Preprocessed data not found. Please run process_data.py first.')
        return None, None
        
    # lncRNA version data loading logic (commented for future activation)
    # def load_data_lnc():
    #     """Load lncRNA data"""
    #     print('Loading lncRNA data...')
    #     
    #     # Load preprocessed data
    #     processed_data_file_train = 'processed/train_lnc0.pt'
    #     processed_data_file_test = 'processed/test_lnc0.pt'
    #     
    #     if os.path.exists(processed_data_file_train) and os.path.exists(processed_data_file_test):
    #         print('Loading preprocessed lncRNA data...')
    #         train_data = TestbedDataset(root='./', dataset='train_lnc0')
    #         test_data = TestbedDataset(root='./', dataset='test_lnc0')
    #         
    #         train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    #         test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    #         
    #         return train_loader, test_loader
    #     else:
    #         print('Preprocessed lncRNA data not found. Please run process_data_lnc.py first.')
    #         return None, None

def create_smile_graph(smiles):
    """Create SMILES molecular graph"""
    # miRNA version SMILES graph creation logic (main code)
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0, [], []
    
    # Get atom features
    features = []
    for atom in mol.GetAtoms():
        feature = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetHybridization().real,
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            atom.GetTotalValence()
        ]
        features.append(feature)
    
    # Get edge indices
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # Undirected graph
    
    return len(features), features, edge_index
    
    # lncRNA version SMILES graph creation logic (commented for future activation)
    # def create_smile_graph_lnc(smiles):
    #     """lncRNA version SMILES graph creation"""
    #     from rdkit import Chem
    #     from rdkit.Chem import Descriptors
    #     
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is None:
    #         return 0, [], []
    #     
    #     # Get atom features
    #     features = []
    #     for atom in mol.GetAtoms():
    #         feature = [
    #             atom.GetAtomicNum(),
    #             atom.GetDegree(),
    #             atom.GetFormalCharge(),
    #             atom.GetHybridization().real,
    #             atom.GetIsAromatic(),
    #             atom.GetTotalNumHs(),
    #             atom.GetTotalValence()
    #         ]
    #         features.append(feature)
    #     
    #     # Get edge indices
    #     edge_index = []
    #     for bond in mol.GetBonds():
    #         i = bond.GetBeginAtomIdx()
    #         j = bond.GetEndAtomIdx()
    #         edge_index.append([i, j])
    #         edge_index.append([j, i])  # Undirected graph
    #     
    #     return len(features), features, edge_index

def create_sequence_features(sequence):
    """Create sequence features"""
    # miRNA version sequence feature creation logic (main code)
    # Simple one-hot encoding
    char_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 4}
    features = []
    
    for char in sequence:
        if char in char_to_idx:
            features.append(char_to_idx[char])
        else:
            features.append(5)  # Unknown character
    
    return features
    
    # lncRNA version sequence feature creation logic (commented for future activation)
    # def create_sequence_features_lnc(sequence):
    #     """lncRNA version sequence feature creation"""
    #     # Simple one-hot encoding
    #     char_to_idx = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 4}
    #     features = []
    #     
    #     for char in sequence:
    #         if char in char_to_idx:
    #             features.append(char_to_idx[char])
    #         else:
    #             features.append(5)  # Unknown character
    #     
    #     return features