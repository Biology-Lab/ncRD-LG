import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import csv
import re
import os

class DataHandler:
	def __init__(self):
		self.data_path = args.data_path
		self.pair_csv = f'{self.data_path}MiPair.csv'
		self.rna_feat_csv = f'{self.data_path}MiRNA_features.csv'
		self.drug_feat_csv = f'{self.data_path}MiDrug_features.csv'
		self.test_ratio = 0.2

	def _normalize_name(self, name):

		if not name:
			return ""
		name = name.strip().upper()
		name = ' '.join(name.split())
		name = re.sub(r'\([^)]*\)', '', name)
		name = re.sub(r'\[[^\]]*\]', '', name)
		name = re.sub(r'[^\w\s]', '', name)
		name = ' '.join(name.split())
		return name

	def _read_pairs(self, path):
		"""Read interaction pairs"""
		nc_names = []
		drug_names = []
		pairs = []
		
		with open(path, 'r', encoding='utf-8') as f:
			reader = csv.reader(f)
			head = next(reader, None)
			
			for row in reader:
				if len(row) < 3:
					continue
				nc = row[0].strip()
				drug = row[2].strip()
				if nc == '' or drug == '':
					continue
				nc_names.append(nc)
				drug_names.append(drug)
				pairs.append((nc, drug))
		
		return nc_names, drug_names, pairs

	def _read_features(self, path):

		features = {}
		with open(path, 'r', encoding='utf-8') as f:
			reader = csv.reader(f)
			head = next(reader, None)
			
			for row in reader:
				if len(row) < 2:
					continue
				name = row[0].strip()
				feat_str = row[1].strip()
				if name == '' or feat_str == '':
					continue
				
					try:
					feat = [float(x) for x in feat_str.split(',')]
					features[name] = feat
					except:
						continue
				
		return features

	def loadData(self):


		log('Loading miRNA data...')
		

		nc_names, drug_names, pairs = self._read_pairs(self.pair_csv)
		log(f'Loaded {len(pairs)} pairs')
		

		nc_features = self._read_features(self.rna_feat_csv)
		drug_features = self._read_features(self.drug_feat_csv)
		log(f'Loaded {len(nc_features)} ncRNA features, {len(drug_features)} drug features')
		

		nc_names = list(set(nc_names))
		drug_names = list(set(drug_names))
		
		nc_name2id = {name: i for i, name in enumerate(nc_names)}
		drug_name2id = {name: i for i, name in enumerate(drug_names)}
		

		interaction_matrix = dok_matrix((len(nc_names), len(drug_names)), dtype=np.float32)
		for nc, drug in pairs:
			if nc in nc_name2id and drug in drug_name2id:
				interaction_matrix[nc_name2id[nc], drug_name2id[drug]] = 1.0
		

		interaction_matrix = interaction_matrix.tocsr()
		

		nc_feat_matrix = np.zeros((len(nc_names), 128), dtype=np.float32)
		drug_feat_matrix = np.zeros((len(drug_names), 128), dtype=np.float32)
		
		for i, name in enumerate(nc_names):
			if name in nc_features:
				feat = nc_features[name]
				if len(feat) >= 128:
					nc_feat_matrix[i] = feat[:128]
				else:
					nc_feat_matrix[i, :len(feat)] = feat
		
		for i, name in enumerate(drug_names):
			if name in drug_features:
				feat = drug_features[name]
				if len(feat) >= 128:
					drug_feat_matrix[i] = feat[:128]
			else:
					drug_feat_matrix[i, :len(feat)] = feat
		

		adj_matrix = self._create_adjacency_matrix(interaction_matrix)
		

		train_data, test_data = self._split_data(interaction_matrix, adj_matrix)

		self.trnLoader = self._create_dataloader(train_data, batch_size=args.batch)
		self.tstLoader = self._create_dataloader(test_data, batch_size=args.tstBat)
		
		log('Data loading completed')
		
		#
		# def loadData(self):
		#
		#     log('Loading lncRNA data...')
		#     
		#
		#     self.check_data_files()
		#     
		#
		#     lncrna_data = self.load_torch_data('LncRNA.pt')
		#     drug_data = self.load_torch_data('LncDrug.pt')
		#     
		#
		#     train_edge_index = self.load_torch_data('LncTrain_edge_index.pt')
		#     train_edge_label = self.load_torch_data('LncTrain_edge_label.pt')
		#     
		#
		#     val_edge_index = self.load_torch_data('LncVal_edge_index.pt')
		#     val_edge_label = self.load_torch_data('LncVal_edge_label.pt')
		#     
		#
		#     test_edge_index = self.load_torch_data('LncTest_edge_index.pt')
		#     test_edge_label = self.load_torch_data('LncTest_edge_label.pt')
		#     
		#
		#     pair_data = self.load_numpy_data('LncPair_new.npy')
		#     
		#
		#     adj_matrix = self._create_adjacency_matrix_lnc(lncrna_data, drug_data, pair_data)
		#     
		#
		#     self.trnLoader = self._create_dataloader_lnc(train_edge_index, train_edge_label, batch_size=args.batch)
		#     self.tstLoader = self._create_dataloader_lnc(test_edge_index, test_edge_label, batch_size=args.tstBat)
		#     
		#     log('lncRNA data loading completed')

	def _create_adjacency_matrix(self, interaction_matrix):


		nc_num, drug_num = interaction_matrix.shape
		

		adj_matrix = dok_matrix((nc_num + drug_num, nc_num + drug_num), dtype=np.float32)
		

		for i in range(nc_num):
			for j in range(drug_num):
				if interaction_matrix[i, j] > 0:
					adj_matrix[i, nc_num + j] = 1.0
					adj_matrix[nc_num + j, i] = 1.0
		

		for i in range(nc_num + drug_num):
			adj_matrix[i, i] = 1.0
		
		return adj_matrix.tocsr()
		

		# def _create_adjacency_matrix_lnc(self, lncrna_data, drug_data, pair_data):
		#     """lncRNA version adjacency matrix creation"""
		#     nc_num = len(lncrna_data)
		#     drug_num = len(drug_data)
		#     
		#
		#     adj_matrix = dok_matrix((nc_num + drug_num, nc_num + drug_num), dtype=np.float32)
		#     
		#
		#     for pair in pair_data:
		#         nc_idx, drug_idx = pair
		#         if nc_idx < nc_num and drug_idx < drug_num:
		#             adj_matrix[nc_idx, nc_num + drug_idx] = 1.0
		#             adj_matrix[nc_num + drug_idx, nc_idx] = 1.0
		#     
		#
		#     for i in range(nc_num + drug_num):
		#         adj_matrix[i, i] = 1.0
		#     
		#     return adj_matrix.tocsr()

	def _split_data(self, interaction_matrix, adj_matrix):


		nc_num, drug_num = interaction_matrix.shape

		pos_pairs = []
		for i in range(nc_num):
			for j in range(drug_num):
				if interaction_matrix[i, j] > 0:
					pos_pairs.append((i, j))
		

		np.random.shuffle(pos_pairs)
		split_idx = int(len(pos_pairs) * (1 - self.test_ratio))
		
		train_pairs = pos_pairs[:split_idx]
		test_pairs = pos_pairs[split_idx:]
		

		train_matrix = dok_matrix((nc_num, drug_num), dtype=np.float32)
		test_matrix = dok_matrix((nc_num, drug_num), dtype=np.float32)
		
		for i, j in train_pairs:
			train_matrix[i, j] = 1.0
		
		for i, j in test_pairs:
			test_matrix[i, j] = 1.0
		
		return train_matrix.tocsr(), test_matrix.tocsr()
		

		# def _split_data_lnc(self, interaction_matrix, adj_matrix):
		#     """lncRNA version data splitting"""
		#     nc_num, drug_num = interaction_matrix.shape
		#     
		#
		#     pos_pairs = []
		#     for i in range(nc_num):
		#         for j in range(drug_num):
		#             if interaction_matrix[i, j] > 0:
		#                 pos_pairs.append((i, j))
		#     
		#
		#     np.random.shuffle(pos_pairs)
		#     split_idx = int(len(pos_pairs) * (1 - self.test_ratio))
		#     
		#     train_pairs = pos_pairs[:split_idx]
		#     test_pairs = pos_pairs[split_idx:]
		#     
		#
		#     train_matrix = dok_matrix((nc_num, drug_num), dtype=np.float32)
		#     test_matrix = dok_matrix((nc_num, drug_num), dtype=np.float32)
		#     
		#     for i, j in train_pairs:
		#         train_matrix[i, j] = 1.0
		#     
		#     for i, j in test_pairs:
		#         test_matrix[i, j] = 1.0
		#     
		#     return train_matrix.tocsr(), test_matrix.tocsr()

	def _create_dataloader(self, data_matrix, batch_size):


		class InteractionDataset(data.Dataset):
			def __init__(self, matrix, adj_matrix):
				self.matrix = matrix
				self.adj_matrix = adj_matrix
				self.samples = []
				

				nc_num, drug_num = matrix.shape
				for i in range(nc_num):
					for j in range(drug_num):
						if matrix[i, j] > 0:
							self.samples.append((i, j, 1))
				

				neg_samples = []
				for i in range(nc_num):
					for j in range(drug_num):
						if matrix[i, j] == 0:
							neg_samples.append((i, j, 0))
				

				np.random.shuffle(neg_samples)
				neg_samples = neg_samples[:len(self.samples)]
				self.samples.extend(neg_samples)
				
				np.random.shuffle(self.samples)

    def __len__(self):
				return len(self.samples)

    def __getitem__(self, idx):
				nc_idx, drug_idx, label = self.samples[idx]
				return (nc_idx, drug_idx), label
		
		dataset = InteractionDataset(data_matrix, self.adj_matrix)
		return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
		
		# lnc
		# def _create_dataloader_lnc(self, edge_index, edge_label, batch_size):
		#
		#     class LncRNADataset(data.Dataset):
		#         def __init__(self, edge_index, edge_label):
		#             self.edge_index = edge_index
		#             self.edge_label = edge_label
		#             self.samples = []
		#             
		#             for i in range(len(edge_index[0])):
		#                 nc_idx = edge_index[0][i]
		#                 drug_idx = edge_index[1][i]
		#                 label = edge_label[i]
		#                 self.samples.append((nc_idx, drug_idx, label))
		#             
		#             np.random.shuffle(self.samples)
		#         
		#         def __len__(self):
		#             return len(self.samples)
		#         
		#         def __getitem__(self, idx):
		#             nc_idx, drug_idx, label = self.samples[idx]
		#             return (nc_idx, drug_idx), label
		#     
		#     dataset = LncRNADataset(edge_index, edge_label)
		#     return data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	def normalizeAdj(self, mat):


		degree = np.array(mat.sum(axis=1)).flatten()
		degree[degree == 0] = 1
		degree = 1.0 / degree
		degree = sp.diags(degree)
		return degree.dot(mat)
		

		# def normalizeAdj(self, mat):
		#     
		#     degree = np.array(mat.sum(axis=1)).flatten()
		#     degree[degree == 0] = 1
		#     degree = 1.0 / degree
		#     degree = sp.diags(degree)
		#     return degree.dot(mat)