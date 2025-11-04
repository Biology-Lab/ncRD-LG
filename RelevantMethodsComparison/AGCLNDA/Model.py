from torch import nn
import torch.nn.functional as F
import torch
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time
import torch_sparse
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

init = nn.init.xavier_uniform_

class AGCLNDA(nn.Module):
	def __init__(self, nc_init=None, drug_init=None):
		super(AGCLNDA, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.nc, args.latdim)))
		self.iEmbeds = nn.Parameter(init(torch.empty(args.drug, args.latdim)))
		
		# Use feature initialization if provided
		if nc_init is not None:
			with torch.no_grad():
				self.uEmbeds.copy_(nc_init)
		if drug_init is not None:
			with torch.no_grad():
				self.iEmbeds.copy_(drug_init)
		
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward_gcn(self, adj):
		iniEmbeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds[:args.nc], mainEmbeds[args.nc:]

	def forward_graphcl(self, adj):
		iniEmbeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		return mainEmbeds[:args.nc], mainEmbeds[args.nc:]

	def forward(self, batch_data):
		"""Forward pass"""
		# miRNA version forward pass logic (main code)
		adj = batch_data.adj
		nc_embeds, drug_embeds = self.forward_gcn(adj)
		return nc_embeds, drug_embeds
		
		# lncRNA version forward pass logic (commented for future activation)
		# def forward(self, batch_data, features=None):
		#     """lncRNA version forward pass"""
		#     adj = batch_data.adj
		#     if features is not None:
		#         # Use feature-guided embeddings
		#         proj_features = self.feature_proj(features)
		#         iniEmbeds = proj_features
		#     else:
		#         iniEmbeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)
		#     
		#     embedsLst = [iniEmbeds]
		#     for gcn in self.gcnLayers:
		#         embeds = gcn(adj, embedsLst[-1])
		#         embedsLst.append(embeds)
		#     mainEmbeds = sum(embedsLst)
		#     
		#     return mainEmbeds[:args.nc], mainEmbeds[args.nc:]

	def getGCN(self):
		return self.gcnLayers

	def getEmbeds(self):
		return self.uEmbeds, self.iEmbeds

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		# miRNA version GCN layer logic (main code)
		return torch.sparse.mm(adj, embeds)
		
		# lncRNA version GCN layer logic (commented for future activation)
		# def forward(self, adj, embeds):
		#     """lncRNA version GCN layer"""
		#     return torch.sparse.mm(adj, embeds)

class vgae_encoder(nn.Module):
	def __init__(self, nc_init=None, drug_init=None):
		super(vgae_encoder, self).__init__()
		# Use AGCLNDA as base encoder
		self.base_encoder = AGCLNDA(nc_init, drug_init)
		
		# miRNA version encoder logic (main code)
		self.mu_layer = nn.Linear(args.latdim, args.latdim)
		self.logvar_layer = nn.Linear(args.latdim, args.latdim)
		
		# lncRNA version encoder logic (commented for future activation)
		# def __init__(self, nc_init=None, drug_init=None):
		#     super(vgae_encoder_lnc, self).__init__()
		#     # Use AGCLNDA_lnc as base encoder
		#     self.base_encoder = AGCLNDA_lnc()
		#     
		#     # Feature projection layer
		#     self.feature_proj = nn.Sequential(
		#         nn.Linear(768, args.latdim),
		#         nn.ReLU(),
		#         nn.Dropout(0.1)
		#     )
		#     
		#     self.mu_layer = nn.Linear(args.latdim, args.latdim)
		#     self.logvar_layer = nn.Linear(args.latdim, args.latdim)

	def forward(self, batch_data):
		# miRNA version encoder forward pass (main code)
		nc_embeds, drug_embeds = self.base_encoder(batch_data)
		
		# Calculate mean and variance
		mu = self.mu_layer(nc_embeds)
		logvar = self.logvar_layer(nc_embeds)
		
		# Reparameterization trick
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		z = mu + eps * std
		
		return z, mu, logvar
		
		# lncRNA version encoder forward pass (commented for future activation)
		# def forward(self, batch_data, features=None):
		#     """lncRNA version encoder forward pass"""
		#     nc_embeds, drug_embeds = self.base_encoder(batch_data, features)
		#     
		#     # Calculate mean and variance
		#     mu = self.mu_layer(nc_embeds)
		#     logvar = self.logvar_layer(nc_embeds)
		#     
		#     # Reparameterization trick
		#     std = torch.exp(0.5 * logvar)
		#     eps = torch.randn_like(std)
		#     z = mu + eps * std
		#     
		#     return z, mu, logvar

class vgae_decoder(nn.Module):
	def __init__(self):
		super(vgae_decoder, self).__init__()
		
		# miRNA version decoder logic (main code)
		self.decoder = nn.Sequential(
			nn.Linear(args.latdim, args.latdim * 2),
			nn.ReLU(),
			nn.Linear(args.latdim * 2, args.latdim),
			nn.Sigmoid()
		)
		
		# lncRNA version decoder logic (commented for future activation)
		# def __init__(self):
		#     super(vgae_decoder_lnc, self).__init__()
		#     
		#     self.decoder = nn.Sequential(
		#         nn.Linear(args.latdim, args.latdim * 2),
		#         nn.ReLU(),
		#         nn.Linear(args.latdim * 2, args.latdim),
		#         nn.Sigmoid()
		#     )

	def forward(self, z):
		# miRNA version decoder forward pass (main code)
		reconstructed = self.decoder(z)
		return reconstructed
		
		# lncRNA version decoder forward pass (commented for future activation)
		# def forward(self, z):
		#     """lncRNA version decoder forward pass"""
		#     reconstructed = self.decoder(z)
		#     return reconstructed

class vgae(nn.Module):
	def __init__(self, encoder, decoder):
		super(vgae, self).__init__()
		self.encoder = encoder
		self.decoder = decoder

	def forward(self, batch_data):
		# miRNA version VAE forward pass (main code)
		z, mu, logvar = self.encoder(batch_data)
		reconstructed = self.decoder(z)
		
		# Calculate reconstruction loss
		recon_loss = F.mse_loss(reconstructed, z, reduction='mean')
		
		# Calculate KL divergence loss
		kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		
		return recon_loss + 0.1 * kl_loss
		
		# lncRNA version VAE forward pass (commented for future activation)
		# def forward(self, batch_data, features=None):
		#     """lncRNA version VAE forward pass"""
		#     z, mu, logvar = self.encoder(batch_data, features)
		#     reconstructed = self.decoder(z)
		#     
		#     # Calculate reconstruction loss
		#     recon_loss = F.mse_loss(reconstructed, z, reduction='mean')
		#     
		#     # Calculate KL divergence loss
		#     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		#     
		#     return recon_loss + 0.1 * kl_loss

class DenoisingNet(nn.Module):
	def __init__(self, gcn_layers, embeds):
		super(DenoisingNet, self).__init__()
		self.gcn_layers = gcn_layers
		self.embeds = embeds
		
		# miRNA version denoising network logic (main code)
		self.denoising_layers = nn.Sequential(
			nn.Linear(args.latdim, args.latdim * 2),
			nn.ReLU(),
			nn.Linear(args.latdim * 2, args.latdim),
			nn.Sigmoid()
		)
		
		# lncRNA version denoising network logic (commented for future activation)
		# def __init__(self, gcn_layers, embeds):
		#     super(DenoisingNet_lnc, self).__init__()
		#     self.gcn_layers = gcn_layers
		#     self.embeds = embeds
		#     
		#     self.denoising_layers = nn.Sequential(
		#         nn.Linear(args.latdim, args.latdim * 2),
		#         nn.ReLU(),
		#         nn.Linear(args.latdim * 2, args.latdim),
		#         nn.Sigmoid()
		#     )

	def forward(self, nc_embeds, drug_embeds):
		# miRNA version denoising network forward pass (main code)
		# Denoise embeddings
		denoised_nc = self.denoising_layers(nc_embeds)
		denoised_drug = self.denoising_layers(drug_embeds)
		
		# Calculate denoising loss
		denoising_loss = F.mse_loss(denoised_nc, nc_embeds) + F.mse_loss(denoised_drug, drug_embeds)
		
		return denoising_loss
		
		# lncRNA version denoising network forward pass (commented for future activation)
		# def forward(self, nc_embeds, drug_embeds):
		#     """lncRNA version denoising network forward pass"""
		#     # Denoise embeddings
		#     denoised_nc = self.denoising_layers(nc_embeds)
		#     denoised_drug = self.denoising_layers(drug_embeds)
		#     
		#     # Calculate denoising loss
		#     denoising_loss = F.mse_loss(denoised_nc, nc_embeds) + F.mse_loss(denoised_drug, drug_embeds)
		#     
		#     return denoising_loss

class FeatureProjector(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(FeatureProjector, self).__init__()
		
		# miRNA version feature projector logic (main code)
		self.proj = nn.Sequential(
			nn.Linear(input_dim, output_dim),
			nn.ReLU(),
			nn.Dropout(0.1)
		)
		
		# lncRNA version feature projector logic (commented for future activation)
		# def __init__(self, input_dim, output_dim):
		#     super(FeatureProjector, self).__init__()
		#     
		#     self.proj = nn.Sequential(
		#         nn.Linear(input_dim, output_dim),
		#         nn.ReLU(),
		#         nn.Dropout(0.1)
		#     )

	def forward(self, features):
		# miRNA version feature projector forward pass (main code)
		return self.proj(features)
		
		# lncRNA version feature projector forward pass (commented for future activation)
		# def forward(self, features):
		#     """lncRNA version feature projector forward pass"""
		#     return self.proj(features)