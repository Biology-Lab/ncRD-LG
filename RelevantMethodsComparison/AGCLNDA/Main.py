import torch
import torch.nn.functional as F
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import AGCLNDA, vgae_encoder, vgae_decoder, vgae, DenoisingNet, FeatureProjector
from DataHandler import DataHandler
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict
import os
from copy import deepcopy
import random
from sklearn.metrics import roc_auc_score,average_precision_score
torch.backends.cudnn.enable = True
torch.backends.cudnn.benchmark = True

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('nc', args.nc, 'drug', args.drug)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		best_auc = 0
		best_aupr = 0
		stloc = 0
		log('Model Initialized')

		for ep in range(stloc, args.epoch):
			temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch(temperature)
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				y_true, y_pred, pred_score = self.testEpoch()
				auc = roc_auc_score(y_true, y_pred)
				aupr = average_precision_score(y_true, y_pred)
				reses = {'AUC': auc, 'AUPR': aupr}
				log(self.makePrint('Test', ep, reses, tstFlag))
				if auc > best_auc:
					best_auc = auc
					best_aupr = aupr
					log('Best AUC: %.4f, Best AUPR: %.4f' % (best_auc, best_aupr))
					self.saveModel()
		log('Best AUC: %.4f, Best AUPR: %.4f' % (best_auc, best_aupr))

	def prepareModel(self):
		# Main model
		self.model = AGCLNDA(args.nc, args.drug).cuda()

		# Generator 1: VGAE
		encoder = vgae_encoder(args.nc, args.drug).cuda()
		decoder = vgae_decoder().cuda()
		self.generator_1 = vgae(encoder, decoder).cuda()
		
		# Generator 2: DenoisingNet
		self.generator_2 = DenoisingNet(self.model.getGCN(), self.model.getEmbeds()).cuda()

		# Optimizers
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.reg)
		self.opt_g1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=args.reg)
		self.opt_g2 = torch.optim.Adam(self.generator_2.parameters(), lr=args.lr, weight_decay=args.reg)

		# lncRNA version model preparation (commented for future activation)
		# def prepareModel(self):
		#     """Prepare lncRNA model with GPU availability check"""
		#     if torch.cuda.is_available() and args.gpu >= 0:
		#         if args.gpu >= torch.cuda.device_count():
		#             args.gpu = 0
		#             print(f"GPU {args.gpu} not available, using GPU 0")
		#         torch.cuda.set_device(args.gpu)
		#         print(f"Using GPU {args.gpu}")
		#     else:
		#         print("Using CPU")
		#         args.gpu = -1
		#     
		#     self.model = AGCLNDA_lnc(args.nc, args.drug).cuda() if args.gpu >= 0 else AGCLNDA_lnc(args.nc, args.drug)
		#     encoder = vgae_encoder_lnc(args.nc, args.drug).cuda() if args.gpu >= 0 else vgae_encoder_lnc(args.nc, args.drug)
		#     decoder = vgae_decoder_lnc().cuda() if args.gpu >= 0 else vgae_decoder_lnc()
		#     self.generator_1 = vgae_lnc(encoder, decoder).cuda() if args.gpu >= 0 else vgae_lnc(encoder, decoder)
		#     self.generator_2 = DenoisingNet_lnc(self.model.getGCN(), self.model.getEmbeds()).cuda() if args.gpu >= 0 else DenoisingNet_lnc(self.model.getGCN(), self.model.getEmbeds())
		#     
		#     self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.reg)
		#     self.opt_g1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=args.reg)
		#     self.opt_g2 = torch.optim.Adam(self.generator_2.parameters(), lr=args.lr, weight_decay=args.reg)

	def trainEpoch(self, temperature):
		self.model.train()
		self.generator_1.train()
		self.generator_2.train()
		
		epoch_loss = 0
		epoch_ssl_loss = 0
		epoch_ib_loss = 0
		epoch_reg_loss = 0
		
		for batch_idx, (batch_data, batch_labels) in enumerate(self.handler.trnLoader):
			batch_data = batch_data.cuda()
			batch_labels = batch_labels.cuda()
			
			# Forward pass
			nc_embeds, drug_embeds = self.model(batch_data)
			
			# Main loss
			main_loss = F.binary_cross_entropy_with_logits(
				pairPredict(nc_embeds, drug_embeds), 
				batch_labels.float()
			)
			
			# Contrastive learning loss
			ssl_loss = self.contrastiveLoss(nc_embeds, drug_embeds, temperature)
			
			# Information bottleneck loss
			ib_loss = self.informationBottleneckLoss(nc_embeds, drug_embeds)
			
			# Regularization loss
			reg_loss = calcRegLoss(self.model)
			
			# Total loss
			total_loss = main_loss + args.ssl_reg * ssl_loss + args.ib_reg * ib_loss + reg_loss
			
			# Backward pass
			self.opt.zero_grad()
			total_loss.backward()
			self.opt.step()
			
			# Train generators
			self.trainGenerators(batch_data, nc_embeds, drug_embeds)
			
			# Statistics
			epoch_loss += main_loss.item()
			epoch_ssl_loss += ssl_loss.item()
			epoch_ib_loss += ib_loss.item()
			epoch_reg_loss += reg_loss.item()
		
		return {
			'Loss': epoch_loss / len(self.handler.trnLoader),
			'SSL': epoch_ssl_loss / len(self.handler.trnLoader),
			'IB': epoch_ib_loss / len(self.handler.trnLoader),
			'Reg': epoch_reg_loss / len(self.handler.trnLoader)
		}

	def contrastiveLoss(self, nc_embeds, drug_embeds, temperature):
		"""Compute contrastive learning loss"""
		# miRNA version contrastive learning (main code)
		sim_matrix = torch.mm(nc_embeds, drug_embeds.t()) / temperature
		
		# Create positive sample mask
		batch_size = nc_embeds.size(0)
		pos_mask = torch.eye(batch_size).cuda()
		
		# Compute contrastive loss
		exp_sim = torch.exp(sim_matrix)
		pos_sim = exp_sim * pos_mask
		neg_sim = exp_sim * (1 - pos_mask)
		
		pos_loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
		
		return pos_loss.mean()
		
		# lncRNA version contrastive learning (commented for future activation)
		# def contrastiveLoss(self, nc_embeds, drug_embeds, temperature):
		#     """lncRNA version contrastive learning loss"""
		#     sim_matrix = torch.mm(nc_embeds, drug_embeds.t()) / temperature
		#     batch_size = nc_embeds.size(0)
		#     pos_mask = torch.eye(batch_size).cuda() if args.gpu >= 0 else torch.eye(batch_size)
		#     exp_sim = torch.exp(sim_matrix)
		#     pos_sim = exp_sim * pos_mask
		#     neg_sim = exp_sim * (1 - pos_mask)
		#     pos_loss = -torch.log(pos_sim.sum(dim=1) / (pos_sim.sum(dim=1) + neg_sim.sum(dim=1) + 1e-8))
		#     return pos_loss.mean()

	def informationBottleneckLoss(self, nc_embeds, drug_embeds):
		"""Compute information bottleneck loss"""
		# miRNA version information bottleneck (main code)
		mi_loss = torch.mean(torch.norm(nc_embeds, p=2, dim=1)) + torch.mean(torch.norm(drug_embeds, p=2, dim=1))
		return mi_loss
		
		# lncRNA version information bottleneck (commented for future activation)
		# def informationBottleneckLoss(self, nc_embeds, drug_embeds):
		#     """lncRNA version information bottleneck loss"""
		#     mi_loss = torch.mean(torch.norm(nc_embeds, p=2, dim=1)) + torch.mean(torch.norm(drug_embeds, p=2, dim=1))
		#     return mi_loss

	def trainGenerators(self, batch_data, nc_embeds, drug_embeds):
		"""Train generator networks"""
		# miRNA version generator training (main code)
		# Train generator 1 (VGAE)
		self.opt_g1.zero_grad()
		recon_loss_1 = self.generator_1(batch_data)
		recon_loss_1.backward()
		self.opt_g1.step()
		
		# Train generator 2 (DenoisingNet)
		self.opt_g2.zero_grad()
		recon_loss_2 = self.generator_2(nc_embeds, drug_embeds)
		recon_loss_2.backward()
		self.opt_g2.step()
		
		# lncRNA version generator training (commented for future activation)
		# def trainGenerators(self, batch_data, nc_embeds, drug_embeds):
		#     """lncRNA version generator training"""
		#     self.opt_g1.zero_grad()
		#     recon_loss_1 = self.generator_1(batch_data)
		#     recon_loss_1.backward()
		#     self.opt_g1.step()
		#     
		#     self.opt_g2.zero_grad()
		#     recon_loss_2 = self.generator_2(nc_embeds, drug_embeds)
		#     recon_loss_2.backward()
		#     self.opt_g2.step()

	def testEpoch(self):
		self.model.eval()
		all_preds = []
		all_labels = []
		all_scores = []
		
		with torch.no_grad():
			for batch_data, batch_labels in self.handler.tstLoader:
				batch_data = batch_data.cuda()
				batch_labels = batch_labels.cuda()
				
				nc_embeds, drug_embeds = self.model(batch_data)
				pred_scores = torch.sigmoid(pairPredict(nc_embeds, drug_embeds))
				preds = (pred_scores > 0.5).float()
				
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(batch_labels.cpu().numpy())
				all_scores.extend(pred_scores.cpu().numpy())
		
		return all_labels, all_preds, all_scores

	def saveModel(self):
		"""Save model checkpoint"""
		model_path = f'best_model_epoch_{args.epoch}.pth'
		torch.save({
			'model_state_dict': self.model.state_dict(),
			'generator_1_state_dict': self.generator_1.state_dict(),
			'generator_2_state_dict': self.generator_2.state_dict(),
			'optimizer_state_dict': self.opt.state_dict(),
			'args': args
		}, model_path)
		log(f'Model saved to {model_path}')

if __name__ == '__main__':
	handler = DataHandler()
	coach = Coach(handler)
	coach.run()