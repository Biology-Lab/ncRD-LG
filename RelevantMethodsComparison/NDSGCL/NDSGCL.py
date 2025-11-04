import torch.nn as nn
import faiss
import sys
sys.path.append('../code')

from util import *


class NDSGCL(object):
    def __init__(self, conf, training_set, test_set, i):
        super(NDSGCL, self).__init__()
        self.config = conf
        self.emb_size = int(self.config['embbedding.size'])
        args = OptionConf(self.config['NCL'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_temp = float(args['-tau'])
        self.ssl_reg = float(args['-ssl_reg'])
        self.hyper_layers = int(args['-hyper_layers'])
        self.alpha = float(args['-alpha'])
        self.proto_reg = float(args['-proto_reg'])
        self.k = int(args['-num_clusters'])
        self.data = Interaction(conf, training_set, test_set)
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)
        self.lRate = float(self.config['learnRate'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.batch_size = int(self.config['batch_size'])
        self.reg = float(self.config['reg.lambda'])
        self.lncRNA_centroids = None
        self.lncRNA_2cluster = None
        self.drug_centroids = None
        self.drug_2cluster = None
        self.i = i

    def e_step(self):
        # miRNA version E-step logic (main code)
        lncRNA_embeddings = self.model.embedding_dict['lncRNA_emb'].detach().cpu().numpy()
        drug_embeddings = self.model.embedding_dict['drug_emb'].detach().cpu().numpy()
        self.lncRNA_centroids, self.lncRNA_2cluster = self.run_kmeans(lncRNA_embeddings)
        self.drug_centroids, self.drug_2cluster = self.run_kmeans(drug_embeddings)
        
        # lncRNA version E-step logic (commented for future activation)
        # def e_step_lnc(self):
        #     """lncRNA version E-step"""
        #     lncRNA_embeddings = self.model.embedding_dict['lncRNA_emb'].detach().cpu().numpy()
        #     drug_embeddings = self.model.embedding_dict['drug_emb'].detach().cpu().numpy()
        #     self.lncRNA_centroids, self.lncRNA_2cluster = self.run_kmeans(lncRNA_embeddings)
        #     self.drug_centroids, self.drug_2cluster = self.run_kmeans(drug_embeddings)

    def run_kmeans(self, x):
        # miRNA version K-means clustering logic (main code)
        kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        kmeans.train(x)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x, 1)
        centroids = torch.Tensor(cluster_cents).cuda()
        node2cluster = torch.LongTensor(I).squeeze().cuda()
        return centroids, node2cluster
        
        # lncRNA version K-means clustering logic (commented for future activation)
        # def run_kmeans_lnc(self, x):
        #     """lncRNA version K-means clustering"""
        #     kmeans = faiss.Kmeans(d=self.emb_size, k=self.k, gpu=True)
        #     kmeans.train(x)
        #     cluster_cents = kmeans.centroids
        #     _, I = kmeans.index.search(x, 1)
        #     centroids = torch.Tensor(cluster_cents).cuda()
        #     node2cluster = torch.LongTensor(I).squeeze().cuda()
        #     return centroids, node2cluster

    def ProtoNCE_loss(self, initial_emb, lncRNA_idx, drug_idx):
        # miRNA version prototype contrastive learning loss logic (main code)
        lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        lncRNA_emb = lncRNA_emb[lncRNA_idx]
        drug_emb = drug_emb[drug_idx]
        
        # Calculate prototype contrastive learning loss
        lncRNA_proto_loss = self.proto_loss(lncRNA_emb, self.lncRNA_centroids, self.lncRNA_2cluster[lncRNA_idx])
        drug_proto_loss = self.proto_loss(drug_emb, self.drug_centroids, self.drug_2cluster[drug_idx])
        
        return lncRNA_proto_loss + drug_proto_loss
        
        # lncRNA version prototype contrastive learning loss logic (commented for future activation)
        # def ProtoNCE_loss_lnc(self, initial_emb, lncRNA_idx, drug_idx):
        #     """lncRNA version prototype contrastive learning loss"""
        #     lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        #     lncRNA_emb = lncRNA_emb[lncRNA_idx]
        #     drug_emb = drug_emb[drug_idx]
        #     
        #     # Calculate prototype contrastive learning loss
        #     lncRNA_proto_loss = self.proto_loss(lncRNA_emb, self.lncRNA_centroids, self.lncRNA_2cluster[lncRNA_idx])
        #     drug_proto_loss = self.proto_loss(drug_emb, self.drug_centroids, self.drug_2cluster[drug_idx])
        #     
        #     return lncRNA_proto_loss + drug_proto_loss

    def proto_loss(self, emb, centroids, cluster_ids):
        # miRNA version prototype loss calculation logic (main code)
        # Calculate similarity between embeddings and prototypes
        sim = torch.mm(emb, centroids.t()) / self.ssl_temp
        sim = torch.exp(sim)
        
        # Calculate contrastive loss
        pos_sim = sim[range(len(emb)), cluster_ids]
        neg_sim = sim.sum(dim=1) - pos_sim
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        return loss.mean()
        
        # lncRNA version prototype loss calculation logic (commented for future activation)
        # def proto_loss_lnc(self, emb, centroids, cluster_ids):
        #     """lncRNA version prototype loss calculation"""
        #     # Calculate similarity between embeddings and prototypes
        #     sim = torch.mm(emb, centroids.t()) / self.ssl_temp
        #     sim = torch.exp(sim)
        #     
        #     # Calculate contrastive loss
        #     pos_sim = sim[range(len(emb)), cluster_ids]
        #     neg_sim = sim.sum(dim=1) - pos_sim
        #     
        #     loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        #     return loss.mean()

    def train_one_epoch(self, training_data, negative_ratio, interaction_table):
        # miRNA version train one epoch logic (main code)
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in self.get_batches(training_data, self.batch_size):
            # Forward pass
            lncRNA_idx, drug_idx, labels = batch
            initial_emb = self.model.forward()
            
            # Calculate main loss
            main_loss = self.calculate_main_loss(initial_emb, lncRNA_idx, drug_idx, labels)
            
            # Calculate contrastive learning loss
            ssl_loss = self.calculate_ssl_loss(initial_emb, lncRNA_idx, drug_idx)
            
            # Calculate prototype contrastive learning loss
            proto_loss = self.ProtoNCE_loss(initial_emb, lncRNA_idx, drug_idx)
            
            # Total loss
            total_loss_batch = main_loss + self.ssl_reg * ssl_loss + self.proto_reg * proto_loss
            
            # Backward pass
            self.model.optimizer.zero_grad()
            total_loss_batch.backward()
            self.model.optimizer.step()
            
            total_loss += total_loss_batch.item()
            batch_count += 1
        
        return total_loss / batch_count
        
        # lncRNA version train one epoch logic (commented for future activation)
        # def train_one_epoch_lnc(self, training_data, negative_ratio, interaction_table):
        #     """lncRNA version train one epoch"""
        #     self.model.train()
        #     total_loss = 0
        #     batch_count = 0
        #     
        #     for batch in self.get_batches(training_data, self.batch_size):
        #         # Forward pass
        #         lncRNA_idx, drug_idx, labels = batch
        #         initial_emb = self.model.forward()
        #         
        #         # Calculate main loss
        #         main_loss = self.calculate_main_loss(initial_emb, lncRNA_idx, drug_idx, labels)
        #         
        #         # Calculate contrastive learning loss
        #         ssl_loss = self.calculate_ssl_loss(initial_emb, lncRNA_idx, drug_idx)
        #         
        #         # Calculate prototype contrastive learning loss
        #         proto_loss = self.ProtoNCE_loss(initial_emb, lncRNA_idx, drug_idx)
        #         
        #         # Total loss
        #         total_loss_batch = main_loss + self.ssl_reg * ssl_loss + self.proto_reg * proto_loss
        #         
        #         # Backward pass
        #         self.model.optimizer.zero_grad()
        #         total_loss_batch.backward()
        #         self.model.optimizer.step()
        #         
        #         total_loss += total_loss_batch.item()
        #         batch_count += 1
        #     
        #     return total_loss / batch_count

    def train_one_epoch_hard_negative(self, training_data, negative_ratio, interaction_table, negative_manager, all_drugs):
        # miRNA version hard negative mining training logic (main code)
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in self.get_batches(training_data, self.batch_size):
            # Forward pass
            lncRNA_idx, drug_idx, labels = batch
            initial_emb = self.model.forward()
            
            # Calculate main loss
            main_loss = self.calculate_main_loss(initial_emb, lncRNA_idx, drug_idx, labels)
            
            # Calculate contrastive learning loss
            ssl_loss = self.calculate_ssl_loss(initial_emb, lncRNA_idx, drug_idx)
            
            # Calculate prototype contrastive learning loss
            proto_loss = self.ProtoNCE_loss(initial_emb, lncRNA_idx, drug_idx)
            
            # Calculate hard negative mining loss
            hard_neg_loss = self.calculate_hard_negative_loss(initial_emb, lncRNA_idx, drug_idx, negative_manager, all_drugs)
            
            # Total loss
            total_loss_batch = main_loss + self.ssl_reg * ssl_loss + self.proto_reg * proto_loss + 0.1 * hard_neg_loss
            
            # Backward pass
            self.model.optimizer.zero_grad()
            total_loss_batch.backward()
            self.model.optimizer.step()
            
            total_loss += total_loss_batch.item()
            batch_count += 1
        
        return total_loss / batch_count
        
        # lncRNA version hard negative mining training logic (commented for future activation)
        # def train_one_epoch_hard_negative_lnc(self, training_data, negative_ratio, interaction_table, negative_manager, all_drugs):
        #     """lncRNA version hard negative mining training"""
        #     self.model.train()
        #     total_loss = 0
        #     batch_count = 0
        #     
        #     for batch in self.get_batches(training_data, self.batch_size):
        #         # Forward pass
        #         lncRNA_idx, drug_idx, labels = batch
        #         initial_emb = self.model.forward()
        #         
        #         # Calculate main loss
        #         main_loss = self.calculate_main_loss(initial_emb, lncRNA_idx, drug_idx, labels)
        #         
        #         # Calculate contrastive learning loss
        #         ssl_loss = self.calculate_ssl_loss(initial_emb, lncRNA_idx, drug_idx)
        #         
        #         # Calculate prototype contrastive learning loss
        #         proto_loss = self.ProtoNCE_loss(initial_emb, lncRNA_idx, drug_idx)
        #         
        #         # Calculate hard negative mining loss
        #         hard_neg_loss = self.calculate_hard_negative_loss(initial_emb, lncRNA_idx, drug_idx, negative_manager, all_drugs)
        #         
        #         # Total loss
        #         total_loss_batch = main_loss + self.ssl_reg * ssl_loss + self.proto_reg * proto_loss + 0.1 * hard_neg_loss
        #         
        #         # Backward pass
        #         self.model.optimizer.zero_grad()
        #         total_loss_batch.backward()
        #         self.model.optimizer.step()
        #         
        #         total_loss += total_loss_batch.item()
        #         batch_count += 1
        #     
        #     return total_loss / batch_count

    def calculate_main_loss(self, initial_emb, lncRNA_idx, drug_idx, labels):
        # miRNA version main loss calculation logic (main code)
        lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        lncRNA_emb = lncRNA_emb[lncRNA_idx]
        drug_emb = drug_emb[drug_idx]
        
        # Calculate prediction scores
        scores = torch.sum(lncRNA_emb * drug_emb, dim=1)
        
        # Calculate binary cross entropy loss
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        return loss
        

    def calculate_ssl_loss(self, initial_emb, lncRNA_idx, drug_idx):
        lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        lncRNA_emb = lncRNA_emb[lncRNA_idx]
        drug_emb = drug_emb[drug_idx]
        
        sim = torch.mm(lncRNA_emb, drug_emb.t()) / self.ssl_temp
        sim = torch.exp(sim)
        
        pos_sim = torch.diag(sim)
        
        neg_sim = sim.sum(dim=1) - pos_sim
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        return loss.mean()
        

    def calculate_hard_negative_loss(self, initial_emb, lncRNA_idx, drug_idx, negative_manager, all_drugs):
        lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
        lncRNA_emb = lncRNA_emb[lncRNA_idx]
        drug_emb = drug_emb[drug_idx]
        
        hard_neg_drugs = negative_manager.get_hard_negatives(lncRNA_idx, drug_idx, all_drugs)
        
        hard_neg_emb = drug_emb[hard_neg_drugs]
        hard_neg_scores = torch.sum(lncRNA_emb * hard_neg_emb, dim=1)
        
        margin = 1.0
        loss = F.relu(margin - hard_neg_scores).mean()
        return loss
        

    def get_batches(self, data, batch_size):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            lncRNA_idx = [item[0] for item in batch]
            drug_idx = [item[1] for item in batch]
            labels = [item[2] for item in batch]
            yield lncRNA_idx, drug_idx, labels
            

    def predict(self, lncRNA_id, drug_id):
        self.model.eval()
        with torch.no_grad():
            initial_emb = self.model.forward()
            lncRNA_emb, drug_emb = torch.split(initial_emb, [self.data.lncRNA_num, self.data.drug_num])
            lncRNA_emb = lncRNA_emb[lncRNA_id]
            drug_emb = drug_emb[drug_id]
            score = torch.sum(lncRNA_emb * drug_emb)
            return torch.sigmoid(score).item()
            

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'lncRNA_centroids': self.lncRNA_centroids,
            'drug_centroids': self.drug_centroids
        }, path)
        