"""
Negative sample manager module
Ensures diversity of negative samples, avoids selecting same hard negatives in consecutive rounds
"""

import numpy as np

class NegativeSampleManager:
    """Negative sample manager ensuring diversity"""
    
    def __init__(self, diversity_ratio=0.3):
        """
        
        Args:
            diversity_ratio: Ratio of old negative samples to retain, default 0.3 (30%)
        """
        self.diversity_ratio = diversity_ratio
        self.previous_negatives = {}  # Store previous round negative samples
        print(f"Negative sample manager initialized, diversity ratio: {diversity_ratio}")
    
    def get_diverse_negatives(self, lncRNA_id, pos_drug_id, model, 
                            interaction_table, all_drugs, num_negatives=2):
        """
        Get diverse negative samples
        
        Args:
            lncRNA_id: miRNA ID
            pos_drug_id: Positive sample drug ID
            model: Current training model
            interaction_table: Interaction record table
            all_drugs: All drug list
            num_negatives: Number of negative samples to generate
            
        Returns:
            list: Negative sample drug ID list
        """
        try:
            new_negatives = []
            new_count = int(num_negatives * (1 - self.diversity_ratio))
            
            for _ in range(new_count):
                neg_drug_id = self._hard_negative_mining(lncRNA_id, pos_drug_id, model, 
                                                       interaction_table, all_drugs)
                new_negatives.append(neg_drug_id)
            
            old_negatives = []
            old_count = int(num_negatives * self.diversity_ratio)
            
            if lncRNA_id in self.previous_negatives and len(self.previous_negatives[lncRNA_id]) > 0:
                old_negatives = self.previous_negatives[lncRNA_id][:old_count]
            
            all_negatives = new_negatives + old_negatives
            self.previous_negatives[lncRNA_id] = all_negatives
            
            return all_negatives
            
        except Exception as e:
            print(f"Error getting diverse negative samples: {e}")
            # Use random sampling when error occurs
            return self._random_negative_sampling(pos_drug_id, all_drugs, num_negatives)
    
    def _hard_negative_mining(self, miRNA_id, pos_drug_id, model, 
                            interaction_table, all_drugs, k_ratio=0.2):
        """
        Hard Negative Mining based on model predictions
        
        Args:
            miRNA_id: miRNA ID
            pos_drug_id: Positive sample drug ID
            model: Current training model
            interaction_table: Interaction record table
            all_drugs: All drug list
            k_ratio: Select top k% drugs as hard negative
            
        Returns:
            int: Selected negative sample drug ID
        """
        try:
            # Check if model is trained (has miRNA_emb attribute)
            if not hasattr(model, 'miRNA_emb'):
                print("Model not fully trained, using random negative sampling")
                return self._random_negative_sampling(pos_drug_id, all_drugs, 1)[0]
            
            all_scores = model.predict(miRNA_id)
            if not isinstance(all_scores, np.ndarray) or len(all_scores) == 0:
                return self._random_negative_sampling(pos_drug_id, all_drugs, 1)[0]
            
            non_interaction_drugs = []
            non_interaction_scores = []
            
            for drug_id in all_drugs:
                if drug_id != pos_drug_id:  # Exclude positive sample drug
                    # Check if no real interaction
                    if not interaction_table.has_interaction(miRNA_id, drug_id):
                        drug_idx = model.data.drug.get(drug_id, -1)
                        if 0 <= drug_idx < len(all_scores):
                            non_interaction_drugs.append(drug_id)
                            non_interaction_scores.append(all_scores[drug_idx])
            
            if not non_interaction_drugs:
                return self._random_negative_sampling(pos_drug_id, all_drugs, 1)[0]
            
            # Sort by score, select top k%
            drug_score_pairs = list(zip(non_interaction_drugs, non_interaction_scores))
            drug_score_pairs.sort(key=lambda x: x[1], reverse=True)  # Descending order
            
            k = max(1, int(len(drug_score_pairs) * k_ratio))
            top_k_drugs = [drug_id for drug_id, _ in drug_score_pairs[:k]]
            
            return np.random.choice(top_k_drugs)
            
        except Exception as e:
            print(f"Hard negative mining error: {e}")
            return self._random_negative_sampling(pos_drug_id, all_drugs, 1)[0]
    
    def _random_negative_sampling(self, pos_drug_id, all_drugs, num_negatives=1):
        """
        Random negative sampling
        
        Args:
            pos_drug_id: Positive sample drug ID
            all_drugs: All drug list
            num_negatives: Number of negative samples to generate
            
        Returns:
            list: Negative sample drug ID list
        """
        negatives = []
        for _ in range(num_negatives):
            neg_drug_id = np.random.choice(all_drugs)
            while neg_drug_id == pos_drug_id:
                neg_drug_id = np.random.choice(all_drugs)
            negatives.append(neg_drug_id)
        return negatives
    
    def clear_history(self):
        """Clear history records"""
        self.previous_negatives = {}
        print("Negative sample history records cleared")
