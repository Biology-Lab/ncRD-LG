"""
Interaction record table module
For fast validation of drug-miRNA interactions
"""

class InteractionTable:
    """Interaction record table for fast validation of drug-miRNA interactions"""
    
    def __init__(self, training_data):
        """
        Initialize interaction record table
        
        Args:
            training_data: Training data, format: [(miRNA_id, drug_id, label), ...]
        """
        self.interactions = set()
        for miRNA_id, drug_id, _ in training_data:
            self.interactions.add((miRNA_id, drug_id))
        
        print(f"Interaction record table initialized, recorded {len(self.interactions)} interactions")
    
    def has_interaction(self, miRNA_id, drug_id):
        """Check if interaction exists"""
        return (miRNA_id, drug_id) in self.interactions
    
    def get_non_interaction_drugs(self, miRNA_id, all_drugs):
        """Get list of drugs that have no interaction with specified miRNA"""
        return [drug_id for drug_id in all_drugs 
                if not self.has_interaction(miRNA_id, drug_id)]
    
    def get_interaction_count(self):
        """Get total interaction count"""
        return len(self.interactions)
