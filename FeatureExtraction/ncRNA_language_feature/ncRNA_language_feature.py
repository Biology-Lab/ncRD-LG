import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import transformers

data = pd.read_excel(r"../../Dataset/miRNA_with_sequence.xlsx")  # Curated ncRNAs targeted by drugs file from the ncRNADrug database
# data = pd.read_excel(r"./lncRNA_with_sequence.xlsx")

# Define the name of the output folder
output_folder = "../../Dataset"

# Load the pre-trained tokenizer to convert RNA sequences into an input format acceptable to the model
tokenizer = AutoTokenizer.from_pretrained(r"./BiRNA-Tokenizer")
# Load the configuration information of the pre-trained model
config = transformers.BertConfig.from_pretrained(r"./BiRNA-BERT")

config.vocab_size = tokenizer.vocab_size

# Load the pre-trained masked language model
mysterybert = AutoModelForMaskedLM.from_pretrained(
    r"./BiRNA-BERT",
    config=config,
    trust_remote_code=True
)
#Replace the classification layer of the model with the identity mapping to avoid additional classification operations
mysterybert.cls = torch.nn.Identity()

# Used to store the embedded vectors obtained by each pooling method
head_embeddings = []
avg_embeddings = []
max_embeddings = []
# Used to store the corresponding RNA name
rna_names = []

# Traverse each RNA sequence in the data
for i, seq in enumerate(data['sequence']):
    # Use tokenizer to encode the sequence and convert it to PyTorch tensor
    inputs = tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
    # Do not calculate gradients to reduce memory consumption
    with torch.no_grad():
        # Pass the input into the model and get the output of the model
        seq_embed = mysterybert(**inputs)

    # Get the hidden state of the model output
    hidden_states = seq_embed.logits

    # 1. Take the head (feature of the first token) as the embedding vector
    head_embed = hidden_states[0, 0, :].numpy()
    head_embeddings.append(head_embed)

    # 2. Average pooling: average the hidden state in dimension 1 to get the average embedding vector
    avg_embed = torch.mean(hidden_states, dim=1).squeeze(0).numpy()
    avg_embeddings.append(avg_embed)

    # 3. Max pooling: Take the maximum value of the hidden state in dimension 1 to get the maximum embedding vector
    max_embed = torch.max(hidden_states, dim=1).values.squeeze(0).numpy()
    max_embeddings.append(max_embed)
    # Record the name of the current RNA
    rna_names.append(data['ncRNA_Name'][i])

# Convert the list of embedding vectors obtained by each pooling method to a PyTorch tensor
head_tensor = torch.tensor(np.array(head_embeddings))
avg_tensor = torch.tensor(np.array(avg_embeddings))
max_tensor = torch.tensor(np.array(max_embeddings))

# Save the tensor as a .pt file
torch.save(head_tensor, os.path.join(output_folder, "LncRNA.pt"))

# Save the embedding vectors and corresponding RNA names as a CSV file
def save_as_csv(embeddings, names, filename):
    # Convert the embedding vector to a DataFrame
    df = pd.DataFrame(embeddings)
    df.insert(0, "RNA_Name", names)

    df.to_csv(os.path.join(output_folder, filename), index=False)

print(f"Processing is complete! The results are saved in the {output_folder} folder:")
print("- LncRNA.pt(Head method)")