import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from rdkit import Chem

# Step 1: Load the ChemBERTa model and tokenizer
def load_model_and_tokenizer(local_path="./ChemBERTa-zinc-base-v1"):
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModel.from_pretrained(local_path)
    return model, tokenizer

# Step 2: Verify SMILES format
def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Step 3: Encode SMILES
def encode_smiles(smiles_list, tokenizer):
    inputs = tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs

# Step 4: Extract feature vectors
def extract_features(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)
    features = outputs.last_hidden_state.mean(dim=1)  # Average pooling
    return features


# Step 5: Extract drug features from Excel file
def process_excel(file_path, model, tokenizer, output_path="../../Dataset/MiDrug_features.csv"):
    # Reading Excel files
    df = pd.read_csv(file_path)
    # df = pd.read_excel(file_path)

    # Validate SMILES data
    print("Validate SMILES format...")
    df["is_valid"] = df["SMILES"].apply(validate_smiles)
    valid_df = df[df["is_valid"] == True]
    invalid_df = df[df["is_valid"] == False]

    if len(invalid_df) > 0:
        print(f"The following SMILES are invalid and will be skipped：\n{invalid_df[['Drug_Name', 'SMILES']]}")

    smiles_list = valid_df["SMILES"].tolist()

    # Encode SMILES
    print("Encode SMILES...")
    inputs = encode_smiles(smiles_list, tokenizer)

    #Extract features
    print("Extract eigenvectors...")
    features = extract_features(model, inputs)

    # Save features to CSV file
    print("Save features to CSV file...")
    feature_df = pd.DataFrame(features.numpy())
    feature_df.insert(0, "Drug_Name", valid_df["Drug_Name"].tolist())  # Insert the drug name as the first column
    feature_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Features are saved to {output_path}")

# Main program
if __name__ == "__main__":
    #Configuration file path
    input_file = "../../Dataset/mi_drug_with_SMILES.csv"  # Curated ncRNAs targeted by drugs file from the ncRNADrug database
    # input_file = "../../Dataset/lnc_drug_with_SMILES.xlsx"
    output_file = "../../Dataset/MiDrug_features.csv"  # Output feature CSV file path
    # output_file = "../../Dataset/LncDrug_features.csv"
    local_model_path = "./ChemBERTa-zinc-base-v1"  # Local ChemBERTa model path

    # Load the ChemBERTa model and tokenizer
    print("Load ChemBERTa model from local path...")
    model, tokenizer = load_model_and_tokenizer(local_path=local_model_path)

    # Extract drug features
    process_excel(input_file, model, tokenizer, output_file)