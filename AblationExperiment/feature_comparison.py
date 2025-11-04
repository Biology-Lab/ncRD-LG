import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import random

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from main import run_experiment, parser
import argparse

from rdkit import Chem
from rdkit.Chem import AllChem
from mol2vec import features, helpers
from gensim.models import Word2Vec
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

# assign cuda device ID
device = torch.device('cuda:1')


# ncRD-LG
def get_original_features(orig_RNA, orig_Drug):
    return orig_RNA.clone(), orig_Drug.clone()


# onehot
def get_onehot_features(orig_RNA, orig_Drug):
    num_RNA = orig_RNA.shape[0]
    num_drug = orig_Drug.shape[0]

    RNA_labels = torch.arange(num_RNA, device='cuda:0')
    RNA_onehot = torch.nn.functional.one_hot(RNA_labels, num_classes=num_RNA).float()
    drug_labels = torch.arange(num_drug, device='cuda:0')
    Drug_onehot = torch.nn.functional.one_hot(drug_labels, num_classes=num_drug).float()
    return RNA_onehot, Drug_onehot


# mol2vec
def get_mol2vec_drug_features():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, '../Dataset'))
    df = pd.read_excel(os.path.join(dataset_path, 'lnc_drug_with_SMILES.xlsx'))
    # df = pd.read_csv(os.path.join(dataset_path, 'mi_drug_with_SMILES.csv'))
    df.columns = df.columns.str.strip()
    df = df[['Drug_Name', 'SMILES']]

    # Generate sentences
    sentences = []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue
        fps = [x for x in AllChem.GetMorganFingerprintAsBitVect(mol, radius=1, nBits=1024)]
        sentence = ['MORGAN_' + str(x) for x in fps]
        sentences.append(sentence)

    # Train model
    model = Word2Vec(sentences, vector_size=300, window=10, min_count=3, sg=1, workers=4)

    # Generate features
    features_list = []
    keys = set(model.wv.key_to_index.keys())
    for sentence in sentences:
        vec = np.zeros(model.vector_size)
        for word in sentence:
            if word in keys:
                vec += model.wv[word]
            else:
                vec += model.wv['UNK']
        vec /= len(sentence)
        features_list.append(vec)

    Drug_mol2vec = torch.tensor(np.array(features_list), dtype=torch.float32)
    return Drug_mol2vec


# word2vec
def get_word2vec_rna_features():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, '../Dataset'))
    df = pd.read_excel(os.path.join(dataset_path, 'lncRNA_with_sequence.xlsx'))
    sequences = df["Sequence"].values

    def kmer_tokenize(seq, k=6):
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]

    corpus = [kmer_tokenize(seq, k=6) for seq in sequences]

    # train model
    model = Word2Vec(sentences=corpus, vector_size=768, window=5, min_count=1, workers=4, sg=1)
    all_vectors = []
    for seq in sequences:
        kmers = kmer_tokenize(seq, k=6)
        vecs = [model.wv[kmer] for kmer in kmers if kmer in model.wv]
        if len(vecs) == 0:
            vectors = np.zeros(model.vector_size)
        else:
            vectors = np.mean(vecs, axis=0)
        all_vectors.append(vectors)
    RNA_word2vec = torch.tensor(np.array(all_vectors), dtype=torch.float32)
    return RNA_word2vec


# Seq2Seq
class SMILES_Dataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, max_len=600):
        chars = set("".join(smiles_list))
        self.vocab = {"<PAD>": 0, **{c: i+1 for i, c in enumerate(chars)}}
        self.pad_idx = 0
        self.smiles_list = smiles_list
        self.max_len = max_len

    def tokenize(self, smiles):
        tokens = [self.vocab.get(c, 0) for c in smiles]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens += [self.pad_idx] * (self.max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.tokenize(self.smiles_list[idx])

class Seq2Seq_Encoder(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=768):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        feat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch, hidden_dim*2)
        return feat

def get_seq2seq_drug_features():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, '../Dataset'))
    df = pd.read_excel(os.path.join(dataset_path, 'lnc_drug_with_SMILES.xlsx'))
    # df = pd.read_csv(os.path.join(dataset_path, 'mi_drug_with_SMILES.csv'))
    smiles_list = df["SMILES"].tolist()

    dataset = SMILES_Dataset(smiles_list)
    vocab_size = len(dataset.vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Seq2Seq_Encoder(vocab_size, embed_dim=256, hidden_dim=768).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=dataset.pad_idx)

    # Train
    model.train()
    for epoch in range(10):
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model.embedding(batch)
            output, _ = model.lstm(logits)
            loss = criterion(output.reshape(-1, output.shape[-1]), batch.reshape(-1))
            loss.backward()
            optimizer.step()

    # Extract features
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(dataset, batch_size=32):
            batch = batch.to(device)
            embedded = model.embedding(batch)
            _, (hidden, _) = model.lstm(embedded)
            emb = hidden[-1].cpu()  # (batch_size, hidden_dim)
            embeddings.append(emb)
    Drug_Seq2Seq = torch.cat(embeddings, dim=0)
    return Drug_Seq2Seq

# TF-IDF
def smiles_tokenizer(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return []
    tokens = [atom.GetSymbol() for atom in mol.GetAtoms()]
    tokens += [str(bond.GetBondType()) for bond in mol.GetBonds()]
    return tokens

def standardize_smiles(s):
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except:
        return None

def get_tfidf_drug_features():
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, '../Dataset'))
    df = pd.read_excel(os.path.join(dataset_path, 'lnc_drug_with_SMILES.xlsx'))
    # df = pd.read_csv(os.path.join(dataset_path, 'mi_drug_with_SMILES.csv'))
    smiles_list = df['SMILES'].tolist()

    standardized = [standardize_smiles(s) for s in smiles_list]
    valid_smiles = [s for s in standardized if s is not None]

    vectorizer = TfidfVectorizer(
        tokenizer=smiles_tokenizer,
        lowercase=False,
        max_features=1000,
        min_df=2,
        norm='l2'
    )
    tfidf_matrix = vectorizer.fit_transform(valid_smiles)
    Drug_TFIDF = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)
    return Drug_TFIDF

def run_ablation_study():
    # Arguments
    parser = argparse.ArgumentParser(description='ncRD-LG')
    # general settings
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='if set, skip the training')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='turn on debugging mode which uses a small number of data')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                        help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                        valid only for ml_1m and ml_10m')
    parser.add_argument('--keep-old', action='store_true', default=False,
                        help='if True, do not overwrite old .py files in the result folder')
    parser.add_argument('--save-interval', type=int, default=20,
                        help='save model states every # epochs ')
    # subgraph extraction settings
    parser.add_argument('--hop', default=3, metavar='S',
                        help='hop number of extracting local graph')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max-nodes-per-hop', default=100,
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    parser.add_argument('--use-features', action='store_true', default=True,
                        help='whether to use node features (side information)')
    # edge dropout settings
    parser.add_argument('--adj-dropout', type=float, default=0,
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False,
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    # optimization settings
    parser.add_argument('--continue-from', type=int, default=None,
                        help="from which epoch's checkpoint to continue training")
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 1e-2)')
    parser.add_argument('--lr-decay-step-size', type=int, default=50,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--lr-decay-factor', type=float, default=0.2,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=300, metavar='N',
                        help='batch size during training')
    parser.add_argument('--test-freq',
                        type=int, default=40, metavar='N',
                        help='test every n epochs')
    parser.add_argument('--ARR', type=float, default=0.00,
                        help='The adjacenct rating regularizer. If not 0, regularize the \
                        differences between graph convolution parameters W associated with\
                        adjacent ratings')

    parser.add_argument('--data-name', default='ml_100k', help='dataset name')
    parser.add_argument('--data-appendix', default='',
                        help='what to append to save-names when saving datasets')

    parser.add_argument('--testing', action='store_true', default=False,
                        help='if set, use testing mode which splits all ratings into train/test;\
                        otherwise, use validation model which splits all ratings into \
                        train/val/test and evaluate on val only')

    parser.add_argument('--save-appendix', default='',
                        help='what to append to save-names when saving results')

    parser.add_argument('--max-train-num', type=int, default=None,
                        help='set maximum number of train data to use')
    parser.add_argument('--max-val-num', type=int, default=None,
                        help='set maximum number of val data to use')

    args = parser.parse_args()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)


    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.hop = int(args.hop)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)

    rating_map = None

    # load data
    file_path = os.path.dirname(__file__)
    dataset_path = os.path.abspath(os.path.join(file_path, '../Dataset'))
    orig_RNA = torch.load(os.path.join(dataset_path, 'LncRNA.pt'))
    orig_Drug = torch.load(os.path.join(dataset_path, 'LncDrug.pt'))
    Pair = np.load(os.path.join(dataset_path, 'LncPair.npy'))
    Pair_new = np.load(os.path.join(dataset_path, 'LncPair_new.npy'))
    # orig_RNA = torch.load(os.path.join(dataset_path, 'MiRNA.pt'))
    # orig_Drug = torch.load(os.path.join(dataset_path, 'MiDrug.pt'))
    # Pair = np.load(os.path.join(dataset_path, 'MiPair.npy'))
    # Pair_new = np.load(os.path.join(dataset_path, 'MiPair_new.npy'))


    results = {}

    experiments = [
        ("ncRD-LG", lambda: (orig_RNA, orig_Drug)),
        ("onehot", lambda: get_onehot_features(orig_RNA, orig_Drug)),
        ("mol2vec", lambda: (orig_RNA, get_mol2vec_drug_features())),
        ("word2vec", lambda: (get_word2vec_rna_features(), orig_Drug)),
        ("seq2seq", lambda: (orig_RNA, get_seq2seq_drug_features())),
        ("tfidf", lambda: (orig_RNA, get_tfidf_drug_features())),
    ]

    for name, feat_func in experiments:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        try:
            RNA_feat, Drug_feat = feat_func()
            best_metric = run_experiment(RNA_feat, Drug_feat, Pair, Pair_new, args, device)

        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = None

if __name__ == '__main__':
    run_ablation_study()

