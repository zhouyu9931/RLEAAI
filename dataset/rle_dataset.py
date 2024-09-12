import torch
import os
import numpy as np
from torch.utils.data import Dataset
from tensor import pad_data_to_same_shape
from Bio import SeqIO
from utils.cksaap import CKSAAP


class rleDataset(Dataset):
    def __init__(self, npz_dir, seq_dir, npz_files, virus_name='HIV'):
        self.virus_name = virus_name
        self.seq_dir = seq_dir
        self.npz_dir = npz_dir
        self.npz_files = npz_files
        self.fasta_files = [f for f in os.listdir(seq_dir) if f.endswith(".fasta")]


    def __len__(self):
        # 返回数据集的大小
        return len(self.npz_files)

    def __getitem__(self, idx):
        # print("idx:", idx)
        npz_file = self.npz_files[idx]
        name = os.path.splitext(npz_file)[0]
        npz_file_path = os.path.join(self.npz_dir, npz_file)
        data = np.load(npz_file_path)
        # print(data)
        ab_emb = data['abemb']
        ag_emb = data['agemb']
        label = data['label']

        ab_emb = torch.tensor(ab_emb)
        ag_emb = torch.tensor(ag_emb)
        label = torch.tensor(label)

        # fasta_file = [f for f in self.fasta_files if f.startswith(name)]
        fasta_file_path = os.path.join(self.seq_dir, f"{name}_{int(data['label'])}.fasta")
        seq_dict = self.get_fasta_chain_dict(fasta_file_path)
        ab_seq = ''.join(list(seq_dict.values())[0])
        ag_seq = ''.join(list(seq_dict.values())[1])
        ab_seq, ag_seq = str(ab_seq), str(ag_seq)


        CKSAAP_ = CKSAAP()
        ab_cc = CKSAAP_.return_CKSAAP_Emb_code(ab_seq, ab_emb.squeeze(), is_shape_for_3d=True)
        ag_cc = CKSAAP_.return_CKSAAP_Emb_code(ag_seq, ag_emb.squeeze(), is_shape_for_3d=True)
        ab_cc, ag_cc = ab_cc.unsqueeze(0), ag_cc.unsqueeze(0)

        return ab_emb.float(), ag_emb.float(), ab_cc, ag_cc, label

    def get_fasta_chain_dict(self, fasta_file):
        seq_dict = {}
        for chain in SeqIO.parse(fasta_file, 'fasta'):
            seq_dict[chain.id] = str(chain.seq)
        return seq_dict

    @staticmethod
    def to_batch(batch):
        ab_emb, ag_emb, ab_cc, ag_cc, label = zip(*batch)
        # print(type(ab_emb), type(ag_emb), type(label))
        # print(len(ab_emb), len(ag_emb), len(label))

        ab_emb = pad_data_to_same_shape([a.squeeze(0) for a in ab_emb])
        # print(ab_emb.shape)
        ag_emb = pad_data_to_same_shape([a.squeeze(0) for a in ag_emb])
        ab_cc = pad_data_to_same_shape([a.squeeze(0) for a in ab_cc])
        ag_cc = pad_data_to_same_shape([a.squeeze(0) for a in ag_cc])
        # print(ag_emb.shape)
        # label = pad_data_to_same_shape([t.squeeze(0) for t in label])
        label = torch.tensor(label)
        # print(label.shape)

        # ab_emb = pad_data_to_same_shape(ab_emb)
        # ag_emb = pad_data_to_same_shape(ag_emb)
        # label = pad_data_to_same_shape(label)

        return ab_emb, ag_emb, ab_cc, ag_cc, label