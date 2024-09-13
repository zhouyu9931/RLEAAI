import torch
import os
import torch.nn.functional as F
from Bio import SeqIO
import argparse

from utils.embedding import esm_embedding
from utils.cksaap import CKSAAP
from model.RLEAAI import rleModel


def _get_args():
    """Gets command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument('--ab_fa', type=str, default='./data/ab.fasta', help='path to antibody fasta file')
    parser.add_argument('--ag_fa', type=str, default='./data/ag.fasta', help='path to antigen fasta file')
    parser.add_argument('--virus', type=str, default='HIV', help='HIV or SARS-CoV-2')
    parser.add_argument('--ckp_dir', type=str, default='./ckp')

    return parser.parse_args()


def pre_fa(fa_file):
    seq_dict = {}
    for chain in SeqIO.parse(fa_file, 'fasta'):
        seq_dict[chain.id] = str(chain.seq)

    return seq_dict


def main():
    args = _get_args()
    ab_dic = pre_fa(args.ab_fa)
    ag_dic = pre_fa(args.ag_fa)
    ab_seq, ag_seq = ''.join(list(ab_dic.values())), ''.join(list(ag_dic.values()))
    # print(ab_seq, ag_seq)

    ab_emb = esm_embedding(ab_dic)
    ag_emb = esm_embedding(ag_dic)

    CKSAAP_ = CKSAAP()
    ab_cks = CKSAAP_.return_CKSAAP_Emb_code(ab_seq, ab_emb.squeeze(), is_shape_for_3d=True)
    ag_cks = CKSAAP_.return_CKSAAP_Emb_code(ag_seq, ag_emb.squeeze(), is_shape_for_3d=True)
    ab_cks, ag_cks = ab_cks.unsqueeze(0), ag_cks.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = rleModel().to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckp_dir, f'{args.virus}.pth'), map_location=device))

    with torch.no_grad():
        model.eval()
        ab_emb, ag_emb = ab_emb.to(device), ag_emb.to(device)
        ab_cks, ag_cks = ab_cks.to(device), ag_cks.to(device)
        # print(ab_emb.shape, ag_emb.shape, ab_cks.shape, ag_cks.shape)
        # print(ab_emb, ag_emb)
        # print(ab_cks, ag_cks)

        output = model(ab_emb, ag_emb, ab_cks, ag_cks)
        prob = F.sigmoid(output)
        # print(prob)

        print(f"Predicted probability of neutralization: {prob.item():.4f}")



if __name__ == '__main__':
    main()


