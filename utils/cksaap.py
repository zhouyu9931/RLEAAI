# -*- coding: UTF-8 -*-
import esm
import torch.nn as nn
import torch
import numpy as np
import sys
import math
from torch.autograd import Variable
from itertools import product


class CKSAAP(nn.Module):
    def __init__(self, position_d_model=None):
        super(CKSAAP, self).__init__()

        AA = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        DP = list(product(AA, AA))
        # print(DP) #[('A', 'A'), ('A', 'C'), ..., ('Y', 'W'), ('Y', 'Y')]
        # print(len(DP)) #400
        self.DP_list = []
        for i in DP:
            self.DP_list.append(str(i[0]) + str(i[1]))
        # print(self.DP_list)  #['AA', 'AC', ..., 'YW', 'YY']

        self.position_func = None
        self.position_d_model = position_d_model

    def returnCKSAAPcode(self, query_seq, k=3):
        code_final = []
        for turns in range(k + 1):
            DP_dic = {}
            code = []
            code_order = []
            for i in self.DP_list:
                DP_dic[i] = 0
            for i in range(len(query_seq) - turns - 1):
                tmp_dp_1 = query_seq[i]
                tmp_dp_2 = query_seq[i + turns + 1]
                tmp_dp = tmp_dp_1 + tmp_dp_2
                if tmp_dp in DP_dic.keys():
                    DP_dic[tmp_dp] += 1
                else:
                    DP_dic[tmp_dp] = 1
            for i, j in DP_dic.items():
                code.append(j / (len(query_seq) - turns - 1))
            for i in self.DP_list:
                code_order.append(code[self.DP_list.index(i)])
            code_final += code

        code_final = torch.FloatTensor(code_final)
        code_final = code_final.view(k + 1, 20, 20)
        return code_final

    def return_CKSAAP_Emb_code(self, query_seq, emb, k=3, is_shape_for_3d=False):
        code_final = torch.zeros((k + 1, 20, 20, emb.size(-1)))
        for turns in range(k + 1):
            DP_dic = {}
            for i in self.DP_list:
                DP_dic[i] = torch.zeros(emb.size(-1))

            for i in range(len(query_seq) - turns - 1):
                tmp_dp_1 = query_seq[i]
                tmp_dp_2 = query_seq[i + turns + 1]
                tmp_emb_1 = emb[i]
                tmp_emb_2 = emb[i + turns + 1]
                tmp_emb = 0.5 * (tmp_emb_1 + tmp_emb_2)

                tmp_dp = tmp_dp_1 + tmp_dp_2
                if tmp_dp in DP_dic.keys():
                    DP_dic[tmp_dp] += tmp_emb
                else:
                    DP_dic[tmp_dp] = tmp_emb

            for idx, i in enumerate(self.DP_list):
                code_final[turns, idx // 20, idx % 20, :] = DP_dic[i] / (len(query_seq) - turns - 1)

        if is_shape_for_3d:
            k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = code_final.size()
            code_final = code_final.permute(0, 3, 1, 2).contiguous(). \
                view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
        return code_final

    def return_CKSAAP_position_code(self, query_seq, k=3):
        """
        :param query_seq: L
        :param embs: [L, D] tensor
        :param k:
        :return: [(k+1)*position_posi_emb_size, 20, 20]
        """
        posi_emb = self.position_func(
            torch.zeros(1, len(query_seq), self.position_d_model)
        ).squeeze(0)

        # [(k+1), 20, 20, position_posi_emb_size]
        emb = self.return_CKSAAP_Emb_code(query_seq, posi_emb, k)

        # [(k+1), 20, 20, position_posi_emb_size] --> [(k+1)*position_posi_emb_size, 20, 20]
        k_plus_one, aa_num_1, aa_num_2, position_posi_emb_size = emb.size()
        emb = emb.permute(0, 3, 1, 2).contiguous().view(k_plus_one * position_posi_emb_size, aa_num_1, aa_num_2)
        return emb


# CKSAAP_ = CKSAAP()
#
# seq1 = 'QMKLMQSGGV'
# emb = torch.randn(1, 10, 1280)
# EKS_coding1 = CKSAAP_.return_CKSAAP_Emb_code(seq1, emb.squeeze(), is_shape_for_3d=True)
# EKS_coding1 =EKS_coding1.unsqueeze(0)
#
# print(EKS_coding1.shape)
# print(EKS_coding1)
