import numpy as np

def conf_max(prob, label, thred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    prob = np.array(prob)
    label = np.array(label)
    # print(prob.shape, label.shape)
    # print(prob, label)
    for prob_value, label_value in zip(prob, label):
        # print(prob_value, label_value)
        if prob_value < thred:
            if label_value == 0:
                TN += 1
            else:
                FN += 1
        elif prob_value >= thred:
            if label_value == 1:
                TP += 1
            else:
                FP += 1
    return TP, FP, TN, FN

def cal_metric (prob_list, label_list, thred):
    TP, FP, TN, FN = conf_max(prob_list, label_list, thred)
    print(TP, FP, TN, FN)
    ACC, RECALL, SPEC, PREC, F1, MCC = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if (TP + FP) != 0:
        ACC = (TP + TN) / (TP + FN + FP + TN)
        RECALL = TP / (TP + FN)
        SPEC = TN / (TN + FP)
        PREC = TP / (TP + FP)
        if RECALL + PREC != 0:
            F1 = (2 * RECALL * PREC) / (RECALL + PREC)

    if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5 != 0:
        MCC = (TP * TN - FN * FP) / (((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5)


    return ACC, RECALL, SPEC, PREC, F1, MCC