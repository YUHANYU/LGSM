# analysis for result

import numpy as np

def anylasis(pre_lab, rea_lab):
    location = []
    for i in range(pre_lab.shape[0]):
        if all(pre_lab[i] == rea_lab[i]):
            continue
        else:
            for j in range(pre_lab.shape[1]):
                if pre_lab[i][j] != rea_lab[i][j]:
                    location.append(j)
    for i in range(pre_lab.shape[1]):
        print('%2.0f-th label' % i, 'the number of error %4.0f' % location.count(i))