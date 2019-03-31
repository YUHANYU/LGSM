# label embedding
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import torch
import gensim
from gensim.models import word2vec, Word2Vec
import numpy as np

def lab_emb(lab_num, fea_num):

    lab_list = [] 
    pos_lab_list = ['Bos'] 
    neg_lab_list = ['Bos_'] 

    for i in range(lab_num):
        pos_lab_list.append(str('L' + str(i))) 
        neg_lab_list.append(str('L' + str(i) + '_')) 


    for i in range(1, lab_num + 1, 1):
        for j in range(lab_num + 1 - i):
            temp_1 = pos_lab_list[j: j + 1 + i]
            lab_list.append(temp_1)
    for i in range(1, lab_num + 1, 1):
        for j in range(lab_num + 1 - i):
            temp_2 = neg_lab_list[j: j + 1 + i]
            lab_list.append(temp_2)


    emb_model = Word2Vec(sentences=lab_list, size=fea_num, sg=0, min_count=1, window=1)

    lab_mat = [] 
    for i in range(lab_num + 1):
        lab_mat.append(emb_model[pos_lab_list[i]])
        lab_mat.append(emb_model[neg_lab_list[i]])

    lab_mat = np.array(lab_mat)

    return lab_mat