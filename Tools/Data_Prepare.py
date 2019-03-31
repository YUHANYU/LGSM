# data prepare

import numpy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import identity
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split

from Tools.Lab_Emb import lab_emb

class Prepare(object):

    def __init__(self, data_type, k):
        self.data_type = data_type
        self.k = k

    def get_data(self, data_ratio=0):
        dataset_path = 'Data\\' + self.data_type + '.csv'
        dataset = numpy.loadtxt(open(dataset_path, 'rb'), delimiter=',', skiprows=0)

        if self.data_type == 'yeast':
            lab_num = 14
        elif self.data_type =='scene':
            lab_num = 6
        elif self.data_type == 'emotions':
            lab_num = 6
        elif self.data_type == 'enron':
            lab_num = 53
        elif self.data_type == 'image':
            lab_num = 5
        if not data_ratio:
            return dataset[:, :-lab_num], dataset[:, -lab_num:]
        else: 
            part_fea, _, part_lab, _ = train_test_split(dataset[:, :-lab_num], dataset[:, -lab_num:], random_state=2018,
                                                        train_size=data_ratio, test_size=1 - data_ratio, shuffle=True)
            return part_fea, part_lab

    def show_labels(self, show_data):
        sns.set()
        x = []
        for i in range(show_data.shape[1]):
            a = 'L' + str(i + 1)
            x.append(a)
        y = []
        for i in range(show_data.shape[0] - 1):
            b = 'I' + str(i + 1)
            y.append(b)
        y.append('I')

        show = sns.heatmap(data=show_data, annot=True, center=0.3, fmt='.0f',
                           cmap='Blues', linewidths=0.001, linecolor='white', cbar=False,
                           xticklabels = x, yticklabels = y)

        show.set_title('Instance-Lable', fontsize=20)
        show.set_xlabel('Label', fontsize=20, rotation=0)
        show.set_ylabel('Instance', fontsize=20, rotation=90)
        show.set_yticklabels(show.get_yticklabels(), rotation=0)

        plt.show()


    def sort_lab(self, sort_lab, sort_mode):

        if sort_mode == 'random':
            return sort_lab
        ins_num, lab_num = sort_lab.shape
        sum_every_col = (sort_lab.sum(axis=0)).tolist()
        index_lab_num = {}
        for i in range(0, len(sum_every_col)):
            index_lab_num[i] = sum_every_col[i]
        if sort_mode == 'descend':
            reverse = True
        if sort_mode == 'acsend': 
            reverse = False
        new_list = sorted(index_lab_num, key=lambda x:index_lab_num[x], reverse=reverse)
        new_lab = np.zeros((ins_num, lab_num))
        for i in range(0, len(new_list)):
            new_lab[:, i] = sort_lab[:, new_list[i]]

        return new_lab

    def get_k_ins_in_train(self, train_fea):

        ins_num, fea_num = train_fea.shape
        k_ins = []
        # kdtree = KDTree(train_fea, metric='euclidean') 
        ball_tree = BallTree(train_fea, metric='euclidean') 
        for i in range(ins_num):
            one_ins = train_fea[i].reshape((1, fea_num))
            # _, k_ins_list = kdtree.query(X=one_ins, k=self.k + 1)
            _, k_ins_list = ball_tree.query(X=one_ins, k=self.k + 1)
            temp = []
            for j in range(1):
                for p in range(self.k + 1):
                    temp.append(k_ins_list[j][p])
            temp = list(reversed(temp))
            k_ins.append(temp)
        k_ins = np.array(k_ins)

        return k_ins


    def get_k_ins_from_train(self, train_fea, target_fea):

        test_ins_num, fea_num = target_fea.shape
        k_ins = []
        # kdtree = KDTree(train_fea, metric='euclidean') # kd-tree
        ball_tree = BallTree(train_fea, metric='euclidean') # ball-tree
        for ins_index in range(test_ins_num): 
            one_ins = target_fea[ins_index].reshape(1, fea_num) 
            # _, k_ins_list = kdtree.query(X=one_ins, k=self.k)
            _, k_ins_list = ball_tree.query(X=one_ins, k=self.k) 
            temp = []
            for i in range(1):
                for j in range(self.k):
                    temp.append(k_ins_list[i][j]) 
            temp = list(reversed(temp)) 
            temp.append(ins_index) 
            k_ins.append(temp)
        k_ins = np.array(k_ins)

        return k_ins


    def train_alpha(self, k_ins, lab):

        train_ins_num = k_ins.shape[0]
        train_lab_num = lab.shape[1]
        alpha = np.zeros((train_ins_num, train_lab_num))
        for i in range(train_ins_num):
            temp = 0
            for j in range(self.k):
                temp += lab[k_ins[i][j]]
            alpha[i] = temp / self.k

        return alpha


    def test_alpha(self, k_ins, lab):
        test_ins_num = k_ins.shape[0] 
        test_lab_num = lab.shape[1] 
        alpha = np.zeros((test_ins_num, test_lab_num))
        for i in range(test_ins_num):
            temp = 0
            for j in range(self.k):
                temp += lab[k_ins[i][j]]
            alpha[i] = temp / self.k

        return alpha

class Data(object):

    def __init__(self, data_type, k, train_fea, train_lab, test_fea, test_lab):


        obj = Prepare(data_type=data_type, k=k) 

        train_k_ins = obj.get_k_ins_in_train(train_fea=train_fea.copy())

        train_alpha = obj.train_alpha(k_ins=train_k_ins.copy(), lab=train_lab.copy())


        test_k_ins = obj.get_k_ins_from_train(train_fea=train_fea.copy(), target_fea=test_fea.copy())

        test_alpha = obj.test_alpha(k_ins=test_k_ins.copy(), lab=train_lab.copy())


        self.fea_num = test_fea.shape[1] 
        self.lab_num = test_lab.shape[1] 
        self.Bos = 0 

        # self.lab_mat = np.random.randn((1 + self.lab_num) * 2, self.fea_num)
        # 
        # self.lab_mat = np.hstack((identity((1 + self.lab_num) * 2),
        #                             np.zeros(((1 + self.lab_num) *2, self.fea_num - (1 + self.lab_num) * 2))))
        # 
        self.lab_mat = lab_emb(lab_num=self.lab_num, fea_num=self.fea_num)

    
        self.k = k 
        self.train_ins_num = train_fea.shape[0] 
        self.train_fea = train_fea.copy() 
        self.train_lab = train_lab.copy() 
        self.train_k_ins = train_k_ins.copy() 
        self.train_alpha = train_alpha.copy() 

        # 测试用到的数据
        self.test_ins_num = test_fea.shape[0] 
        self.test_fea = test_fea.copy() 
        self.test_lab = test_lab.copy() 
        self.test_k_ins = test_k_ins.copy()
        self.test_alpha = test_alpha.copy() 
