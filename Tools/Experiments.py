# 分析一些数据和现象

from Tools.Data_Prepare import Prepare
import numpy as np

data_obj = Prepare(data_type='scene', k=20) # 声明数据准备类对象
fea, lab = data_obj.get_data() # 获取示例特征集和标记集
k_ins_set = data_obj.get_k_ins_in_train(train_fea=fea.copy()) # 获取示例集中每个示例的k个特征最近领
lab_set = np.zeros((20 + 1, 6)) # 收集第一个示例的k个特征最近邻和自己的标记集
for i in range(len(k_ins_set[0])): # 循环第一个示例及其k个特征最近邻的序号
    lab_set[i] = lab[k_ins_set[0][i]]
data_obj.show_labels(show_data=lab_set) # 显示这个数据集