# The MLL_Seq2Seq of LSGM

import torch
from torch import nn, optim

from Tools.Data_Prepare import Data, Prepare
from Tools.Parameters import Parameters
from Tools.Evaluate import MLL_Evaluate, MLL_Mean_Std
from Tools.Analysis import anylasis
from Models.Fea_Encoder import Feature_Encoder
from Models.Lab_Decoder import Label_Decoder
from Models.Lab_Dec_Global import Label_Decoder_Global_Infor


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
import numpy as np

param = Parameters()
torch.backends.cudnn.benchmark = True 

# define the MLL_Seq2Seq of LSGM, include the encoder and decoder
class Seq2Seq(object):

    def __init__(self, train_fea_mat, test_fea_mat, lab_emb, k,):

        fea_num = train_fea_mat.shape[1] 
        lab_num = lab_emb.shape[1] 

        self.encoder = Feature_Encoder(train_fea_mat=train_fea_mat, test_fea_mat=test_fea_mat) # the encoder
        self.encoder = self.encoder.cuda() if param.Use_cuda else self.encoder

        decoder = Label_Decoder(lab_emb=lab_emb, k=k) # the decoder
        deccodr = decoder.cuda() if param.Use_cuda else decoder

        decoder_global = Label_Decoder_Global_Infor(lab_emb=lab_emb, k=k) # the decoder with GLI
        decoder_global = decoder_global.cuda() if param.Use_cuda else decoder_atten

        LR = param.Lr_5
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=LR)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=LR)
        decoder_global_optimizer = optim.Adam(decoder_global.parameters(), lr=LR)

        if param.atten_decoder:
            self.decoder = decoder_global
            self.decoder_optimizer = decoder_global_optimizer
        else: # 普通解码器
            self.decoder = decoder
            self.decoder_optimizer = decoder_optimizer

        self.criterion = nn.NLLLoss()

    # train
    def train(self, data):
        """
        :param data: the object of data class
        :return:
        """
        for epoch in range(param.Train_epoches):
            start_time = time.time()
            # temp = np.zeros((data.train_ins_num, data.lab_num)) 

            for i in range(data.train_ins_num):
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                loss = 0 

                # encode
                mark = 0 
                if param.lstm_gru:
                    encoder_state = self.encoder.init_LSTM()
                else:
                    encoder_state = self.encoder.init_GRU()
                encoder_outputs = torch.zeros(1, data.k + 1, data.fea_num * 2)
                encoder_outputs = encoder_outputs.cuda() if param.Use_cuda else encoder_outputs
                for j in range(data.k + 1):
                    encoder_input = data.train_k_ins[i][j]
                    encoder_input = torch.LongTensor([encoder_input])
                    encoder_input = encoder_input.cuda() if param.Use_cuda else encoder_input
                    encoder_output, encoder_state = self.encoder(encoder_input, encoder_state, mark)
                    encoder_outputs[0][j] = encoder_output[0][0]

                # decode
                decoder_state = encoder_state
                decoder_input = []
                decoder_input.append(data.Bos) 
                for t in range(data.lab_num):
                    decoder_alpha = data.train_alpha[i][t]
                    decoder_output, decoder_state = self.decoder(t, decoder_input, decoder_alpha,
                                                                    decoder_state, encoder_outputs)
                    if param.Train_Teach:
                        decoder_input.append(data.train_lab[i][t])
                    else: 
                        top_1_value, top_1_index = decoder_output.topk(1) 
                        if param.one_zero:
                            decoder_input.append(top_1_index)
                        else: 
                            input = 1 if top_1_index % 2 == 0 else 0 
                            decoder_input.append(input) 
                        temp[i][t] = input 

                    if param.one_zero: 
                        input = decoder_output.squeeze(0)
                        target = data.train_lab[i][t]
                        target = torch.LongTensor([target])
                        target = target.cuda() if param.Use_cuda else target
                        loss += self.criterion(input, target)
                    else : 
                        input = decoder_output.squeeze(0)
                        value = data.train_lab[i][t]
                        if value: 
                            target = (t + 1) * 2  
                        else: 
                            target = (t + 1) * 2 + 1 
                        target = torch.LongTensor([target])
                        target = target.cuda() if param.Use_cuda else target
                        loss += self.criterion(input, target)

                loss.backward() 
                self.encoder_optimizer.step() 
                self.decoder_optimizer.step() 

            # MLL_Evaluate(y_real=data.train_lab, y_pred=temp) 
            end_time = time.time()
            take_time = end_time - start_time
            # print('epoch %2.0f' % epoch, 'loss %8.5f' % float(loss.data), 'time %-4.2fs' % take_time)

    # test
    def test(self, data):
        """
        :param data: the obejct of data class
        :return: 
        """
        pre_lab = np.zeros((data.test_ins_num, data.lab_num)) 
        for i in range(data.test_ins_num):
            # encode
            mark = 0 
            if param.lstm_gru:
                encoder_state = self.encoder.init_LSTM()
            else:
                encoder_state = self.encoder.init_GRU()
            encoder_outputs = torch.zeros(1, data.k + 1, data.fea_num * 2)
            encoder_outputs = encoder_outputs.cuda() if param.Use_cuda else encoder_outputs
            for j in range(data.k + 1):
                mark = 1 if j == data.k + 1 else mark
                encoder_input = data.test_k_ins[i][j] 
                encoder_input = torch.LongTensor([encoder_input])
                encoder_input = encoder_input.cuda() if param.Use_cuda else encoder_input
                encoder_output, encoder_state = self.encoder(encoder_input, encoder_state, mark)
                encoder_outputs[0][j] = encoder_output[0][0]

            # decode
            decoder_state = encoder_state 
            decoder_input = [] 
            decoder_input.append(data.Bos) 
            for t in range(data.lab_num): 
                decoder_alpha = data.test_alpha[i][t]
                decoder_output, decoder_state = self.decoder(t, decoder_input, decoder_alpha,
                                                                decoder_state, encoder_outputs)

                top_1_value, top_1_index = decoder_output.topk(1)
                if param.one_zero:
                    decoder_input = top_1_index
                else: 
                    input = 1 if top_1_index % 2 == 0 else 0  
                    decoder_input.append(input)

                pre_lab[i][t] = input 

        print('***************Current K-fold Result：******************')
        # anylasis(pre_lab=pre_lab.copy(), rea_lab=data.test_lab.copy()) 
        result = MLL_Evaluate(y_pred=pre_lab, y_real=data.test_lab, val_test='test')

        return result


def MLL(data_type, k, k_fold):
    """
    :param data_type: MLL data type
    :param k: the k nearest neighbor instances
    :param k_fold: k-fold
    :return: result（mean+std）
    """
    print(data_type, 'k =', k) 
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=2018) 
    obj = Prepare(data_type=data_type, k=k) 
    fea, lab = obj.get_data() 
    new_lab = obj.sort_lab(sort_lab=lab.copy(), sort_mode='acsend') # random， acsend， descend

    results = []

    for train_index, test_index in kf.split(fea):
        train_fea, test_fea = fea[train_index], fea[test_index]
        train_lab, test_lab = new_lab[train_index], new_lab[test_index]

        data = Data(data_type=data_type, k=k, train_fea=train_fea, train_lab=train_lab,
                    test_fea=test_fea, test_lab=test_lab)

        # 实例化Seq2Seq对象
        seq2seq = Seq2Seq(train_fea_mat=data.train_fea, test_fea_mat=data.test_fea, lab_emb=data.lab_mat, k=data.k)

        seq2seq.train(data=data) 

        result = seq2seq.test(data=data) 

        results.append(result) 

    MLL_Mean_Std(results=results) 
