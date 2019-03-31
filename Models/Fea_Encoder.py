# The Feature Encoder

import torch
from torch import nn
from torch.autograd import Variable

from Tools.Parameters import Parameters

param = Parameters() # the object of Parameter class
torch.backends.cudnn.benchmark = True 

class Feature_Encoder(nn.Module):

    def __init__(self, train_fea_mat, test_fea_mat):
        """
        :param train_fea_mat: the matrix of training instances
        :param test_fea_mat: the matrix of testing instances
        """
        super(Feature_Encoder, self).__init__()
        train_ins_num, fea_num = train_fea_mat.shape 
        test_ins_num = test_fea_mat.shape[0] 
        self.hidden_size = fea_num 
        self.bi_lstm = param.bi_lstm 
        self.lstm_gru = param.lstm_gru 

        self.train_fea_mat = nn.Embedding(num_embeddings=train_ins_num,
                                          embedding_dim=fea_num,
                                          _weight=torch.from_numpy(train_fea_mat))
        self.train_fea_mat.weight.requires_grad = False 

        self.test_fea_mat = nn.Embedding(num_embeddings=test_ins_num,
                                         embedding_dim=fea_num,
                                         _weight=torch.from_numpy(test_fea_mat))
        self.test_fea_mat.weight.requires_grad = False 

        if self.lstm_gru: # 如果是LSTM
            self.rnn = nn.LSTM(input_size=self.hidden_size,
                               hidden_size=self.hidden_size,
                               batch_first=True,
                               bias=True,
                               bidirectional=self.bi_lstm, 
                               dropout=param.dropout, 
                               num_layers=param.num_layers)
        else: # 如果是GRU
            self.rnn = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              batch_first=True,
                              bias=True,
                              bidirectional=self.bi_lstm, 
                              dropout=param.dropout, 
                              num_layers=param.num_layers)


    def forward(self, input, state, mark):
        """
        :param index: the current instance
        :param state: (h_t+c_t)
        :param mark: train or test instances
        :return:
        """
        if mark == 1: # train
            emb = self.test_fea_mat(input)
        elif mark == 0: # test
            emb = self.train_fea_mat(input)
        emb = emb.unsqueeze(0).float()

        if self.lstm_gru: # LSTM
            output, (h_t, c_t) = self.rnn(emb, state)
            state = (h_t, c_t)
        else: # GRU
            output, h_t = self.rnn(emb, state)
            state = h_t

        return output, state

    def init_LSTM(self): 
        direct = 2 if self.bi_lstm else 1 

        h_0 = Variable(torch.zeros(direct * param.num_layers, 1, self.hidden_size))
        h_0 = h_0.cuda() if param.Use_cuda else h_0

        c_0 = Variable(torch.zeros(direct * param.num_layers, 1, self.hidden_size))
        c_0 = c_0.cuda() if param.Use_cuda else c_0

        return (h_0, c_0)

    def init_GRU(self): 
        direct = 2 if self.bi_lstm else 1 

        h_0 = Variable(torch.zeros(direct * param.num_layers, 1, self.hidden_size))
        h_0 = h_0.cuda(0) if param.Use_cuda else h_0

        return h_0