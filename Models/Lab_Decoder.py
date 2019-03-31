# The Normal Label Decoder

import torch
from torch import nn
import torch.nn.functional as F

from Tools.Parameters import Parameters

param = Parameters()
torch.backends.cudnn.benchmark = True

class Label_Decoder(nn.Module):

    def __init__(self, lab_emb, k):
        super(Label_Decoder, self).__init__()
        lab_bos_num, fea_num = lab_emb.shape 
        lab_num = lab_bos_num - 2
        self.hidden_size = fea_num

        self.lab_emb = nn.Embedding(num_embeddings=lab_bos_num,
                                    embedding_dim=fea_num,
                                    _weight=torch.from_numpy(lab_emb))
        self.lab_emb.weight.requires_grad = False 

        if param.lstm_gru: # 使用LSTM
            self.rnn = nn.LSTM(input_size=self.hidden_size,
                               hidden_size=self.hidden_size,
                               batch_first=True,
                               bidirectional=param.bi_lstm, 
                               num_layers=param.num_layers, 
                               bias=True,
                               dropout=param.dropout)
        else: # 使用GRU
            self.rnn = nn.GRU(input_size=self.hidden_size,
                              hidden_size=self.hidden_size,
                              batch_first=True,
                              bidirectional=param.bi_lstm,
                              num_layers=param.num_layers,
                              bias=True,
                              dropout=param.dropout) 

        # 融合已知（上一个）标记emb+预测（下一个）标记emb+预测标记的非的emb
        self.lab_2_merge = nn.Linear(in_features=fea_num * 2, out_features=fea_num) 
        self.lab_3_merge = nn.Linear(in_features=fea_num * 3, out_features=fea_num) 

        out_num = 2 if param.one_zero else lab_bos_num 
        bidirection = 2 if param.bi_lstm else 1 
        self.out = nn.Linear(in_features=fea_num * bidirection, out_features=out_num)

        self.dropout = nn.Dropout(p=param.dropout)

        self.softmax = nn.LogSoftmax(dim=2) 

        self.all_lab_emb = nn.Linear(in_features=fea_num * int(lab_num / 2), out_features=fea_num) 

        self.emb_hidden = nn.Linear(in_features=(bidirection * param.num_layers + 1) * fea_num,
                                    out_features=fea_num) 

        self.atten_w = nn.Linear(in_features=fea_num, out_features=k + 1) 

        self.linear = nn.Linear(in_features=fea_num * 2, out_features=fea_num) 

    def forward(self, index, lab_value, alpha, state, encoder_outputs):


        index = torch.LongTensor([index]) 
        index = index.cuda() if param.Use_cuda else index
        emb1 = self.lab_emb(index) if lab_value[index] else self.lab_emb(index + 1)


        if alpha == 0: 
            emb2 = self.lab_emb(index + 3) * (1 - alpha)
            emb = torch.cat((emb1, emb2), 1)
            emb = self.lab_2_merge(emb).unsqueeze(0)
        elif alpha == 1: # 如果alpha的值为1：
            emb2 = self.lab_emb(index + 2) * alpha 
            emb = torch.cat((emb1, emb2), 1)
            emb = self.lab_2_merge(emb).unsqueeze(0)
        else: # alpha的值在0和1之间
            emb2 = self.lab_emb(index + 2) * alpha
            emb3 = self.lab_emb(index + 3) * (1 - alpha) 
            emb = torch.cat((emb1, emb2, emb3), 1)
            emb = self.lab_3_merge(emb).unsqueeze(0)


        hidden = state[0] 
        hidden = torch.transpose(hidden, 0, 1) 
        input = torch.cat((emb[0], hidden[0]), 0).view(1, -1) 
        input = self.emb_hidden(input) 
        atten_weights = self.atten_w(input) 
        atten_w_outputs = torch.bmm(atten_weights.unsqueeze(0), encoder_outputs) 
        atten_applpy = self.linear(atten_w_outputs) 
        atten_emb = torch.cat((emb[0], atten_applpy[0]), 1) 
        input = self.linear(atten_emb).unsqueeze(0) 
        input = F.relu(input) 

        if param.lstm_gru:
            output, (h_t, c_t) = self.rnn(input, state)
            state = (h_t, c_t)
        else:
            output, h_t = self.rnn(emb, state)
            state = h_t

        output = self.out(output)
        output = self.softmax(output)

        return output, state
