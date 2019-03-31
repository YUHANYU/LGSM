
import torch

class Parameters(object):

    def __init__(self):

        self.Use_cuda = torch.cuda.is_available()

        self.Train_epoches = 20

        self.Lr_1 = 0.00001
        self.Lr_2 = 0.01
        self.Lr_3 = 0.1
        self.Lr_4 = 0.000001
        self.Lr_5 = 0.00009 

        self.Train_Teach = True 

        self.one_zero = False 

        self.bi_lstm = True 

        self.atten_decoder = True 

        self.lstm_gru = True

        self.num_layers = 2 

        self.dropout = 0.5 