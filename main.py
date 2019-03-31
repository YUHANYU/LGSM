# main function for Seq2Seq_MLL

from Models.Seq2Seq import MLL

if __name__ == '__main__':
    MLL(data_type='yeast', k=20, k_fold=10)