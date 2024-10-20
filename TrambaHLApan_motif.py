from models.models import Model_IM
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
from weblogo import *
import matplotlib
matplotlib.use('Agg')

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42


symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'*':21}

AA_pos_dict = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5, 'Q':6, 'E':7, 'G':8, 'H':9, 'I':10,
                            'L':11, 'K':12, 'M':13, 'F':14, 'P':15, 'S':16, 'T':17, 'W':18, 'Y':19, 'V':20}

def get_args():
    parser = argparse.ArgumentParser(description='Immune epitope Motif from model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--HLA', dest='HLA', type=str, default='',
                        help='The HLA name',metavar='E')
    parser.add_argument('--HLAseq', dest='HLAseq', type=str, default='',
                        help='The HLA sequence',metavar='E')


    return parser.parse_args()

class randomPepData(Dataset):
    def __init__(self, data_path = './data/uniprot/9_peptides.csv',mhcSeq = ''):
        super(randomPepData,self).__init__()

        #Load data file
        self.data = pd.read_csv(data_path).values.tolist()
        self.mhcSeq = mhcSeq
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptideSeq = self.data[i][0]
        mhcSeq = self.mhcSeq

        # Get input
        peptideSeq = peptideSeq.ljust(11, "*")
        peptide = np.asarray([symbol_dic_FASTA[char] for char in peptideSeq])
        mhc = np.asarray([symbol_dic_FASTA[char] for char in mhcSeq])

        # To tensor
        sample = dict()
        sample['peptide'] = torch.LongTensor(peptide)  # torch.DoubleTensor(emd)
        sample['mhcSeq'] = torch.LongTensor(mhc)

        # return data
        return sample


if __name__ == '__main__':
    # python TrambaHLApan_motif.py --HLA HLA-A*02:01 --HLAseq YFAMYGEKVAHTHVDTLYVRYHYYTWAVLAYTWY
    
    #Get argument parse
    args = get_args()
    HLA = args.HLA
    HLASeq = args.HLAseq

    #Init
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    #Load Data
    testDataset = randomPepData(mhcSeq = HLASeq)
    test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)

    model_dir = './output/models/'
    model_basename = 'Model-IM_fold*_index0_IM.model'
    models = []
    for n in range(5):
        model = Model_IM(num_encoder_layers = 1).to(device)
        model_name = model_basename.replace('*', str(n))
        model_path = model_dir + model_name
        # weights = torch.load(model_path)
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(weights)

        models.append(model)
    
    # attn_output_weights = torch.Tensor()
    total_preds_EL = torch.Tensor()
    total_preds_IM = torch.Tensor()
    #Test 
    for data in tqdm(test_loader):
        #Get input
        peptide = data['peptide'].to(device)
        mhcSeq = data['mhcSeq'].to(device)

        output_ave_EL = 0
        output_ave_IM = 0
        for model in models:
            model.eval()
            with torch.no_grad():
                y_EL,y_IM = model(peptide, mhcSeq)

                y_EL = y_EL.cpu()
                y_IM = y_IM.cpu()
                
                output_ave_EL = output_ave_EL + y_EL
                output_ave_IM = output_ave_IM + y_IM

        output_ave_EL = output_ave_EL / len(models)     
        output_ave_IM = output_ave_IM / len(models)

        total_preds_EL = torch.cat((total_preds_EL, output_ave_EL), 0)
        total_preds_IM = torch.cat((total_preds_IM, output_ave_IM), 0)
        

    # attn_output_weights = attn_output_weights.numpy()  #45*45
    P_EL = total_preds_EL.numpy().flatten()
    P_IM = total_preds_IM.numpy().flatten()


    # atten_data = list()
    for motif_name in ['EL','IM']:
        title_heatmap =  HLA + '_heatmap(' + motif_name + ')'
        title_logo =  HLA + '_logo(' + motif_name + ')'

        #Find top 1000
        TOP_NUM = 1000
        #Find top
        if motif_name == 'EL':
            top_indexs = np.argsort(1-P_EL)[:TOP_NUM]
        elif motif_name == 'IM':
            top_indexs = np.argsort(1-P_IM)[:TOP_NUM]
            
        top_scores = [total_preds_IM[idx] for idx in top_indexs]

        #####Draw heatmap
        # atten_scores = list()
        peptide_list = list()
        for n in range(TOP_NUM):
            idx = top_indexs[n]
            peptide = testDataset.data[idx][0]
            HLA = HLA

            peptide_list.append(peptide)
        
        ##Draw logo
        f = open('./data/temp.txt','w')
        f.write('\n'.join(peptide_list))
        f.close()
        
        f = open('./data/temp.txt')
        seqs = read_seq_data(f)
        data = LogoData.from_seqs(seqs)

        options = LogoOptions()
        options.fineprint = ''# MHC + '(' + motif_name + ')'
        # options.xaxis_label = 'Position'
        options.yaxis_label = 'Bits'
        options.show_xaxis = True
        options.resolution = 600
        options.number_interval = 1
        format = LogoFormat(data,options)

        output = jpeg_formatter(data,format)

        with open(title_logo + '.jpg','wb') as f:
            f.write(output)
        f.close()
        os.remove('./data/temp.txt')
