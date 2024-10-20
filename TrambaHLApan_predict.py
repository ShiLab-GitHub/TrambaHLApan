from models.models import Model_IM
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score,precision_recall_curve,auc
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
import sys

symbol_dic_FASTA = {'X':0,'Y':1, 'S':2, 'M':3, 'R':4, 'E':5, 'I':6, 'N':7, 'V':8, 'G':9, 'L':10,
                    'D':11, 'T':12, 'W':13, 'H':14, 'K':15, 'A':16, 'F':17, 'Q':18, 'C':19, 'P':20,'*':21}


def get_args():
    parser = argparse.ArgumentParser(description='The application of baseline',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', type=str, default='',
                        help='The input file',metavar='E')
    parser.add_argument('--output', dest='output', type=str, default='',
                        help='The output file',metavar='E')


    return parser.parse_args()
 

class new_dataset(Dataset):
    def __init__(self, data_path = './test.csv'):
        super(new_dataset,self).__init__()
        
        #Load pseudo sequences
        pseudo_dir = './data/datasets/pseudoseqs.csv'
        temp = pd.read_csv(pseudo_dir).values.tolist()
        self.pseudoMHC_Dic = dict()
        for item in temp:
            MHCname,pseudoMHC = item[0],item[1]
            MHCname = MHCname.replace('*','').replace(':','')
            self.pseudoMHC_Dic[MHCname] = pseudoMHC
            
        #Load data file
        fulldata = pd.read_csv(data_path).values.tolist()
        self.data = list()
        not_supported_HLA = list()
        for item in fulldata:
            mhcName =item[1].replace('*','').replace(':','')
            if mhcName not in self.pseudoMHC_Dic.keys():
                if mhcName not in not_supported_HLA:
                    print('{} is not supported'.format(mhcName))
                    not_supported_HLA.append(mhcName)
                continue
            self.data.append(item)
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        peptideSeq = self.data[i][0]
        mhcName = self.data[i][1].replace('*','').replace(':','')
        mhcSeq = self.pseudoMHC_Dic[mhcName]
        gt = self.data[i][2]

        #Get input
        peptideSeq = peptideSeq.ljust(11, "*")
        peptide = np.asarray([symbol_dic_FASTA[char] for char in peptideSeq])
        mhc = np.asarray([symbol_dic_FASTA[char] for char in mhcSeq])
        
        #To tensor
        sample = dict()
        sample['peptide'] = torch.LongTensor(peptide) #torch.DoubleTensor(emd)
        sample['mhcSeq'] = torch.LongTensor(mhc)
        sample['labels'] = torch.FloatTensor([float(gt)])

        # return data
        return sample


def evalute(G, P):
    ######evalute EL
    GT_labels = list()
    Pre = list()
    for n, item in enumerate(G):
        if (not np.isnan(item)):
            GT_labels.append(int(item))
            Pre.append(P[n])

    AUC_ROC = roc_auc_score(GT_labels, Pre)
    precision_list, recall_list, _ = precision_recall_curve(GT_labels, Pre)
    AUC_PR = auc(recall_list, precision_list)

    Thresh = 0.5
    pre_labels = [1 if item > Thresh else 0 for item in Pre]

    accuracy = accuracy_score(GT_labels, pre_labels)
    recall = recall_score(GT_labels, pre_labels)
    precision = precision_score(GT_labels, pre_labels)
    F1_score = f1_score(GT_labels, pre_labels)

    evaluation = [accuracy, precision, recall, F1_score, AUC_ROC, AUC_PR]

    return evaluation

if __name__ == '__main__':

    # python TrambaHLApan_predict.py --input ./data/datasets/DataS3.csv --output ./output/results/DataS3_by_TrambaHLApan.csv
    # python TrambaHLApan_predict.py --input ./data/datasets/DataS4.csv --output ./output/results/DataS4_by_TrambaHLApan.csv
    # python TrambaHLApan_predict.py --input ./data/datasets/DataS5.csv --output ./output/results/DataS5_by_TrambaHLApan.csv
    # python TrambaHLApan_predict.py --input ./data/datasets/DataS6.csv --output ./output/results/DataS6_by_TrambaHLApan.csv
    
    
    #Get argument parse
    args = get_args()
    input_file = './' + args.input
    output_file = './' + args.output
    
    if not os.path.exists(input_file):
        print("Not find {}".format(input_file))
        sys.exit()

    #Init 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testDataset = new_dataset(data_path = args.input)
    test_loader = DataLoader(testDataset, batch_size=2048, shuffle=False)

    model_dir = './output/models/'
    model_basename = 'Model-IM_fold*_index0_IM.model'
    
    models = []
    for n in range(5):
        model = Model_IM(num_encoder_layers = 1).to(device)
        # print(model.state_dict().keys())
        model_name = model_basename.replace('*', str(n))
        model_path = model_dir + model_name
        # weights = torch.load(model_path)
        weights = torch.load(model_path,map_location=torch.device('cpu'))
        # print(weights.keys())
        model.load_state_dict(weights)

        models.append(model)

    #Test 
    total_preds_EL = torch.Tensor()
    total_preds_IM = torch.Tensor()
    labels = torch.Tensor()
    for data in tqdm(test_loader):
        #Get input
        peptide = data['peptide'].to(device)
        mhcSeq = data['mhcSeq'].to(device)
        g = data['labels']

        #Calculate output
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
        labels = torch.cat((labels, g), 0)

    P_EL = total_preds_EL.numpy().flatten()
    P_IM = total_preds_IM.numpy().flatten()
    G = labels.numpy().flatten()

    [accuracy, precision, recall, F1_score, AUC_ROC, AUC_PR] = evalute(G, P_EL)
    print('Dataset {} test: accuracy={:.4f},precision={:.4f},recall={:.4f},F1_score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}'.format(
            'P_EL', accuracy, precision, recall, F1_score, AUC_ROC, AUC_PR))
    [accuracy, precision, recall, F1_score, AUC_ROC, AUC_PR] = evalute(G, P_IM)
    print('Dataset {} test: accuracy={:.4f},precision={:.4f},recall={:.4f},F1_score={:.4f},AUC_ROC={:.4f},AUC_PR={:.4f}'.format(
            'P_IM', accuracy, precision, recall, F1_score, AUC_ROC, AUC_PR))


    #Save to local
    column=['peptide','HLA','EL score','IM score']
    results = list()
    for n in range(len(P_IM)):
        results.append([testDataset.data[n][0],testDataset.data[n][1],P_EL[n],P_IM[n]])
        
    output = pd.DataFrame(columns=column,data=results)
    output.to_csv(output_file,index = None)
