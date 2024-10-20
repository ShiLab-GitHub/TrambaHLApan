import torch
import torch.nn as nn
from mamba_ssm import Mamba, Mamba2

class xformer(nn.Module):
    def __init__(self,
                 dropout=0.2,
                 num_heads=8,
                 vocab_size=22,
                 num_encoder_layers=1,
                 d_model=128,  # 128
                 Max_len=45,  # 34 + 11 = 45
                 ):
        super(xformer, self).__init__()

        self.embeddingLayer = nn.Embedding(vocab_size, d_model)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_model), requires_grad=True)

        ##Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)


    def forward(self, seq):
        # # Get padding mask
        pad_mask = seq.eq(21)

        Embedding = self.embeddingLayer(seq)  # batch * seq * feature
        Embedding = Embedding + self.positionalEncodings[:Embedding.shape[1], :]

        # input feed-forward:
        Embedding = Embedding.permute(1, 0, 2)  # seq * batch * feature
        feature = self.transformer_encoder(Embedding, src_key_padding_mask=pad_mask)
        feature = feature.permute(1, 0, 2)  # batch * seq * feature

        coff = 1 - pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * feature, dim=1) / torch.sum(coff, dim=1).unsqueeze(1) #batch * feature

        return representation, feature

class transformer_fusion(nn.Module):
    def __init__(self,
                 dropout=0.2,
                 num_heads=8,
                 vocab_size=22,
                 num_encoder_layers=1,
                 d_model=128,  # 128
                 Max_len=45,  # 34 + 11 = 45
                 ):
        super(transformer_fusion, self).__init__()

        # self.embeddingLayer = nn.Embedding(vocab_size, d_model)
        self.positionalEncodings = nn.Parameter(torch.rand(Max_len, d_model), requires_grad=True)

        ##Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dim_feedforward=1024, dropout=dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers, encoder_norm)

    def forward(self, ConcatEmbedding, pad_mask):

        ConcatEmbedding = ConcatEmbedding + self.positionalEncodings[:ConcatEmbedding.shape[1], :]

        # input feed-forward:
        ConcatEmbedding = ConcatEmbedding.permute(1, 0, 2)  # seq * batch * feature
        Concatfeature = self.transformer_encoder(ConcatEmbedding, src_key_padding_mask=pad_mask)
        Concatfeature = Concatfeature.permute(1, 0, 2)  # batch * seq * feature


        coff = 1 - pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature, dim=1) / torch.sum(coff, dim=1).unsqueeze(1) #batch * feature

        return representation, Concatfeature

class mamba_fusion(nn.Module):
    def __init__(self, d_model=128, d_state=4, d_conv=2, expand=2,
                 Max_len=45
                 ):
        super(mamba_fusion, self).__init__()

        self.d_model = d_model
        self.pos_emb = nn.Parameter(torch.rand(Max_len, d_model), requires_grad=True)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # self.mamba2 = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=4)


    def forward(self, ConcatEmbedding, pad_mask):
        # ConcatEmbedding = ConcatEmbedding + self.pos_emb[:ConcatEmbedding.shape[1], :]

        Concatfeature = self.mamba(ConcatEmbedding)
        # Concatfeature = self.mamba2(ConcatEmbedding)


        coff = 1 - pad_mask.float()
        representation = torch.sum(coff.unsqueeze(2) * Concatfeature, dim=1) / torch.sum(coff, dim=1).unsqueeze(1) #[batch_size, d_model]

        return representation, Concatfeature

class Model_IM(nn.Module):
    def __init__(self,
                num_encoder_layers=1,
                d_embedding = 256, #128/384
                d_model = 128,
                ):
        super(Model_IM, self).__init__()

        self.tamba_peptide = xformer(num_heads=8, num_encoder_layers=num_encoder_layers, d_model=d_model, Max_len=11)
        self.tamba_mhc = xformer(num_heads=8, num_encoder_layers=num_encoder_layers, d_model=d_model, Max_len=34)

        self.fusion1 = transformer_fusion(num_heads=8, num_encoder_layers=1, d_model=d_model, Max_len=45)
        # self.fusion2 = MultiModalCrossAttention(d_model=d_model, num_heads=8, dropout=0.)
        # self.fusion2 = crossEncoderLayer(d_model=d_model, num_heads=8, dff=512, dropout_posffn=0.2, dropout_attn=0.)
        self.fusion3 = mamba_fusion(d_model=d_model, d_state=4, d_conv=2, expand=2, Max_len=45)

        # prediction layers for EL
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 =  nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 =  nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        
        # prediction layers for IM
        self.FC_1_IM = nn.Linear(d_embedding, 1024)
        self.BN_1_IM =  nn.BatchNorm1d(1024)
        self.FC_2_IM = nn.Linear(1024, 256)
        self.BN_2_IM =  nn.BatchNorm1d(256)
        self.predict_IM = nn.Linear(257, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self,peptide,mhcSeq):

        # Get padding mask
        # pad_mask = peptide.eq(21)
        ConcatSeq = torch.cat([mhcSeq, peptide], dim=1)
        mask = ConcatSeq.eq(21)

        mhc_representation, mhc_feature = self.tamba_mhc(mhcSeq) #[batch_size, seq_len, d_model]
        peptide_representation, peptide_feature = self.tamba_peptide(peptide) #[batch_size, seq_len, d_model]

        # representation = torch.cat([mhc_representation, peptide_representation], dim=1)
        representation = torch.cat([mhc_feature, peptide_feature], dim=1)

        representation1, feature1 = self.fusion1(representation, mask)
        # representation2 = self.fusion2(peptide_feature, mhc_feature, pad_mask) #[batch_size, seq_len, d_model]
        representation3, feature3 = self.fusion3(representation, mask)

        representation = torch.cat([representation1, representation3], dim=1) #[batch_size, 3*d_model]

        #Predict EL
        x_EL = self.fc1(representation)
        x_EL = self.bn1(x_EL)
        x_EL = self.relu(x_EL)
        x_EL = self.fc2(x_EL)
        x_EL = self.bn2(x_EL)
        x_EL = self.relu(x_EL)
        y_EL = self.sigmoid(self.outputlayer(x_EL)).detach()

        #Predict IM
        x_IM = self.FC_1_IM(representation)
        x_IM = self.BN_1_IM(x_IM)
        x_IM = self.relu(x_IM)
        x_IM = self.FC_2_IM(x_IM)
        x_IM = self.BN_2_IM(x_IM)
        x_IM = self.relu(x_IM)
        
        x_IM = torch.cat([y_EL,x_IM],dim=1)

        y_IM = self.sigmoid(self.predict_IM(x_IM))
        
        #Logical 
        return y_EL,y_IM

class baseline(nn.Module):
    def __init__(self,
                num_encoder_layers=1,
                d_embedding = 256, #128
                d_model=128,
                ):
        super(baseline, self).__init__()

        self.tamba_peptide = xformer(num_heads=8, num_encoder_layers=num_encoder_layers, d_model=d_model, Max_len=11)
        self.tamba_mhc = xformer(num_heads=8, num_encoder_layers=num_encoder_layers, d_model=d_model, Max_len=34)

        self.fusion1 = transformer_fusion(num_heads=8, num_encoder_layers=1, d_model=d_model, Max_len=45)
        # self.fusion2 = MultiModalCrossAttention(d_model=d_model, num_heads=8, dropout=0.)
        # self.fusion2 = crossEncoderLayer(d_model=d_model, num_heads=8, dff=512, dropout_posffn=0.2, dropout_attn=0.)
        self.fusion3 = mamba_fusion(d_model=d_model, d_state=4, d_conv=2, expand=2, Max_len=45)

        # prediction layers for EL
        self.fc1 = nn.Linear(d_embedding, 1024)
        self.bn1 =  nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.bn2 =  nn.BatchNorm1d(256)
        self.outputlayer = nn.Linear(256, 1)
        
        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,peptide,mhcSeq):

        # Get padding mask
        # pad_mask = peptide.eq(21)
        ConcatSeq = torch.cat([mhcSeq, peptide], dim=1)
        mask = ConcatSeq.eq(21)

        mhc_representation, mhc_feature = self.tamba_mhc(mhcSeq) #[batch_size, seq_len, d_model]
        peptide_representation, peptide_feature = self.tamba_peptide(peptide) #[batch_size, seq_len, d_model]

        # representation = torch.cat([mhc_representation, peptide_representation], dim=1)
        representation = torch.cat([mhc_feature, peptide_feature], dim=1)  # , seq_onehot, seq_opf, seq_zscale, seq_blosum62

        representation1, feature1 = self.fusion1(representation, mask)
        # representation2 = self.fusion2(peptide_feature, mhc_feature, pad_mask) #[batch_size, seq_len, d_model]
        representation3, feature3 = self.fusion3(representation, mask)

        representation = torch.cat([representation1, representation3], dim=1) #[batch_size, 3*d_model]

        #Predict
        x = self.fc1(representation)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        y = self.sigmoid(self.outputlayer(x))


        #Logical
        return y
