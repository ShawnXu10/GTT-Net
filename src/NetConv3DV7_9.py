#with layer normalization
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torch.optim as optim
from PIL import Image

import math
import pickle

import json
import random as rd

from src.LossFunc import RecLoss, ReconError, X_opt_np
from src.gttlib import assginKnnFeature, LocalNormal2, AffinityGen, Variants2, Vec2Rot, PseudAffGen, ConstAffGen, GlobOrderGen, Valid

class DynStructDataset(Dataset):
    def __init__(self, json_file,BatchSize=40,Knn=10,train = 'Train', order = 0, transform=None):
        self.transform = transform
        
        with open(json_file) as data: # read data from json file
            self.data = json.load(data)
        self.output_data = []
        self.label = []
        if train == 'Test' or train == 'Valid':
            self.train = train
            self.data['C'] = np.array(self.data['C'])
            self.data['K'] = np.array(self.data['K'])
            self.data['camID'] = np.array(self.data['camID'])
            self.data['point3D'] = np.array(self.data['point3D'])
            self.data['ray'] = np.array(self.data['ray'])
            self.data['t'] = np.array(self.data['t'])
            if train == 'Valid':
                self.data['A_GT'] = np.array(self.data['A_GT'])
                

            nframes = self.data['point3D'].shape[1]
            njoints = round(self.data['point3D'].shape[0]/3)
        
            if order == 2:
                self.data['rayDInd'] = GlobOrderGen(self.data['camID'], nframes)#np.array(self.data['rayDInd'])-1
                self.data['X_init'] = np.array(self.data['X_initorder'])
            elif order == 1:
                self.data['rayDInd'] = np.array(self.data['rayDInd'])-1
                self.data['X_init'] = np.array(self.data['X_init'])
            elif order == 0:
                self.data['rayDInd'] = np.array(self.data['rayDInd_indep'])-1
                self.data['X_init'] = np.array(self.data['X_init_indep'])
                raydot = np.matmul(self.data['ray'].transpose(), self.data['ray'])#np.zeros((nframes, nframes))
                for f1 in range(nframes):
                    for f2 in range(f1+1,nframes):
                        N_valid = int(np.count_nonzero((self.data['ray'][:,f1] != 0) | (self.data['ray'][:,f2] != 0))/3)
                        raydot[f1, f2] = raydot[f1, f2]/(N_valid+1e-5)
                        raydot[f2, f1] = raydot[f1, f2]

                rayDotInd = np.flip(np.argsort(raydot, axis = 1), axis = 1)            
            #========================================= genereate A constant =================================================
            Acont = ConstAffGen(self.data['camID'])
            #================================================================================================================

            #========================================= X_inint based on W_init ==============================================
            A_init = np.zeros((nframes, nframes))
            A_init[np.arange(nframes),self.data['rayDInd'][:,0:2].transpose()] = 1.0
            A_init += Acont.detach().numpy()*0.1 
            X_init = X_opt_np(self.data['ray'], self.data['C'], A_init.transpose(), param = {'lambda1': 0.0015, 'lambda2': 1e10})
            #================================================================================================================

            if order == 0:
                Batchdata = {'point3DInit': X_init, 'rayDInd': self.data['rayDInd'], 'rayDotInd': rayDotInd}
            else:
                Batchdata = {'point3DInit': X_init, 'rayDInd': self.data['rayDInd']}#'point3DInit_W': X_init,
            self.output_data.append(Batchdata) 
            #======================generate lable===================
            if train == 'Test':
                Batchlable = {'point3D': self.data['point3D'], 'camID': self.data['camID'],'t': self.data['t'], 'K': self.data['K'], 'C': self.data['C'],  'ray': self.data['ray']}
            elif train == 'Valid':
                Batchlable = {'point3D': self.data['point3D'], 'A_GT': self.data['A_GT'], 'camID': self.data['camID'],'t': self.data['t'], 'K': self.data['K'], 'C': self.data['C'],  'ray': self.data['ray']}
            self.label.append(Batchlable)
        elif train == 'Train':
            self.train = train
            self.data['C'] = np.array(self.data['C'])
            self.data['K'] = np.array(self.data['K'])
            self.data['camID'] = np.array(self.data['camID'])
            self.data['point3D'] = np.array(self.data['point3D'])
            self.data['ray'] = np.array(self.data['ray'])
            self.data['t'] = np.array(self.data['t'])
            self.data['rayDInd'] = np.array(self.data['rayDInd'])-1
            self.data['X_init'] = np.array(self.data['X_init'])
            self.data['rayDInd_indep'] = np.array(self.data['rayDInd_indep'])-1
            self.data['X_init_indep'] = np.array(self.data['X_init_indep'])
            self.data['rayDIndOrder'] = np.array(self.data['rayDInd'])-1
            self.data['X_initorder'] = np.array(self.data['X_initorder'])
            self.data['rayRot'] = np.array(self.data['rayRot'])
            self.data['CRot'] = np.array(self.data['CRot'])
            self.data['ind_indepRot'] = np.array(self.data['ind_indepRot'])-1
            self.data['X_init_indepRot'] = np.array(self.data['X_init_indepRot'])
            self.data['CGlbRot'] = np.array(self.data['CGlbRot'])
            self.data['X_initGlbRot'] = np.array(self.data['X_initGlbRot'])
            self.data['A_GT'] = np.array(self.data['A_GT'])

            nframes = self.data['point3D'].shape[1]
            njoints = round(self.data['point3D'].shape[0]/3)
            #==================== if globale orders is known ===================
            self.data['rayDIndOrder'] = GlobOrderGen(self.data['camID'], nframes)
            #================================================================
            #========================================= genereate A constant =================================================
            Acont = ConstAffGen(self.data['camID'])
            
            A_init = np.zeros((nframes, nframes))
            A_initRot = np.zeros((nframes, nframes))
            A_initGlbRot = np.zeros((nframes, nframes))
            A_initIndep = np.zeros((nframes, nframes))
            A_initOrder = np.zeros((nframes, nframes))

            A_init[np.arange(nframes),self.data['rayDInd'][:,0:2].transpose()] = 1.0
            A_initRot[np.arange(nframes),self.data['ind_indepRot'][:,0:2].transpose()] = 1.0
            A_initIndep[np.arange(nframes),self.data['rayDInd_indep'][:,0:2].transpose()] = 1.0
            A_initOrder[np.arange(nframes),self.data['rayDIndOrder'][:,0:2].transpose()] = 1.0

            A_init += Acont.detach().numpy()*0.1
            A_initRot += Acont.detach().numpy()*0.1
            A_initIndep += Acont.detach().numpy()*0.1
            A_initOrder += Acont.detach().numpy()*0.1
            RotM = Vec2Rot(self.data['C'], self.data['CGlbRot'])

            X_init = X_opt_np(self.data['ray'], self.data['C'], A_init.transpose(), param = {'lambda1': 0.0015, 'lambda2': 1e10})
            X_initRot = X_opt_np(self.data['rayRot'], self.data['CRot'], A_initRot.transpose(), param = {'lambda1': 0.0015, 'lambda2': 1e10})
            X_initGlbRot = np.dot(RotM, X_init.transpose().reshape((-1,3)).transpose()).transpose().reshape((-1,93)).transpose()
            X_initIndep = X_opt_np(self.data['ray'], self.data['C'], A_initIndep.transpose(), param = {'lambda1': 0.0015, 'lambda2': 1e10})
            X_initOrder = X_opt_np(self.data['ray'], self.data['C'], A_initOrder.transpose(), param = {'lambda1': 0.0015, 'lambda2': 1e10})

            Batchdata = {'point3DInit': X_init, 'point3DInitRot': X_initRot, 'point3DInitGlbRot': X_initGlbRot,'point3DInitIndep': X_initIndep ,'point3DInitOrder': X_initOrder,\
                #'point3DInit': self.data['X_init'], 'point3DInitRot': self.data['X_init_indepRot'], 'point3DInitGlbRot': self.data['X_initGlbRot'],'point3DInitIndep': self.data['X_init_indep'] ,'point3DInitOrder': self.data['X_initorder'],\
                    'rayDInd': self.data['rayDInd'], 'rayDIndRot': self.data['ind_indepRot'], 'rayDIndGlbRot': self.data['rayDInd'],'rayDIndIndep': self.data['rayDInd_indep'],'rayDIndOrder': self.data['rayDIndOrder']}
            self.output_data.append(Batchdata) 
            #======================generate lable===================
            Batchlable = {'point3D': self.data['point3D'], 'A_GT': self.data['A_GT'], 'camID': self.data['camID'],'t': self.data['t'], 'K': self.data['K'], 'C': self.data['C'], 'ray': self.data['ray'], 'rayRot': self.data['rayRot'], 'CRot': self.data['CRot'], 'CGlbRot': self.data['CGlbRot']}
            self.label.append(Batchlable)
            
    def __len__(self):
        return(len(self.output_data))
        
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.output_data[idx]
        label = self.label[idx]
            
        if self.transform:
            self.transform(sample,label, self.train)

        return sample, label
                            
class ToTensor(object):
    def __call__(self, Batchdata,label, train):
        if not torch.is_tensor(Batchdata['point3DInit']):
            Batchdata['point3DInit'] = torch.from_numpy(Batchdata['point3DInit'])           

            label['C'] = torch.from_numpy(label['C'])
            label['K'] = torch.tensor(label['K'], dtype = torch.float64)
            label['point3D'] = torch.from_numpy(label['point3D'])
            label['ray'] = torch.from_numpy(label['ray'])
            if train == 'Train':
                Batchdata['point3DInitRot'] = torch.from_numpy(Batchdata['point3DInitRot'])
                Batchdata['point3DInitGlbRot'] = torch.from_numpy(Batchdata['point3DInitGlbRot'])
                Batchdata['point3DInitIndep'] = torch.from_numpy(Batchdata['point3DInitIndep'])
                Batchdata['point3DInitOrder'] = torch.from_numpy(Batchdata['point3DInitOrder'])
                label['CRot'] = torch.from_numpy(label['CRot'])
                label['rayRot'] = torch.from_numpy(label['rayRot'])   
                label['CGlbRot'] = torch.from_numpy(label['CGlbRot'])  
                label['A_GT'] = torch.from_numpy(label['A_GT'])      
            elif train == 'Valid':
                label['A_GT'] = torch.from_numpy(label['A_GT'])      

class Con3DFlat(nn.Module):
    def __init__(self, SpDim, n_points, feature_in, feature_out, use_bias = True, Semantic = True):
        super(Con3DFlat,self).__init__()
        self.use_bias = use_bias
        if use_bias:
            self.biasF = nn.Parameter(torch.Tensor(int(feature_out/3)), requires_grad=True)
            bound = math.sqrt(0.01) * math.sqrt(0.01/ (feature_in + feature_out))
            self.biasF.data.uniform_(-bound, bound)

            self.biasP = nn.Parameter(torch.Tensor(n_points), requires_grad=True)
            bound = math.sqrt(0.01) * math.sqrt(0.01/ (feature_in + n_points))
            self.biasP.data.uniform_(-bound, bound)
        njoints = int(SpDim/3)
        njointsOut = int(feature_out/3)
        self.P1 = nn.Linear(njoints, int(0.5*njoints+0.5*n_points))
        self.P2 = nn.Linear(int(0.5*njoints+0.5*n_points), int(0.25*njoints+0.75*n_points))
        self.P3 = nn.Linear(int(0.25*njoints+0.75*n_points), n_points)
        
        self.F1 = nn.Linear(n_points, int(0.5*n_points+0.5*njointsOut))
        self.F2 = nn.Linear(int(0.5*n_points+0.5*njointsOut), int(0.25*n_points+0.75*njointsOut))
        self.F3 = nn.Linear(int(0.25*n_points+0.75*njointsOut), njointsOut)

    def forward(self, KnnPoint3D, Semantic = True):
        njoints = int(KnnPoint3D.shape[0]/3)
        nframes = KnnPoint3D.shape[2]
        Knn = KnnPoint3D.shape[1]

        KnnPoint3D = KnnPoint3D.reshape(njoints,3,Knn, nframes)
        center = torch.mean(KnnPoint3D, dim = 0)
       
        SpWeightP = F.relu(self.P1((KnnPoint3D - KnnPoint3D[:,:,0:1,:]).permute(1,2,3,0)))
        SpWeightP = F.relu(self.P2(SpWeightP))
        SpWeightP = self.P3(SpWeightP).permute(3,0,1,2)

        if Semantic:
            SpWeightF = F.relu(self.F1((KnnPoint3D - center).permute(0,1,3,2)))#KnnPoint3D[0:1,:,:,:]
            SpWeightF = F.relu(self.F2(SpWeightF))
            SpWeightF = self.F3(SpWeightF).permute(0,1,3,2)
        else:
            SpWeightF = F.relu(self.F1((KnnPoint3D - KnnPoint3D[:,:,0:1,:]).permute(0,1,3,2)))
            SpWeightF = F.relu(self.F2(SpWeightF))
            SpWeightF = self.F3(SpWeightF).permute(0,1,3,2)

        KnnPoint3D = torch.matmul(KnnPoint3D.permute(1,3,2,0), SpWeightF.permute(1,3,0,2)) + self.biasF
        KnnPoint3D = (torch.matmul(KnnPoint3D.permute(0,1,3,2), SpWeightP.permute(1,3,2,0)) + self.biasP).permute(2,0,3,1)
        KnnPoint3D = KnnPoint3D.reshape(-1, Knn, nframes)
        
        return KnnPoint3D

class SimilarityNet(nn.Module):
    def __init__(self, dim):
        super(SimilarityNet, self).__init__()
        #=====================version 1============================
        self.FC = nn.Linear(dim*2, int(dim))
        self.FC2 = nn.Linear(int(dim), int(dim/2))
        self.FC3 = nn.Linear(int(dim/2), int(dim/4))
        self.FC4 = nn.Linear(int(dim/4), 1)
        #==========================================================
    def forward(self, feature):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nframes = feature.shape[2]
        Knn = feature.shape[1]-1
        #=====================version 1============================
        feature2 = feature[:,0,torch.arange(0,nframes).repeat(Knn,1)]
        feature = torch.cat((feature2, feature[:,1:Knn+1, :]), dim = 0)
        feature = F.relu(self.FC(feature.permute(1,2,0)))
        feature = F.relu(self.FC2(feature))
        feature = F.relu(self.FC3(feature))

        A = self.FC4(feature).view(feature.shape[0], feature.shape[1]).transpose(1,0)
        A = A/(torch.max(A)+1e-15)
        return A

class Net(nn.Module):
    def __init__(self, SpDim, dim, K, Aff = 'Learned', Semantic = True):
        super(Net,self).__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #encoder
        self.Enlinear1 = Con3DFlat(31*3, K+1, 31*3,dim[0], use_bias = True).to(device)
        self.Enlinear2 = Con3DFlat(dim[0], K+1, dim[0],dim[1], use_bias = True).to(device)#dim[0]
        self.Enlinear3 = Con3DFlat(dim[1], K+1, dim[1],dim[2], use_bias = True).to(device)#dim[1]
        self.Enlinear4 = Con3DFlat(dim[2], K+1, dim[2],dim[3], use_bias = True).to(device)#dim[2]
        
        #decoder
        self.Delinear1 = Con3DFlat(dim[3], K+1, dim[3],dim[2], use_bias = True).to(device)#dim[3]
        self.Delinear2 = Con3DFlat(2*dim[2], K+1, 2*dim[2],dim[1], use_bias = True).to(device)#2*dim[2]
        self.Delinear3 = Con3DFlat(2*dim[1], K+1, 2*dim[1],dim[0], use_bias = True).to(device)#2*dim[1]
        self.Delinear4 = Con3DFlat(2*dim[0], K+1, 2*dim[0],31*3, use_bias = True).to(device)#2*dim[0]         

        if Aff == 'Learned':
            self.SL = SimilarityNet(dim[3])

    def forward(self, Point3D, rayDInd, C, K,Knn, Aff = 'Learned', Cadd = True, Semantic = True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #============================Encoder=========================
        KnnPoint3D = assginKnnFeature(Point3D, rayDInd, K)
        KnnPoint3DNormal, _ = LocalNormal2(KnnPoint3D)

        latent1AE = self.Enlinear1(KnnPoint3DNormal, Semantic = Semantic)
        latent1AE, _ = LocalNormal2(latent1AE)
        KnnPoint3D = F.relu(latent1AE)
        
        latent2AE = self.Enlinear2(KnnPoint3D, Semantic = Semantic)
        latent2AE, _ = LocalNormal2(latent2AE)
        KnnPoint3D = F.relu(latent2AE)

        latent3AE = self.Enlinear3(KnnPoint3D, Semantic = Semantic)
        latent3AE, _ = LocalNormal2(latent3AE)
        KnnPoint3D = F.relu(latent3AE)

        latent4AE = self.Enlinear4(KnnPoint3D, Semantic = Semantic)
        latent4AE, _ = LocalNormal2(latent4AE)
        KnnPoint3D = F.relu(latent4AE)
        
        #===========================similarity learning================================
        latentFC1 = 0.0
        latentFC2 = 0.0
        latentFC3 = latent4AE
        #============================Decoder=========================
        latent1AD = self.Delinear1(KnnPoint3D, Semantic = Semantic)
        #concate data
        latent1AD, _ = LocalNormal2(latent1AD)
        Point3D = torch.cat((latent1AD,latent3AE), 0)
        KnnPoint3D = F.relu(Point3D)
        
        latent2AD = self.Delinear2(KnnPoint3D, Semantic = Semantic)
        #concate data
        latent2AD, _ = LocalNormal2(latent2AD)
        Point3D = torch.cat((latent2AD,latent2AE), 0)
        KnnPoint3D = F.relu(Point3D)

        latent3AD = self.Delinear3(KnnPoint3D, Semantic = Semantic)
        #concate data
        latent3AD, _ = LocalNormal2(latent3AD)
        Point3D = torch.cat((latent3AD,latent1AE), 0)
        KnnPoint3D = F.relu(Point3D)

        KnnPoint3D= self.Delinear4(KnnPoint3D, Semantic = Semantic)
        KnnPoint3D, _ = LocalNormal2(KnnPoint3D)
        
        if Aff == 'Learned':
            Af = self.SL(latentFC3[:,0:K+1])
        else:
            Af = AffinityGen(latentFC3[:,0,:], latentFC3[:,1:K+1].permute(0,2,1), Aff)

        return latent1AE,latent2AE,latent3AE,latent4AE,latentFC1,latentFC2,latentFC3,latent1AD,latent2AD,latent3AD,KnnPoint3DNormal,KnnPoint3D,Af

class train(object):
    def __init__(self, net,optimizer, structdata, structdata_valid, paramtrain,SpCont, device,VarPath, PATH = './3DynRec_netTmp.pth',Knn = 10,KnnC = 10, Naffinity_min = 2, Naffinity_max = 2, NumEpoch = 100, Train_Ver = 'first', Aff = 'Learned', Cadd = True, Semantic = True):

        Loss3D_meanReco = np.zeros(NumEpoch)
        Loss3D_meanGlbRotReco = np.zeros(NumEpoch)
        LossRecReco = np.zeros(NumEpoch)
        LossRecGlbRotReco = np.zeros(NumEpoch)
        LossLapReco = np.zeros(NumEpoch)
        LossLapGlbRotReco = np.zeros(NumEpoch)
        LossAHFReco = np.zeros(NumEpoch)
        LossAHFGlbRotReco = np.zeros(NumEpoch)
        LossTotal = np.zeros(NumEpoch)

        # Valid lose value
        RecMeanValid = np.zeros(NumEpoch)
        Loss3D_meanValid = np.zeros(NumEpoch)
        LossRecValid = np.zeros(NumEpoch)
        LossAHFValid = np.zeros(NumEpoch)
        LossTotalValid = np.zeros(NumEpoch)

        torch.autograd.set_detect_anomaly(True)

        # scheduler for learning rate
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 10, verbose = True, threshold = 1e-4)
        
        for epoch in range(NumEpoch):
            for imotion in range(len(structdata)):
                NumBatch = len(structdata[imotion])
                for ibatch in range(NumBatch):
                    
                    databatch, labelbatch = structdata[imotion][ibatch]
                    nframes = databatch['point3DInit'].shape[1]
                    njoints = int(databatch['point3DInit'].shape[0]/3)

                    if Train_Ver == 'super' or Train_Ver == 'super_first':
                        AHF = labelbatch['A_GT'].to(device)#PseudAffGen(labelbatch['camID']).to(device)
                    elif Train_Ver == 'unsuper' or Train_Ver == 'weak_first' or Train_Ver == 'weak':
                        AHF =  PseudAffGen(labelbatch['camID']).to(device)
                     
                    Loss3D_mean, LossRec, LossLap, LossAHF, _, A, A_full = Variants2(net, paramtrain, device, labelbatch['point3D'], AHF, databatch['point3DInit'], labelbatch['C'], labelbatch['ray'], databatch['rayDInd'], labelbatch['t'], databatch['rayDIndOrder'], labelbatch['camID'], Knn, KnnC, SpCont, Naffinity_min, Naffinity_max, Train_Ver, Aff, Cadd, order = 1, Vari = 'Orig', Semantic = Semantic)
                    Loss3D_meanGlbRot, LossRecGlbRot, LossLapGlbRot, LossAHFGlbRot, _, _, _ = Variants2(net, paramtrain, device, labelbatch['point3D'], AHF, databatch['point3DInitGlbRot'], labelbatch['C'], labelbatch['ray'], databatch['rayDIndGlbRot'], labelbatch['t'], databatch['rayDIndOrder'], labelbatch['camID'], Knn, KnnC, SpCont, Naffinity_min, Naffinity_max, Train_Ver, Aff, Cadd, order = 1, Vari = 'GlbRot', Semantic = Semantic)
                    
                    if Train_Ver == 'unsuper':
                        Loss = Loss3D_mean + Loss3D_meanGlbRot + paramtrain[0]*(LossRec + LossRecGlbRot)
                    elif Train_Ver == 'super_first' or Train_Ver == 'weak_first':
                        Loss = paramtrain[0]*(LossRec + LossRecGlbRot) + paramtrain[1]*(LossAHF + LossAHFGlbRot)
                    elif Train_Ver == 'weak' or Train_Ver == 'super':
                        Loss = Loss3D_mean + Loss3D_meanGlbRot + paramtrain[0]*(LossRec + LossRecGlbRot) + paramtrain[1]*(LossAHF + LossAHFGlbRot)

                    optimizer.zero_grad()                    
                    Loss.backward(retain_graph=True)
                    optimizer.step()

                    if epoch % 20 == 0 and imotion % 5 == 0:
                        img = Image.fromarray(A.cpu().detach().numpy()*225)
                        img.show()
                        img = Image.fromarray(A_full.cpu().detach().numpy()*225)
                        img.show()

                    Loss3D_meanReco[epoch] += Loss3D_mean.cpu().detach().numpy()
                    Loss3D_meanGlbRotReco[epoch] += Loss3D_meanGlbRot.cpu().detach().numpy()
                    LossRecReco[epoch] += LossRec.cpu().detach().numpy()
                    LossRecGlbRotReco[epoch] += LossRecGlbRot.cpu().detach().numpy()
                    LossLapReco[epoch] += LossLap.cpu().detach().numpy()
                    LossLapGlbRotReco[epoch] += LossLapGlbRot.cpu().detach().numpy()
                    LossAHFReco[epoch] += LossAHF.cpu().detach().numpy()
                    LossAHFGlbRotReco[epoch] += LossAHFGlbRot.cpu().detach().numpy()

            if epoch % 5 == 0:
                RecMeanValid[epoch],Loss3D_meanValid[epoch], LossRecValid[epoch], LossAHFValid[epoch], LossTotalValid[epoch] = Valid(net, structdata_valid, paramtrain, SpCont, device, Knn = Knn, KnnC = KnnC, Naffinity_min = 2, Naffinity_max = 2, niter = 1, order = 1, param = {'lambda1': 0.0015, 'lambda2': 1e10}, Train_Ver = 'super', Aff = Aff, Cadd = Cadd)
                print('===================== Valid ===================== [Iteration %d] [average RecError is %.8f] [average Loss3D is %.8f] [average LossRec is %.10f] [LossAHF is %.8f] [LossTotal is %.8f]' \
                    % (epoch, RecMeanValid[epoch], Loss3D_meanValid[epoch], LossRecValid[epoch], LossAHFValid[epoch], LossTotalValid[epoch] ))

                fvar = open(VarPath[0:len(VarPath)-5]+'_valid.pckl', 'wb')
                pickle.dump([RecMeanValid, Loss3D_meanValid, LossRecValid, LossAHFValid, LossTotalValid], fvar)
                fvar.close()
            
            NumBatch = 0
            for imotion in range(len(structdata)):
                NumBatch += len(structdata[imotion])

            Loss3D_meanReco[epoch] = Loss3D_meanReco[epoch]/NumBatch
            Loss3D_meanGlbRotReco[epoch] = Loss3D_meanGlbRotReco[epoch]/NumBatch
            LossRecReco[epoch] =LossRecReco[epoch]/NumBatch
            LossRecGlbRotReco[epoch] =LossRecGlbRotReco[epoch]/NumBatch
            LossLapReco[epoch] =LossLapReco[epoch]/NumBatch
            LossLapGlbRotReco[epoch] =LossLapGlbRotReco[epoch]/NumBatch
            LossAHFReco[epoch] = LossAHFReco[epoch]/NumBatch
            LossAHFGlbRotReco[epoch] = LossAHFGlbRotReco[epoch]/NumBatch

            if Train_Ver == 'unsuper':
                LossTotal[epoch] = Loss3D_meanReco[epoch] + Loss3D_meanGlbRotReco[epoch] + paramtrain[0]*(LossRecReco[epoch] + LossRecGlbRotReco[epoch])
            elif Train_Ver == 'weak_first' or Train_Ver == 'super_first':
                LossTotal[epoch] = paramtrain[0]*(LossRecReco[epoch] + LossRecGlbRotReco[epoch]) + paramtrain[1]*(LossAHFReco[epoch] + LossAHFGlbRotReco[epoch])
            elif Train_Ver == 'weak' or Train_Ver == 'super':
                LossTotal[epoch] = Loss3D_meanReco[epoch] + Loss3D_meanGlbRotReco[epoch] + paramtrain[0]*(LossRecReco[epoch] + LossRecGlbRotReco[epoch]) + paramtrain[1]*(LossAHFReco[epoch] + LossAHFGlbRotReco[epoch])

            scheduler.step(LossTotal[epoch])#update learning rate
            print('[Epoch %d] [average Loss3D is %.8f] [average Loss3DGlbRot is %.8f] [average LossRec is %.10f] [average LossRecGlbRot is %.10f] [LossAHF is %.8f] [LossAHFGlbRot is %.8f] [LossTotal is %.8f]' \
                % (epoch, Loss3D_meanReco[epoch], Loss3D_meanGlbRotReco[epoch], LossRecReco[epoch], LossRecGlbRotReco[epoch], LossAHFReco[epoch], LossAHFGlbRotReco[epoch], LossTotal[epoch] ))
            torch.save(net.state_dict(), PATH+ '_epch'+str(epoch)+'.pth')#str(epoch%2)
            fvar = open(VarPath, 'wb')
            pickle.dump([Loss3D_meanReco, Loss3D_meanGlbRotReco, LossRecReco, LossRecGlbRotReco, LossAHFReco, LossAHFGlbRotReco, LossTotal ], fvar)
            fvar.close()

class test(object):
    def __call__(self,net,structdata,SpCont, device, Knn = 10,KnnC = 10, Naffinity_min = 2, Naffinity_max = 2, niter = 1, order = 2, param = {'lambda1': 0.0015, 'lambda2': 1e10}, Aff = 'Learned', Cadd = False, Semantic = True):
        data, label = structdata[0]        
        nframes = data['point3DInit'].shape[1]
        njoints = int(data['point3DInit'].shape[0]/3)
        if order == 2:
            Knn = 2
        if order == 0:
            rayDotInd = data['rayDotInd']
 
        X = label['point3D'].to(device)
        for iter in range(niter):
            #X = X.to(device)
            A = torch.zeros(nframes,nframes,dtype = torch.float64).to(device)
            _, _, _, _, _, _, _,_,_,_,_,_,A_full = net(X, data['rayDInd'],label['C'].to(device), KnnC, Knn, Aff, Cadd, Semantic = Semantic)

            A[np.arange(nframes),data['rayDInd'][:,0:Knn].transpose()] = A_full[:,0:Knn].permute(1,0)
            _, indice = torch.sort(A, dim = 1, descending = True)
            A[np.arange(nframes),indice[:,2:Knn].transpose(0, 1)] = 0
            #============= genereate A constant ================
            if order == 1 or order == 2:
                Acont = ConstAffGen(label['camID']).to(device)
            elif order == 0:
                Acont = torch.zeros(nframes,nframes,dtype = torch.float64)
                Acont[np.arange(nframes), rayDotInd[:,0:2].transpose().copy()] = 1.0
                Acont = Acont.to(device)
            Acont = Acont*A[np.arange(nframes),indice[:,1:2].transpose(0, 1)].transpose(1,0)
            #===================================================
            if order == 1 or order == 0:
                A += Acont*0.01

            
            _, X = RecLoss(label['ray'].to(device), label['C'].to(device), label['point3D'].to(device), A, param = param)
            RecMean, RecStd, error = ReconError(X.cpu().detach().numpy(), label['point3D'].detach().numpy())
            if niter - iter > 1:
                #=========================update ray_indice based on K nearest neighbor=================================
                _, indice = torch.sort(A_full, dim = 1, descending = True)
                data['rayDInd'][:,0:20] = data['rayDInd'][np.arange(nframes),indice.cpu().detach().numpy().transpose()].transpose()
                #=========================================================================================================
        return X,A, RecMean, RecStd, error