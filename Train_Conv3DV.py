import src.NetConv3DV7_9 as DeepDyn

import torch
import torch.optim as optim
import numpy as np

import sys
import os
import yaml

def main(yaml_path):#( Net_save_path, net_load, Train_data_path = '../data/train_center_full', Valid_data_path = '../data/Valid', ):
    with open(yaml_path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    train_data_path = data['train_data_path']
    valid_data_path = data['valid_data_path']
    net_save_path = data['net_save_path']
    var_save_path = data['var_save_path']
    net_load_path = data['net_load_path']
    
    SpDim = data['SpDim']
    HidDim = data['HidDim']
    KnnC = data['KnnC']
    Knn = data['Knn']
    lr = data['lr']
    param = data['param']
    NumEpoch = data['NumEpoch']

    Aff = data['Aff']
    Train_Ver = data['Train_Ver']

    Semantic = data['Semantic']
    net_load = data['net_load']
    Cadd = data['Cadd']


    structdata_train = []    
    filename = os.listdir(train_data_path)
    for imotion in range(len(filename)):
        structdata_train.append(DeepDyn.DynStructDataset(json_file= train_data_path+'/'+filename[imotion],BatchSize = 100,Knn = 99,train = 'Train', transform = DeepDyn.ToTensor()))

    structdata_valid = []    
    filename = os.listdir(valid_data_path)
    for imotion in range(len(filename)):
        structdata_valid.append(DeepDyn.DynStructDataset(json_file= valid_data_path + '/' +filename[imotion],BatchSize = 100,Knn = 99,train = 'Valid', transform = DeepDyn.ToTensor()))

    net  = DeepDyn.Net(SpDim, np.array(HidDim), KnnC, Aff = Aff, Semantic = Semantic)#(31*3, np.array([48, 24, 12, 12]), 20, Aff = 'Euc', Semantic = True)
    net = net.double()
    if net_load:
        net.load_state_dict(torch.load(net_load_path))#'./net_param/Conv3DV7_9_weak_first_full/3DynRec_net_Conv3D_epch0.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    #retain_graph=True
    optimizer = optim.Adam(net.parameters(), lr=lr)
    DeepDyn.train(net,optimizer, structdata_train, structdata_valid, param, 1.2, device ,var_save_path, PATH = net_save_path,Knn = Knn,KnnC = KnnC,Naffinity_min = 2, Naffinity_max = 2, NumEpoch = NumEpoch, Train_Ver = Train_Ver, Aff = Aff, Cadd = Cadd, Semantic = Semantic)
    #Loss =  (Loss3D_mean + Loss3D_meanRot + Loss3D_meanGlbRot + paramtrain[0]*(LossRec + LossRecRot + LossRecGlbRot)+paramtrain[1]*(LossLap + LossLapRot + LossLapGlbRot) + paramtrain[2]*(LossSymm + LossSymmRot + LossSymmGlbRot) + paramtrain[3]*(LossAHF + LossARotHF + LossAGlbRotHF) + paramtrain[4]*LossRot)/niter

if __name__ == "__main__":
    main(sys.argv[1])