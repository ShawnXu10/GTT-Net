import src.NetConv3DV7_9_PN as DeepDyn

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle_to_mat_converter

import os
import pickle

print(torch.__version__)

net  = DeepDyn.Net(30, np.array([30, 24, 12, 12]), np.array([30, 120, 48]), 20, Aff = 'Euc', Cadd = True)
net = net.double()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net.load_state_dict(torch.load('./net_param/Conv3DV7_9_super_first_PN_Cadd/3DynRec_net_Conv3D_epch35.pth'))

RecErrMean = np.zeros((1,3), dtype = np.double)
RecErrStd = np.zeros((1,3), dtype = np.double)
test = DeepDyn.test()

filename = os.listdir('dog_test/data')
for imotion in range(len(filename)):
    structdata_test = DeepDyn.DynStructDataset(json_file='dog_test/data/'+filename[imotion],BatchSize = 40,Knn = 20,train = 'Test', order = 1, transform = DeepDyn.ToTensor())
    X,A,ErrMean,ErrStd,_ = test(net,structdata_test,1.2, device,Knn = 10,KnnC = 20, Naffinity_min = 2, Naffinity_max = 2, niter = 1, order = 1, param = {'lambda1': 0.1, 'lambda2': 1e10}, Aff = 'Euc', Cadd = True, Semantic = True)

print(ErrMean)