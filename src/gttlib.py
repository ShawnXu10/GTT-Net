import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mpl_toolkits.mplot3d import Axes3D
# from sklearn.neighbors import NearestNeighbors
from PIL import Image
import matplotlib.pyplot as plt
from Animation.Animate3D import Animate3DSeleton

import time
import math
import pickle

import os

from src.LossFunc import RecLoss, LapLoss, LapLossV2, LapLossV2_2, LapLossV3, ReprojLoss, ReconError, X_opt

#================================================================================
#                              currently not using
#================================================================================
def ray_interact(r1, r2, C1, C2):
        njoints = int(r1.shape[0]/3)
        #SpDif = np.zeros((njoints+1,3))
        point3D = np.zeros((njoints,3))
        point3D_2 = np.zeros((njoints,3))
        for p in range(njoints):
            n1_2 = np.cross(r1[3*p:3*(p+1)].transpose(), np.cross(r1[3*p:3*(p+1)].transpose(),r2[3*p:3*(p+1)].transpose()))
            n2_1 = np.cross(r2[3*p:3*(p+1)].transpose(), np.cross(r2[3*p:3*(p+1)].transpose(),r1[3*p:3*(p+1)].transpose()))

            C1_2 = C1.transpose() + (np.matmul(n2_1, (C2-C1))/np.matmul(r1[3*p:3*(p+1)].transpose(), n2_1.transpose()))*r1[3*p:3*(p+1)].transpose()
            C2_1 = C2.transpose() + (np.matmul(n1_2, (C1-C2))/np.matmul(r2[3*p:3*(p+1)].transpose(), n1_2.transpose()))*r2[3*p:3*(p+1)].transpose()
            
            #SpDif[p,:] = C2_1 - C1_2
            point3D[p,:] = C1_2
            point3D_2[p,:] = C2_1

        #SpDif[njoints,:] = (C2-C1).transpose()
        dist = np.linalg.norm(point3D-point3D_2)/njoints
        return dist, point3D

def euler2Rotation(theta):
            R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])], [0, math.sin(theta[0]), math.cos(theta[0])]])
            R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])], [0, 1, 0], [-math.sin(theta[1]), 0, math.cos(theta[1])]])
            R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0], [math.sin(theta[2]), math.cos(theta[2]), 0], [0, 0, 1]])
            R = np.dot(R_z, np.dot(R_y, R_x))
            return R

def RandRotat(C,R,point3D):
#         fig = plt.figure(1)
#         ax = fig.add_subplot(111, projection='3d')
    center = np.zeros((3,1))
    center[:,0] = np.mean(point3D, axis = 0)
    RotM = euler2Rotation(np.random.rand(3)*2*math.pi-math.pi)
#         ax.scatter(C[0,0], C[1,0], C[2,0], marker = '+')
#         ax.scatter(point3D[:,0], point3D[:,1], point3D[:,2], marker = 'o')
    C = np.matmul(RotM,(C - center)) + center + np.random.rand(3,1)*2*20-20
    R = np.matmul(RotM,R)

    
    ray = point3D - C.transpose()
    tmpnorm = np.zeros((ray.shape[0],1))
    tmpnorm[:,0] = np.linalg.norm(ray, axis = 1)
    ray = ray/tmpnorm
    plucker = np.concatenate((ray, np.cross(np.concatenate(C.transpose(), axis = 0),ray)), axis = 1)
    tmpnorm[:,0] = np.linalg.norm(plucker, axis = 1)
    plucker = plucker/tmpnorm
#         ax.scatter(C[0,0], C[1,0], C[2,0], marker = '^')     
#         #ax.set_aspect('equal')
#         plt.show()
     
    return plucker, ray, C
#====================================================================================================
#                              currently not using
#====================================================================================================


#====================================================================================================
#                              function for network
#====================================================================================================
def Vec2Rot(X1, X2):
    X1 = X1
    X1 = X1# - np.mean(X1, axis = 1).reshape(3,1)
    X2 = X2# - np.mean(X2, axis = 1).reshape(3,1)
    H = np.dot(X2, X1.transpose())
    U, S, V = np.linalg.svd(H)
    R = np.dot(U, V)

    if np.linalg.det(R) < 0:
        V[2,:] *= -1
        R = np.dot(U, V)

    return R
def assginKnnFeature(Point3D, rayDInd,K):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        KnnPoint3D = torch.zeros(Point3D.shape[0],K+1,Point3D.shape[1],dtype = torch.float64).to(device)
        KnnPoint3D[:,0,:] = Point3D[:,:]
        KnnPoint3D[:,1:K+1,:] = Point3D[:,rayDInd[:,0:K].transpose()]
        return KnnPoint3D
def LayerNormSelf(point3D):
    if len(point3D.shape) == 2:
        njoints = int(point3D.shape[0]/3)
        nframes = point3D.shape[1]
        point3D = point3D.view(njoints, 3, nframes)
        mean = torch.mean(point3D, dim = [0, 2])
        var = torch.var(point3D, dim = [0, 2])
        point3D_norm = (point3D - mean.view(1, 3, 1))/torch.sqrt(var+1e-5).view(1, 3, 1)
        point3D_norm = point3D_norm.view(point3D.shape[0]*point3D.shape[1], point3D.shape[2])
    elif len(point3D.shape) == 3:
        njoints = point3D.shape[0]
        nframes = point3D.shape[2]
        mean = torch.mean(point3D, dim = [0, 2])
        var = torch.var(point3D, dim = [0, 2])
        point3D_norm = (point3D - mean.view(1, point3D.shape[1], 1))/torch.sqrt(var+1e-5).view(1, point3D.shape[1], 1)
    return point3D_norm, mean, var
def LocalNormal(KnnPoint3D):
    njoints = int(KnnPoint3D.shape[0]/3)
    Knn = KnnPoint3D.shape[1]
    nframes = KnnPoint3D.shape[2]
    CenterPoint = torch.mean(KnnPoint3D[:,0:1,:].reshape(njoints,3,1,nframes), dim = 0)
    #CenterPoint = KnnPoint3D[0:3,0:1,:]
    KnnPoint3DNormal = (KnnPoint3D.reshape(njoints,3,Knn,nframes) - CenterPoint).reshape(njoints*3,Knn,nframes)
    return KnnPoint3DNormal, CenterPoint

def LocalNormal2(KnnPoint3D):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    njoints = int(KnnPoint3D.shape[0]/3)
    Knn = KnnPoint3D.shape[1]
    nframes = KnnPoint3D.shape[2]
    VarPoint = torch.zeros(3,1,1,dtype = torch.float64).to(device)

    CenterPoint = torch.mean(KnnPoint3D[:,0:1,:].reshape(njoints,3,1,nframes), dim = 0)
    VarPoint[:,0,0] = torch.var(KnnPoint3D.reshape(njoints,3,Knn,nframes), dim = [0,2,3])
    KnnPoint3DNormal = ((KnnPoint3D.reshape(njoints,3,Knn,nframes) - CenterPoint)/torch.sqrt(VarPoint+1e-5)).reshape(njoints*3,Knn,nframes)
    return KnnPoint3DNormal, CenterPoint
#====================================================================================================
#                              function for network
#====================================================================================================


#====================================================================================================
#                              funciton for training
#====================================================================================================
def PairwiseArcDist(feature, timeindex):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nframes = feature.shape[1]
        njoints = int(feature.shape[0]/3)
        feature = feature[:, timeindex]
        featurediff = feature[:,0:(nframes-1)]-feature[:,1:(nframes)]
        prev_cur_dist = torch.norm(featurediff+1e-8, dim = 0)
        pwdist = torch.zeros(nframes,nframes,dtype = torch.float64).to(device)
        for f in range(0,nframes-1):
            pwdist[f, (f+1):nframes] = torch.cumsum(prev_cur_dist[f:nframes-1], dim = 0)  
            pwdist[(f+1):nframes, f] = pwdist[f, (f+1):nframes]
        return pwdist

def AffinityGen(feature, feature2, Aff):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nframes = feature.shape[1]
    if Aff == 'Euc':
        #A = 1.0/(1.0 + torch.exp(torch.sum((feature -feature2.permute(2,0,1)).permute(1,2,0)**2, dim = 0).transpose(0, 1)))
        #A = 1.0/(torch.exp(torch.sum((feature -feature2.permute(2,0,1)).permute(1,2,0)**2, dim = 0).transpose(0, 1))+1e-3)
        A = 1.0/(torch.exp(torch.sum(torch.abs((feature -feature2.permute(2,0,1)).permute(1,2,0)), dim = 0).transpose(0, 1))+1e-1).permute(1,0)
    elif Aff == 'Cos':
        #A = (torch.sum((feature*feature2.permute(2,0,1)).permute(1,2,0), dim = 0)).transpose(0, 1) + 1.0
        feature = feature/torch.norm(feature+1e-15, dim = 0)
        A = torch.matmul(feature.t(), feature) + 1.0
        #/(torch.norm(feature, dim = 0)*torch.norm(feature[:,rayDInd[:,0:Knn]], dim = 0).permute(1,0)).permute(1,0)
    A = A/(torch.max(A)+1e-15)
    #A[np.arange(nframes), np.arange(nframes)] = torch.tensor(0.0,dtype = torch.float64).to(device)
    return A

def PseudAffGen(camID):
    # =======================================create pseudo groudtruth affinity matrix=================================================
    nframes = len(camID)
    AHF = torch.zeros(nframes,nframes,dtype = torch.float64)
    for f in range(nframes):
        for f2 in range(f+1,nframes):
            if camID[f] != camID[f2]:
                AHF[f,f2] = 1
                break
        for f2 in range(f-1,-1,-1):
            if camID[f] != camID[f2]:
                AHF[f,f2] = 1
                break    
    AHF = AHF#.to(device)
    return AHF
def PseudAffGen2(camID):
    # =======================================create pseudo groudtruth affinity matrix=================================================
    nframes = len(camID)
    AHF = torch.zeros(nframes,nframes,dtype = torch.float64)
    for f in range(nframes):
        for f2 in range(f+1,nframes):
            if camID[f] != camID[f2]:
                AHF[f,f2] = 1.0/np.abs(camID[f2]-camID[f])**0.8
                break
        for f2 in range(f-1,-1,-1):
            if camID[f] != camID[f2]:
                AHF[f,f2] = 1.0/np.abs(camID[f2]-camID[f])**0.8
                break    
    AHF = AHF/torch.max(AHF)#.to(device)
    return AHF

def ConstAffGen(camID):
    #========================================= genereate A constant =================================================
    nframes = len(camID)
    Acont = torch.zeros(nframes,nframes,dtype = torch.float64)
    camIDunique = np.unique(camID)
    camlen = torch.zeros(len(camIDunique)+1,dtype = torch.int)
    idxtmp = np.where(camID == camIDunique[0])[0]
    for i in range(1,len(camIDunique)):
        camlen[i] = len(np.where(camID == camIDunique[i-1])[0]) + camlen[i-1]
        idxtmp  = np.concatenate([idxtmp, np.where(camID == camIDunique[i])[0]])
    camlen[len(camIDunique)] = len(np.where(camID == camIDunique[len(camIDunique)-1])[0]) + camlen[len(camIDunique)-1]

    for i in range(len(camIDunique)):
        Acont[camlen[i]:camlen[i+1]-1,1+camlen[i]:camlen[i+1]] += torch.diag(torch.ones(camlen[i+1]-camlen[i]-1,dtype = torch.float64))
        Acont[1+camlen[i]:camlen[i+1],camlen[i]:camlen[i+1]-1] += torch.diag(torch.ones(camlen[i+1]-camlen[i]-1,dtype = torch.float64))
    idxtmp = np.argsort(idxtmp)
    # cam0 = np.where(camID == 0)
    # cam1 = np.where(camID == 1)
    # cam2 = np.where(camID == 2)
    # cam3 = np.where(camID == 3)
    # cam4 = np.where(camID == 4)
    # cam5 = np.where(camID == 6)

    # idxtmp = np.argsort(np.concatenate([cam0[0],cam1[0], cam2[0], cam3[0]]))

    # Acont[0:len(cam0[0])-1,1:len(cam0[0])] += torch.diag(torch.ones(len(cam0[0])-1,dtype = torch.float64))
    # Acont[1:len(cam0[0]),0:len(cam0[0])-1] += torch.diag(torch.ones(len(cam0[0])-1,dtype = torch.float64))
    # Acont[len(cam0[0]):len(cam1[0])+len(cam0[0])-1,1+len(cam0[0]):len(cam1[0])+len(cam0[0])] += torch.diag(torch.ones(len(cam1[0])-1,dtype = torch.float64))
    # Acont[1+len(cam0[0]):len(cam1[0])+len(cam0[0]),len(cam0[0]):len(cam1[0])+len(cam0[0])-1] += torch.diag(torch.ones(len(cam1[0])-1,dtype = torch.float64))
    # Acont[len(cam1[0])+len(cam0[0]):len(cam2[0])+len(cam1[0])+len(cam0[0])-1,1+len(cam1[0])+len(cam0[0]):len(cam2[0])+len(cam1[0])+len(cam0[0])] += torch.diag(torch.ones(len(cam2[0])-1,dtype = torch.float64))
    # Acont[1+len(cam1[0])+len(cam0[0]):len(cam2[0])+len(cam1[0])+len(cam0[0]),len(cam1[0])+len(cam0[0]):len(cam2[0])+len(cam1[0])+len(cam0[0])-1] += torch.diag(torch.ones(len(cam2[0])-1,dtype = torch.float64))
    # Acont[len(cam2[0])+len(cam1[0])+len(cam0[0]):len(cam3[0])+len(cam2[0])+len(cam1[0])+len(cam0[0])-1,1+len(cam2[0])+len(cam1[0])+len(cam0[0]):len(cam3[0])+len(cam2[0])+len(cam1[0])+len(cam0[0])] += torch.diag(torch.ones(len(cam3[0])-1,dtype = torch.float64))
    # Acont[1+len(cam2[0])+len(cam1[0])+len(cam0[0]):len(cam3[0])+len(cam2[0])+len(cam1[0])+len(cam0[0]),len(cam2[0])+len(cam1[0])+len(cam0[0]):len(cam3[0])+len(cam2[0])+len(cam1[0])+len(cam0[0])-1] += torch.diag(torch.ones(len(cam3[0])-1,dtype = torch.float64))

    Acont = Acont[idxtmp,:]
    Acont = Acont[:,idxtmp]#.to(device)
    return Acont

def GlobOrderGen(camID, Knn):
    #==================== if globale orders is known ===================
    nframes = len(camID)
    rayDInd = np.zeros((nframes, nframes), dtype = int)    

    for f1 in range(nframes):
        Knn_cur = 0
        time_dist_left = 0
        time_dist_right = 0
        while True:
            while True:
                time_dist_right += 1
                # if time_dist_right+time_dist_left == Knn:
                #     break                        
                if  f1+time_dist_right  < nframes:
                    if camID[f1+time_dist_right] != camID[f1]:
                        rayDInd[f1, Knn_cur] = f1+time_dist_right
                        Knn_cur += 1
                        break
                else:
                    break
            if Knn_cur == Knn:
                break
            
            while True:
                time_dist_left += 1
                # if time_dist_right+time_dist_left == Knn:
                #     break                        
                if  f1-time_dist_left  >= 0:
                    if camID[f1-time_dist_left] != camID[f1]:
                        rayDInd[f1, Knn_cur] = f1-time_dist_left
                        Knn_cur += 1
                        break
                else:
                    break
            if Knn_cur == Knn:
                break
            if f1+time_dist_right  >= nframes and f1-time_dist_left  < 0:
                break
    return rayDInd

def Variants2(net, paramtrain, device, point3D, AHF, point3DInit, C, ray, rayDInd, t, rayDIndOrder,camID, Knn, KnnC, SpCont, Naffinity_min, Naffinity_max, Train_Ver, Aff, Cadd, order = 1, Vari = 'Orig', Semantic = True):    
    #Animate3DSeleton(labelbatch['point3D'].cpu().detach().numpy())
    #Animate3DSeleton(databatch['point3DInitGlbRot'].cpu().detach().numpy())
    nframes = point3DInit.shape[1]
    njoints = int(point3DInit.shape[0]/3)

    A = torch.zeros(nframes,nframes, dtype = torch.float64).to(device)
    A_sparse = torch.zeros(nframes,nframes, dtype = torch.float64).to(device)

    X = point3DInit.to(device)
    latentP1Hiden, latentP2Hiden, latentP3Hiden, latent, _, _, _,latentD1AD,latentD2AD,latentD3AD,X_norm,out,A_full = net(X, rayDInd, C.to(device), KnnC, Knn, Aff, Cadd, Semantic = Semantic)

    LossRec = torch.norm(out-X_norm)/(out.shape[0]*out.shape[1]*out.shape[2])+torch.norm(latentD3AD-latentP1Hiden)/(latentD3AD.shape[0]*latentD3AD.shape[1]*latentD3AD.shape[2])+torch.norm(latentD2AD-latentP2Hiden)/(latentD2AD.shape[0]*latentD2AD.shape[1]*latentD2AD.shape[2])+torch.norm(latentD1AD-latentP3Hiden)/(latentD1AD.shape[0]*latentD1AD.shape[1]*latentD1AD.shape[2])

    
    A_sparse[np.arange(nframes),rayDInd[:,0:KnnC].transpose()] = A_full.permute(1,0)

    weight_A = 1.2/(AHF+0.2)
    if order == 0 or order == 1:
        
        A[np.arange(nframes),rayDInd[:,0:Knn].transpose()] = A_full[:,0:Knn].permute(1,0)
        _, indice = torch.sort(A, dim = 1, descending = True)
        A[np.arange(nframes),indice[:,2:Knn].transpose(0, 1)] = 0.0
        LossAHF = torch.norm((A_sparse[np.arange(nframes),rayDInd[:,0:KnnC].transpose()] - AHF[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]))/(nframes)\
            + 1.0*torch.norm((A[np.arange(nframes),indice[:,0:2].transpose(0, 1)] - AHF[np.arange(nframes),indice[:,0:2].transpose(0, 1)]))/nframes#*weight_A[np.arange(nframes),indice[:,0:2].transpose(0, 1)]##*weight_A[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]
        # A[np.arange(A.shape[0]),indice[:,Naffinity_min:Naffinity_max].transpose(0, 1)] = torch.where(A[np.arange(A.shape[0]),indice[:,Naffinity_min:Naffinity_max].transpose(0, 1)] > SpCont, A[np.arange(A.shape[0]),indice[:,Naffinity_min:Naffinity_max].transpose(0, 1)], torch.tensor(0.0,dtype = torch.float64).to(device))
    else:
        A[np.arange(nframes),rayDIndOrder[:,0:2].transpose()] = A_full[:,0:2].permute(1,0)
        LossAHF = torch.norm((A_sparse[np.arange(nframes),rayDInd[:,0:KnnC].transpose()] - AHF[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]))/(nframes)\
            + 1.0*torch.norm((A[np.arange(nframes),rayDIndOrder[:,0:2].transpose()] - AHF[np.arange(nframes),rayDIndOrder[:,0:2].transpose()]))/nframes#*weight_A[np.arange(nframes),rayDIndOrder[:,0:2].transpose()]#*weight_A[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]
    #============= genereate A constant ================
    Acont = ConstAffGen(camID).to(device)
    if order == 1 or order == 0:
        Weight_A = A[np.arange(nframes),indice[:,1:2].transpose(0, 1)].transpose(1,0)
        Acont = Acont*Weight_A
    else:
        Weight_A = A[np.arange(nframes),rayDIndOrder[:,1:2].transpose()].transpose(1,0)
        Acont = Acont*Weight_A
    #===================================================
    A += Acont*0.2

    LossLap = torch.tensor(0.0,dtype = torch.float64).to(device)
    # LossAHF = torch.norm(A_sparse - AHF+1e-8)/(nframes) + 1.0*torch.norm(A - AHF+1e-8)/nframes#-1
    Loss3D_meantmp, Xtmp = RecLoss(ray.to(device), C.to(device), point3D.to(device), A, param = {'lambda1': 0.1, 'lambda2': 1e10})

    if Train_Ver == 'unsuper':
        Lout = torch.diag(torch.sum(A_sparse.t(),dim = 0)) - A_sparse.t()
        L = torch.diag(torch.sum(A_sparse + A_sparse.t(),dim = 0)) - (A_sparse + A_sparse.t())
        Xtmp = Xtmp.to(device)
        Loss3D_mean = torch.norm(torch.mm(Xtmp,Lout))/(nframes*njoints) + 0.0015*torch.trace(torch.mm(torch.mm(Xtmp, L), Xtmp.t()))/(nframes*njoints)
    elif Train_Ver == 'weak_first' or Train_Ver == 'super_first':
        Loss3D_mean = torch.tensor(0.0,dtype = torch.float64).to(device)
    elif Train_Ver == 'weak':
        Lout = torch.diag(torch.sum(AHF.t(),dim = 0)) - AHF.t()
        L = torch.diag(torch.sum(AHF + AHF.t(),dim = 0)) - (AHF + AHF.t())
        Xtmp = Xtmp.to(device)
        Loss3D_mean = torch.norm(torch.mm(Xtmp,Lout))/(nframes*njoints) + 0.0015*torch.trace(torch.mm(torch.mm(Xtmp, L), Xtmp.t()))/(nframes*njoints)
    elif Train_Ver == 'super':
        Loss3D_mean = Loss3D_meantmp.to(device)

    # Loss = paramtrain[0]*(LossRec) + paramtrain[2]*(LossAHF)
    # optimizer.zero_grad()                    
    # Loss.backward(retain_graph=True)
    # optimizer.step()
    return Loss3D_mean, LossRec, LossLap, LossAHF, Xtmp, A, A_sparse

def VariantsPN(net, paramtrain, device, point3D, AHF, point3DInit, C, ray, rayDInd, t, rayDIndOrder,camID, Knn, KnnC, SpCont, Naffinity_min, Naffinity_max, Train_Ver, Aff, Cadd, limbSeq,label, train = 'Train', order = 1, Vari = 'Orig', Semantic = True):    
    #Animate3DSeleton(labelbatch['point3D'].cpu().detach().numpy())
    #Animate3DSeleton(databatch['point3DInitGlbRot'].cpu().detach().numpy())
    nframes = point3DInit.shape[1]
    njoints = int(point3DInit.shape[0]/3)

    A = torch.zeros(nframes,nframes, dtype = torch.float64).to(device)
    A_sparse = torch.zeros(nframes,nframes, dtype = torch.float64).to(device)

    X = point3DInit.to(device)
    latentP1Hiden, latentP2Hiden, latentP3Hiden, latentD1AD,latentD2AD,latentD3AD,X_norm,out,Point3DNormal,PNout,A_full = net(X, rayDInd, C.to(device), KnnC, Knn, limbSeq,label, train = train, Aff = Aff,Cadd = Cadd, Semantic = Semantic)

    if train == 'Train':
        LossRec = torch.norm(PNout-Point3DNormal)/(PNout.shape[0]*PNout.shape[1])+torch.norm(out-X_norm)/(out.shape[0]*out.shape[1]*out.shape[2])+torch.norm(latentD3AD-latentP1Hiden)/(latentD3AD.shape[0]*latentD3AD.shape[1]*latentD3AD.shape[2])+torch.norm(latentD2AD-latentP2Hiden)/(latentD2AD.shape[0]*latentD2AD.shape[1]*latentD2AD.shape[2])+torch.norm(latentD1AD-latentP3Hiden)/(latentD1AD.shape[0]*latentD1AD.shape[1]*latentD1AD.shape[2])
    elif train == 'Valid':
        LossRec = torch.norm(out-X_norm)/(out.shape[0]*out.shape[1]*out.shape[2])+torch.norm(latentD3AD-latentP1Hiden)/(latentD3AD.shape[0]*latentD3AD.shape[1]*latentD3AD.shape[2])+torch.norm(latentD2AD-latentP2Hiden)/(latentD2AD.shape[0]*latentD2AD.shape[1]*latentD2AD.shape[2])+torch.norm(latentD1AD-latentP3Hiden)/(latentD1AD.shape[0]*latentD1AD.shape[1]*latentD1AD.shape[2])
    
    A_sparse[np.arange(nframes),rayDInd[:,0:KnnC].transpose()] = A_full.permute(1,0)

    weight_A = 1.2/(AHF+0.2)
    if order == 0 or order == 1:
        
        A[np.arange(nframes),rayDInd[:,0:Knn].transpose()] = A_full[:,0:Knn].permute(1,0)
        _, indice = torch.sort(A, dim = 1, descending = True)
        A[np.arange(nframes),indice[:,2:Knn].transpose(0, 1)] = 0.0
        LossAHF = torch.norm((A_sparse[np.arange(nframes),rayDInd[:,0:KnnC].transpose()] - AHF[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]))/(nframes)\
            + 1.0*torch.norm((A[np.arange(nframes),indice[:,0:2].transpose(0, 1)] - AHF[np.arange(nframes),indice[:,0:2].transpose(0, 1)]))/nframes#*weight_A[np.arange(nframes),indice[:,0:2].transpose(0, 1)]##*weight_A[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]
        # A[np.arange(A.shape[0]),indice[:,Naffinity_min:Naffinity_max].transpose(0, 1)] = torch.where(A[np.arange(A.shape[0]),indice[:,Naffinity_min:Naffinity_max].transpose(0, 1)] > SpCont, A[np.arange(A.shape[0]),indice[:,Naffinity_min:Naffinity_max].transpose(0, 1)], torch.tensor(0.0,dtype = torch.float64).to(device))
    else:
        A[np.arange(nframes),rayDIndOrder[:,0:2].transpose()] = A_full[:,0:2].permute(1,0)
        LossAHF = torch.norm((A_sparse[np.arange(nframes),rayDInd[:,0:KnnC].transpose()] - AHF[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]))/(nframes)\
            + 1.0*torch.norm((A[np.arange(nframes),rayDIndOrder[:,0:2].transpose()] - AHF[np.arange(nframes),rayDIndOrder[:,0:2].transpose()]))/nframes#*weight_A[np.arange(nframes),rayDIndOrder[:,0:2].transpose()]#*weight_A[np.arange(nframes),rayDInd[:,0:KnnC].transpose()]
    #============= genereate A constant ================
    Acont = ConstAffGen(camID).to(device)
    if order == 1 or order == 0:
        Weight_A = A[np.arange(nframes),indice[:,1:2].transpose(0, 1)].transpose(1,0)
        Acont = Acont*Weight_A
    else:
        Weight_A = A[np.arange(nframes),rayDIndOrder[:,1:2].transpose()].transpose(1,0)
        Acont = Acont*Weight_A
    #===================================================
    A += Acont*0.3

    LossLap = torch.tensor(0.0,dtype = torch.float64).to(device)
    # LossAHF = torch.norm(A_sparse - AHF+1e-8)/(nframes) + 1.0*torch.norm(A - AHF+1e-8)/nframes#-1
    Loss3D_meantmp, Xtmp = RecLoss(ray.to(device), C.to(device), point3D.to(device), A, param = {'lambda1': 0.1, 'lambda2': 1e10})

    if Train_Ver == 'unsuper':
        Lout = torch.diag(torch.sum(A_sparse.t(),dim = 0)) - A_sparse.t()
        L = torch.diag(torch.sum(A_sparse + A_sparse.t(),dim = 0)) - (A_sparse + A_sparse.t())
        Xtmp = Xtmp.to(device)
        Loss3D_mean = torch.norm(torch.mm(Xtmp,Lout))/(nframes*njoints) + 0.0015*torch.trace(torch.mm(torch.mm(Xtmp, L), Xtmp.t()))/(nframes*njoints)
    elif Train_Ver == 'weak_first' or Train_Ver == 'super_first':
        Loss3D_mean = torch.tensor(0.0,dtype = torch.float64).to(device)
    elif Train_Ver == 'weak':
        Lout = torch.diag(torch.sum(AHF.t(),dim = 0)) - AHF.t()
        L = torch.diag(torch.sum(AHF + AHF.t(),dim = 0)) - (AHF + AHF.t())
        Xtmp = Xtmp.to(device)
        Loss3D_mean = torch.norm(torch.mm(Xtmp,Lout))/(nframes*njoints) + 0.0015*torch.trace(torch.mm(torch.mm(Xtmp, L), Xtmp.t()))/(nframes*njoints)
    elif Train_Ver == 'super':
        Loss3D_mean = Loss3D_meantmp.to(device)

    # Loss = paramtrain[0]*(LossRec) + paramtrain[2]*(LossAHF)
    # optimizer.zero_grad()                    
    # Loss.backward(retain_graph=True)
    # optimizer.step()
    return Loss3D_mean, LossRec, LossLap, LossAHF, Xtmp, A, A_sparse

def Valid(net, validata, paramtrain, SpCont, device, Knn = 10, KnnC = 20, Naffinity_min = 2, Naffinity_max = 2, niter = 1, order = 1, param = {'lambda1': 0.0015, 'lambda2': 1e10}, Train_Ver = 'first', Aff = 'Learned', Cadd = False):

    RecMeanReco = 0.0
    Loss3D_meanReco = 0.0
    LossRecReco = 0.0
    LossAHFReco = 0.0
    LossTotal = 0.0
    
    for imotion in range(len(validata)):
        NumBatch = len(validata[imotion])
        for ibatch in range(NumBatch):
            data, label = validata[imotion][ibatch]
            nframes = data['point3DInit'].shape[1]
            njoints = int(data['point3DInit'].shape[0]/3)

            AHF = label['A_GT'].to(device)

            Loss3D_mean, LossRec, _, LossAHF, X, A, _ = Variants2(net, paramtrain, device, label['point3D'], AHF, data['point3DInit'], label['C'], label['ray'], data['rayDInd'], label['t'], data['rayDInd'],label['camID'], Knn, KnnC, SpCont, Naffinity_min, Naffinity_max, Train_Ver, Aff, Cadd, order = 1, Vari = 'Orig')
            RecMean, _, _ = ReconError(X.cpu().detach().numpy(), label['point3D'].detach().numpy())

            A = A.cpu().detach().numpy()
            AHF = AHF.cpu().detach().numpy()
            Loss3D_mean = Loss3D_mean.cpu().detach().numpy()
            LossRec = LossRec.cpu().detach().numpy()
            LossAHF = LossAHF.cpu().detach().numpy()

            RecMeanReco += RecMean
            Loss3D_meanReco += Loss3D_mean
            LossRecReco += LossRec
            LossAHFReco += LossAHF
            LossTotal += Loss3D_mean + paramtrain[0]*(LossRec) + paramtrain[1]*(LossAHF)

    RecMeanReco /= len(validata)
    Loss3D_meanReco /= len(validata)
    LossRecReco /= len(validata)
    LossAHFReco /= len(validata)
    LossTotal /= len(validata)

    return RecMeanReco, Loss3D_meanReco, LossRecReco, LossAHFReco, LossTotal

def ValidPN(net, validata, paramtrain, SpCont, device, Knn = 10, KnnC = 20, Naffinity_min = 2, Naffinity_max = 2, niter = 1, order = 1, param = {'lambda1': 0.0015, 'lambda2': 1e10}, Train_Ver = 'first', Aff = 'Learned', Cadd = False):

    RecMeanReco = 0.0
    Loss3D_meanReco = 0.0
    LossRecReco = 0.0
    LossAHFReco = 0.0
    LossTotal = 0.0
    
    for imotion in range(len(validata)):
        NumBatch = len(validata[imotion])
        for ibatch in range(NumBatch):
            data, label = validata[imotion][ibatch]
            nframes = data['point3DInit'].shape[1]
            njoints = int(data['point3DInit'].shape[0]/3)

            AHF = label['A_GT'].to(device)

            Loss3D_mean, LossRec, _, LossAHF, X, A, _ = VariantsPN(net, paramtrain, device, label['point3D'], AHF, data['point3DInit'], label['C'], label['ray'], data['rayDInd'], label['t'], data['rayDInd'],label['camID'], Knn, KnnC, SpCont, Naffinity_min, Naffinity_max, Train_Ver, Aff, Cadd,label['limbSeq'],label['label'], train = 'Valid', order = 1, Vari = 'Orig')
            RecMean, _, _ = ReconError(X.cpu().detach().numpy(), label['point3D'].detach().numpy())

            A = A.cpu().detach().numpy()
            AHF = AHF.cpu().detach().numpy()
            Loss3D_mean = Loss3D_mean.cpu().detach().numpy()
            LossRec = LossRec.cpu().detach().numpy()
            LossAHF = LossAHF.cpu().detach().numpy()

            RecMeanReco += RecMean
            Loss3D_meanReco += Loss3D_mean
            LossRecReco += LossRec
            LossAHFReco += LossAHF
            LossTotal += Loss3D_mean + paramtrain[0]*(LossRec) + paramtrain[1]*(LossAHF)

    RecMeanReco /= len(validata)
    Loss3D_meanReco /= len(validata)
    LossRecReco /= len(validata)
    LossAHFReco /= len(validata)
    LossTotal /= len(validata)

    return RecMeanReco, Loss3D_meanReco, LossRecReco, LossAHFReco, LossTotal
#====================================================================================================
#                              funciton for training
#====================================================================================================