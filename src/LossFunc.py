import torch
import time
import numpy as np

def X_opt_CPU( r, C, Af, param = {'lambda1': 0.0015, 'lambda2': 1e10}):
    lambda1 = param['lambda1']
    lambda2 = param['lambda2']
    #reshape t and ray
    F = C.shape[1]
    P = int(r.shape[0]/3)
    # r = np.zeros((P*3,F))
    # C = np.zeros((3,F))
    # for f in range(F):
    #     C[:,f] = t[f,0][0].transpose()
    #     r[:,f] = np.reshape(ray[f,0].transpose(), [3*P, 1])[:,0]
    #laplace matrix    
    L = np.diag(Af.sum(axis=0)) - Af
    
    N_var = F*3
    X_opt = np.zeros((N_var*P,1), dtype = np.double)
    A = np.zeros((N_var*P, N_var*P), dtype = np.double)
    b = np.zeros((N_var*P,1), dtype = np.double)
    #Calculate b
    C_tmp = np.zeros((3,1), dtype = np.double)
    r_tmp = np.zeros((3,1), dtype = np.double)
    for f in range(F):
        C_tmp[:,0] = C[:,f]
        for i in range(P):
            if r[3*i,f] == 0:#np.isnan(r[3*i,f]):
                continue
            else:
                r_tmp[:,0] = r[3*i:3*(i+1), f]
                ib = 3*i*F+f

                b[ib] = -2*(r_tmp[0]**2-1)*((r_tmp[0]**2-1)*C_tmp[0]+r_tmp[0]*r_tmp[1]*C_tmp[1]+r_tmp[0]*r_tmp[2]*C_tmp[2])\
                -2*r_tmp[0]*r_tmp[1]*(r_tmp[0]*r_tmp[1]*C_tmp[0]+(r_tmp[1]**2-1)*C_tmp[1]+r_tmp[1]*r_tmp[2]*C_tmp[2])\
                -2*r_tmp[0]*r_tmp[2]*(r_tmp[0]*r_tmp[2]*C_tmp[0]+r_tmp[1]*r_tmp[2]*C_tmp[1]+(r_tmp[2]**2-1)*C_tmp[2])
                
                b[ib+F] = -2*r_tmp[0]*r_tmp[1]*((r_tmp[0]**2-1)*C_tmp[0]+r_tmp[0]*r_tmp[1]*C_tmp[1]+r_tmp[0]*r_tmp[2]*C_tmp[2])\
                -2*(r_tmp[1]**2-1)*(r_tmp[0]*r_tmp[1]*C_tmp[0]+(r_tmp[1]**2-1)*C_tmp[1]+r_tmp[1]*r_tmp[2]*C_tmp[2])\
                -2*r_tmp[1]*r_tmp[2]*(r_tmp[0]*r_tmp[2]*C_tmp[0]+r_tmp[1]*r_tmp[2]*C_tmp[1]+(r_tmp[2]**2-1)*C_tmp[2])             
     
                b[ib+2*F] = -2*r_tmp[0]*r_tmp[2]*((r_tmp[0]**2-1)*C_tmp[0]+r_tmp[0]*r_tmp[1]*C_tmp[1]+r_tmp[0]*r_tmp[2]*C_tmp[2])\
                -2*r_tmp[1]*r_tmp[2]*(r_tmp[0]*r_tmp[1]*C_tmp[0]+(r_tmp[1]**2-1)*C_tmp[1]+r_tmp[1]*r_tmp[2]*C_tmp[2])\
                -2*(r_tmp[2]**2-1)*(r_tmp[0]*r_tmp[2]*C_tmp[0]+r_tmp[1]*r_tmp[2]*C_tmp[1]+(r_tmp[2]**2-1)*C_tmp[2])

    b = lambda2*b/(F*P)
    
    #Calculate A from term 1 
    A_temp = np.matmul(L,L.transpose())/(F*P)
    for i in range(3*P):
        A[F*i:F*(i+1),F*i:F*(i+1)] = A[F*i:F*(i+1),F*i:F*(i+1)] + A_temp
        
    #Calculate A form term 3    

    W_temp_1 = np.zeros((3,1), dtype = np.double)
    W_temp_2 = np.zeros((3,1), dtype = np.double)
    W_temp_3 = np.zeros((3,1), dtype = np.double)
    for f in range(F):
        for i in range(P):
            if r[3*i,f] == 0:#np.isnan(r[3*i,f]):
                continue
            else:
                r_tmp = r[3*i:3*(i+1), f]
                ia = 3*i*F+f
                W_temp_1[:,0] = np.concatenate(([r_tmp[0]**2-1],[r_tmp[0]*r_tmp[1]],[r_tmp[0]*r_tmp[2]]), axis = 0)
                W_temp_2[:,0] = np.concatenate(([r_tmp[0]*r_tmp[1]],[r_tmp[1]**2-1],[r_tmp[1]*r_tmp[2]]),axis = 0)
                W_temp_3[:,0] = np.concatenate(([r_tmp[0]*r_tmp[2]],[r_tmp[1]*r_tmp[2]],[r_tmp[2]**2-1]),axis = 0)
                A_temp = lambda2*(W_temp_1*W_temp_1.transpose() + W_temp_2*W_temp_2.transpose() +W_temp_3*W_temp_3.transpose())/(F*P)
                A[ia:ia+2*F+1:F, ia:ia+2*F+1:F] = A[ia:ia+2*F+1:F, ia:ia+2*F+1:F] +  A_temp

    A_L = Af+Af.transpose()
    A_temp = lambda1*(np.diag(Af.sum(axis=0))-A_L)/(F*P)
    for i in range(3*P):
        A[F*i:F*(i+1),F*i:F*(i+1)] = A[F*i:F*(i+1),F*i:F*(i+1)] + A_temp

    for p in range(P):   
        X_opt[p*N_var:(p+1)*N_var] = -np.linalg.solve(2*A[p*N_var:(p+1)*N_var,p*N_var:(p+1)*N_var], b[p*N_var:(p+1)*N_var])#-np.linalg.solve(2*A[p*N_var:(p+1)*N_var,p*N_var:(p+1)*N_var], b[p*N_var:(p+1)*N_var])

    H = np.zeros((N_var*P,N_var), dtype = np.double)
    for p in range(P):
        H[p*N_var:(p+1)*N_var, 0:N_var] = A[p*N_var:(p+1)*N_var, p*N_var:(p+1)*N_var]


    #X_opt = -np.linalg.solve(2*A,b)
    X_opt = np.reshape(X_opt, [3*P,F])
    return X_opt, H, b

def X_opt(ray,C,A, param = {'lambda1': 0.0015, 'lambda2': 1e10}):
    # ============================================== speed up the calculation ================================================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lambda1 = torch.tensor(param['lambda1'])
    lambda2 = torch.tensor(param['lambda2'])
    
    nframes = A.shape[0]
    njoints = round(ray.shape[0]/3)
    
    #laplace matrix
    Lout = torch.diag(torch.sum(A,dim = 0)) - A
    L = torch.diag(torch.sum(A+A.t(),dim = 0)) - (A+A.t())
    N_var = nframes*3
    H = torch.zeros(N_var*njoints,N_var,dtype = torch.float64).to(device)
    H_tmp1 = torch.zeros(N_var,N_var,dtype = torch.float64).to(device)
    ft = torch.zeros(N_var*njoints,1,dtype = torch.float64).to(device)
    X_out = torch.zeros(N_var*njoints,1,dtype = torch.float64).to(device)
    

    #Calculate H from term1 and term2
    H_tmp = (torch.mm(Lout,Lout.t())+ lambda1*(L))/(nframes*njoints)  
    for j in range(3):
        H_tmp1[j*nframes:(j+1)*nframes, j*nframes:(j+1)*nframes] = H_tmp1[j*nframes:(j+1)*nframes, j*nframes:(j+1)*nframes] + H_tmp 
    
    C = C.t()
    for p in range(njoints):     
        r_tmp = ray[3*p:3*(p+1), :].t() 

        ft[p*3*nframes:p*3*nframes+nframes,0] = lambda2*(-2*(r_tmp[:,0]**2-1)*((r_tmp[:,0]**2-1)*C[:,0]+r_tmp[:,0]*r_tmp[:,1]*C[:,1]+r_tmp[:,0]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,0]*r_tmp[:,1]*(r_tmp[:,0]*r_tmp[:,1]*C[:,0]+(r_tmp[:,1]**2-1)*C[:,1]+r_tmp[:,1]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,0]*r_tmp[:,2]*(r_tmp[:,0]*r_tmp[:,2]*C[:,0]+r_tmp[:,1]*r_tmp[:,2]*C[:,1]+(r_tmp[:,2]**2-1)*C[:,2]))/(nframes*njoints)
        
        ft[p*3*nframes+nframes:p*3*nframes+2*nframes,0] = lambda2*(-2*r_tmp[:,0]*r_tmp[:,1]*((r_tmp[:,0]**2-1)*C[:,0]+r_tmp[:,0]*r_tmp[:,1]*C[:,1]+r_tmp[:,0]*r_tmp[:,2]*C[:,2])\
        -2*(r_tmp[:,1]**2-1)*(r_tmp[:,0]*r_tmp[:,1]*C[:,0]+(r_tmp[:,1]**2-1)*C[:,1]+r_tmp[:,1]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,1]*r_tmp[:,2]*(r_tmp[:,0]*r_tmp[:,2]*C[:,0]+r_tmp[:,1]*r_tmp[:,2]*C[:,1]+(r_tmp[:,2]**2-1)*C[:,2]))/(nframes*njoints)             

        ft[p*3*nframes+2*nframes:p*3*nframes+3*nframes,0] = lambda2*(-2*r_tmp[:,0]*r_tmp[:,2]*((r_tmp[:,0]**2-1)*C[:,0]+r_tmp[:,0]*r_tmp[:,1]*C[:,1]+r_tmp[:,0]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,1]*r_tmp[:,2]*(r_tmp[:,0]*r_tmp[:,1]*C[:,0]+(r_tmp[:,1]**2-1)*C[:,1]+r_tmp[:,1]*r_tmp[:,2]*C[:,2])\
        -2*(r_tmp[:,2]**2-1)*(r_tmp[:,0]*r_tmp[:,2]*C[:,0]+r_tmp[:,1]*r_tmp[:,2]*C[:,1]+(r_tmp[:,2]**2-1)*C[:,2]))/(nframes*njoints)

        #add value from term1 and term2
        H[p*N_var:(p+1)*N_var, 0:N_var] += H_tmp1

        W_temp_1 = torch.cat((r_tmp[:,0:1]**2-1,r_tmp[:,0:1]*r_tmp[:,1:2],r_tmp[:,0:1]*r_tmp[:,2:3]),0)
        W_temp_2 = torch.cat((r_tmp[:,0:1]*r_tmp[:,1:2],r_tmp[:,1:2]**2-1,r_tmp[:,1:2]*r_tmp[:,2:3]),0)
        W_temp_3 = torch.cat((r_tmp[:,0:1]*r_tmp[:,2:3],r_tmp[:,1:2]*r_tmp[:,2:3],r_tmp[:,2:3]**2-1),0)

        #if there are some joints are missing
        for f in range(nframes):
            if ray[3*p, f] == 0.0:
                ib = 3*p*nframes+f
                ft[ib] = 0.0
                ft[ib+nframes] = 0.0
                ft[ib+2*nframes] = 0.0
                W_temp_1[f,0] = 0.0
                W_temp_2[f+nframes,0] = 0.0
                W_temp_3[f+2*nframes,0] = 0.0


        H_tmp2 = lambda2*(torch.mm(W_temp_1,W_temp_1.t()) + torch.mm(W_temp_2,W_temp_2.t()) + torch.mm(W_temp_3,W_temp_3.t()))/(nframes*njoints)
        H[p*N_var:(p+1)*N_var, 0:N_var] += torch.diag(torch.diag(H_tmp2))
        H[p*N_var:p*N_var+2*nframes, nframes:N_var] += torch.diag(torch.diag(H_tmp2[0:2*nframes, nframes:N_var]))
        H[p*N_var+nframes:(p+1)*N_var, 0:2*nframes] += torch.diag(torch.diag(H_tmp2[nframes:N_var, 0:2*nframes]))
        H[p*N_var:p*N_var+nframes, 2*nframes:N_var] += torch.diag(torch.diag(H_tmp2[0:nframes, 2*nframes:N_var]))
        H[p*N_var+2*nframes:(p+1)*N_var, 0:nframes] += torch.diag(torch.diag(H_tmp2[2*nframes:N_var, 0:nframes]))

    #X_out = np.zeros((N_var*njoints,1), dtype = np.double)
    for p in range(njoints):   
        X_out[p*N_var:(p+1)*N_var],_ = torch.solve(ft[p*N_var:(p+1)*N_var], 2*H[p*N_var:(p+1)*N_var,:])    
        #X_out[p*N_var:(p+1)*N_var] = -np.linalg.solve(2*H[p*N_var:(p+1)*N_var,:].detach().numpy(), ft[p*N_var:(p+1)*N_var].detach().numpy())

    X_out = -X_out
    X_out = torch.reshape(X_out,(3*njoints,nframes))
    #X_out = np.reshape(X_out,(3*njoints,nframes))

    return X_out, H, ft
def X_opt_np(ray,C,A, param = {'lambda1': 0.0015, 'lambda2': 1e10}):
    # ============================================== speed up the calculation ================================================================
    lambda1 = param['lambda1']
    lambda2 = param['lambda2']
    
    nframes = A.shape[0]
    njoints = round(ray.shape[0]/3)
    
    #laplace matrix
    Lout = np.diag(np.sum(A,axis = 0)) - A
    L =  np.diag(np.sum(A+A.transpose(),axis = 0)) - (A+A.transpose())
    N_var = nframes*3
    H = np.zeros((N_var*njoints,N_var))
    H_tmp1 = np.zeros((N_var,N_var))
    ft = np.zeros((N_var*njoints,1))
    X_out = np.zeros((N_var*njoints,1))
    

    #Calculate H from term1 and term2
    H_tmp = (np.matmul(Lout,Lout.transpose())+ lambda1*(L))/(nframes*njoints)  
    for j in range(3):
        H_tmp1[j*nframes:(j+1)*nframes, j*nframes:(j+1)*nframes] = H_tmp1[j*nframes:(j+1)*nframes, j*nframes:(j+1)*nframes] + H_tmp 
    
    C = C.transpose()
    for p in range(njoints):     
        r_tmp = ray[3*p:3*(p+1), :].transpose()

        ft[p*3*nframes:p*3*nframes+nframes,0] = lambda2*(-2*(r_tmp[:,0]**2-1)*((r_tmp[:,0]**2-1)*C[:,0]+r_tmp[:,0]*r_tmp[:,1]*C[:,1]+r_tmp[:,0]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,0]*r_tmp[:,1]*(r_tmp[:,0]*r_tmp[:,1]*C[:,0]+(r_tmp[:,1]**2-1)*C[:,1]+r_tmp[:,1]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,0]*r_tmp[:,2]*(r_tmp[:,0]*r_tmp[:,2]*C[:,0]+r_tmp[:,1]*r_tmp[:,2]*C[:,1]+(r_tmp[:,2]**2-1)*C[:,2]))/(nframes*njoints)
        
        ft[p*3*nframes+nframes:p*3*nframes+2*nframes,0] = lambda2*(-2*r_tmp[:,0]*r_tmp[:,1]*((r_tmp[:,0]**2-1)*C[:,0]+r_tmp[:,0]*r_tmp[:,1]*C[:,1]+r_tmp[:,0]*r_tmp[:,2]*C[:,2])\
        -2*(r_tmp[:,1]**2-1)*(r_tmp[:,0]*r_tmp[:,1]*C[:,0]+(r_tmp[:,1]**2-1)*C[:,1]+r_tmp[:,1]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,1]*r_tmp[:,2]*(r_tmp[:,0]*r_tmp[:,2]*C[:,0]+r_tmp[:,1]*r_tmp[:,2]*C[:,1]+(r_tmp[:,2]**2-1)*C[:,2]))/(nframes*njoints)             

        ft[p*3*nframes+2*nframes:p*3*nframes+3*nframes,0] = lambda2*(-2*r_tmp[:,0]*r_tmp[:,2]*((r_tmp[:,0]**2-1)*C[:,0]+r_tmp[:,0]*r_tmp[:,1]*C[:,1]+r_tmp[:,0]*r_tmp[:,2]*C[:,2])\
        -2*r_tmp[:,1]*r_tmp[:,2]*(r_tmp[:,0]*r_tmp[:,1]*C[:,0]+(r_tmp[:,1]**2-1)*C[:,1]+r_tmp[:,1]*r_tmp[:,2]*C[:,2])\
        -2*(r_tmp[:,2]**2-1)*(r_tmp[:,0]*r_tmp[:,2]*C[:,0]+r_tmp[:,1]*r_tmp[:,2]*C[:,1]+(r_tmp[:,2]**2-1)*C[:,2]))/(nframes*njoints)

        #add value from term1 and term2
        H[p*N_var:(p+1)*N_var, 0:N_var] += H_tmp1

        W_temp_1 = np.concatenate((r_tmp[:,0:1]**2-1,r_tmp[:,0:1]*r_tmp[:,1:2],r_tmp[:,0:1]*r_tmp[:,2:3]), axis = 0)
        W_temp_2 = np.concatenate((r_tmp[:,0:1]*r_tmp[:,1:2],r_tmp[:,1:2]**2-1,r_tmp[:,1:2]*r_tmp[:,2:3]), axis = 0)
        W_temp_3 = np.concatenate((r_tmp[:,0:1]*r_tmp[:,2:3],r_tmp[:,1:2]*r_tmp[:,2:3],r_tmp[:,2:3]**2-1), axis = 0)

        #if there are some joints are missing
        for f in range(nframes):
            if ray[3*p, f] == 0.0:
                ib = 3*p*nframes+f
                ft[ib] = 0.0
                ft[ib+nframes] = 0.0
                ft[ib+2*nframes] = 0.0
                W_temp_1[f,0] = 0.0
                W_temp_2[f+nframes,0] = 0.0
                W_temp_3[f+2*nframes,0] = 0.0


        H_tmp2 = lambda2*(np.matmul(W_temp_1,W_temp_1.transpose()) + np.matmul(W_temp_2,W_temp_2.transpose()) + np.matmul(W_temp_3,W_temp_3.transpose()))/(nframes*njoints)
        H[p*N_var:(p+1)*N_var, 0:N_var] += np.diag(np.diag(H_tmp2))
        H[p*N_var:p*N_var+2*nframes, nframes:N_var] += np.diag(np.diag(H_tmp2[0:2*nframes, nframes:N_var]))
        H[p*N_var+nframes:(p+1)*N_var, 0:2*nframes] += np.diag(np.diag(H_tmp2[nframes:N_var, 0:2*nframes]))
        H[p*N_var:p*N_var+nframes, 2*nframes:N_var] += np.diag(np.diag(H_tmp2[0:nframes, 2*nframes:N_var]))
        H[p*N_var+2*nframes:(p+1)*N_var, 0:nframes] += np.diag(np.diag(H_tmp2[2*nframes:N_var, 0:nframes]))

    #X_out = np.zeros((N_var*njoints,1), dtype = np.double)
    for p in range(njoints):   
        #X_out[p*N_var:(p+1)*N_var],_ = torch.solve(ft[p*N_var:(p+1)*N_var], 2*H[p*N_var:(p+1)*N_var,:])    
        X_out[p*N_var:(p+1)*N_var] = np.linalg.solve(2*H[p*N_var:(p+1)*N_var,:], ft[p*N_var:(p+1)*N_var])

    X_out = -X_out
    X_out = np.reshape(X_out,(3*njoints,nframes))
    #X_out = np.reshape(X_out,(3*njoints,nframes))

    return X_out

def RecLoss_CPU(ray, C, point3D, A, param = {'lambda1': 0.0015, 'lambda2': 1e10}):

    X1, H1, ft1 = X_opt_CPU(ray, C, A, param)
    X2, H2, ft2 = X_opt(torch.from_numpy(ray), torch.from_numpy(C), torch.from_numpy(A), param)
    nframes = X1.shape[1]
    njoints = round(X1.shape[0]/3)

    #Loss = torch.norm(torch.reshape(X.t(),(-1,3))-torch.reshape(GTdata['point3D'].t(),(-1,3)), dim = 1)
    Loss = np.linalg.norm(X-point3D+1e-8)/(nframes*njoints*3)
    return Loss, X

def RecLoss(ray, C, point3D, A, param = {'lambda1': 0.0015, 'lambda2': 1e10}):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X,_,_ = X_opt(ray, C, A.t(), param)
    nframes = X.shape[1]
    njoints = round(X.shape[0]/3)

    #Loss = torch.sum(torch.norm(torch.reshape(X.t(),(-1,3))-torch.reshape(point3D.t(),(-1,3)), dim = 1))/(nframes*njoints)
    Loss = torch.norm(X-point3D)/(nframes*njoints*3)
    return Loss, X

def ReconError(X_GT, X):
    nframes = X.shape[1]
    njoints = round(X.shape[0]/3)
    # Error = np.zeros((njoints, nframes))
    # for f in range(nframes):
    #     for p in range(njoints):
    #         Error[p,f] = np.linalg.norm(X[p*3:(p+1)*3,f] - X_GT[p*3:(p+1)*3,f])
    Error = np.linalg.norm(np.reshape(X.transpose(),(-1,3))-np.reshape(X_GT.transpose(),(-1,3)), axis = 1)
    return np.mean(Error), np.std(Error), Error


def ReprojLoss(GTdata, A, param = {'lambda1': 0.0015, 'lambda2': 1e10}):
    X = X_opt(GTdata['ray'],GTdata['C'], A)
    nframes = GTdata['point3D'].shape[1]
    njoints = round(GTdata['point3D'].shape[0]/3)
    point2d = torch.zeros(njoints*2,nframes, dtype = torch.float64)
    for f in range(nframes):
        point2dtmp = torch.mm(torch.mm(GTdata['K'][:,GTdata['camID'][f]*3:(GTdata['camID'][f]+1)*3],torch.cat((GTdata['R'][:,f*3:(f+1)*3], -torch.mm(GTdata['R'][:,f*3:(f+1)*3],GTdata['C'][:,f:f+1])),1)), torch.cat((torch.reshape(X[:,f],(-1,3)).t(), torch.ones(1,njoints,dtype = torch.float64)), 0))
        point2dtmp = point2dtmp/point2dtmp[2,:]
        point2d[:,f:f+1] = torch.reshape(point2dtmp[0:2,:].t(),(-1,1))

    Loss = torch.norm(point2d-GTdata['point2D']+1e-8)/(nframes*njoints)

    return Loss, X

def LapLoss(latent, timeindex):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    latent2 = latent[:,:,timeindex]
    indexNear = np.where(abs(timeindex - timeindex[:,0:1]) <= 5)
    indexFar = np.where(abs(timeindex - timeindex[:,0:1]) > 5)
    pwdist = torch.norm((latent-latent2.permute(3,0,1,2)+1e-8).permute(1,2,3,0), dim = [0, 1])
    lossNear = torch.sum(pwdist[indexNear[0], indexNear[1]])/(latent.shape[0]*len(indexNear[0]))#
    lossFar = torch.sum(pwdist[indexFar[0], indexFar[1]])/(latent.shape[0]*len(indexFar[0]))#
    loss = torch.max(lossNear - lossFar+5, torch.tensor(0.0,dtype = torch.float64).to(device))
    return loss.to(device), lossNear, lossFar

    # loss = torch.tensor(0.0,dtype = torch.float64).to(device)

    # indexFar = np.where(abs(timeindex - timeindex[0]) > 5)
    # indexFar = indexFar[0]
    # latentFar = latent[:,:,indexFar]

    # loss -= torch.norm(latentFar - latent[:,:,0:1])/(latentFar.shape[0]*latentFar.shape[2])

    # indexNear = np.where(abs(timeindex - timeindex[0]) <= 5)
    # indexNear = indexNear[0][np.argsort(timeindex[indexNear[0]])]
    # latentNear = latent[:,:,indexNear]

    # loss += (0.1*torch.norm(latentNear[:,:,0:(latentNear.shape[2]-1)]-latentNear[:,:,1:latentNear.shape[2]]) + torch.norm((latentNear[:,:,0:(latentNear.shape[2]-2)]+latentNear[:,:,2:latentNear.shape[2]])/2-latentNear[:,:,1:(latentNear.shape[2]-1)]))/(latentNear.shape[0]*latentNear.shape[2])
    

    # for f in range(nframes-1):
    #     if np.abs(timeindex[f] - timeindex[0]) > 5:
    #         loss -= torch.norm(latent[:,:,f] - latent[:,:,0])/(nframes*njoints)
    #     else:
    #         loss += torch.norm(latent[:,:,f] - latent[:,:,f+1])/(nframes*njoints)

    
    # order = np.argsort(timeindex)
    # latent = latent[:,:,order]

    # L = torch.zeros(nframes-2,nframes,dtype = torch.float64).to(device)
    # for f in range(nframes-2):
    #     L[f,f:f+3] = torch.tensor([-1,2,-1]).to(device)
    # for p in range(njoints):
    #     loss += torch.norm(torch.mm(L,latent[p,:,:].t()))/(nframes*njoints)
   
    # return loss

def LapLossV2(latent, timeindex):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #njoints, ndim, nframes = latent.shape
    loss = torch.tensor(0.0,dtype = torch.float64).to(device)


    indexFar = np.where(abs(timeindex - timeindex[0]) > 5)
    indexFar = indexFar[0]
    if indexFar.shape[0] >0:
        latentFar = latent[:,:,indexFar]
        loss += torch.sum(latentFar*latent[:,:,0:1])/(latentFar.shape[0]*latentFar.shape[2])#

    indexNear = np.where(abs(timeindex - timeindex[0]) <= 5)
    indexNear = indexNear[0][np.argsort(timeindex[indexNear[0]])]
    if indexNear.shape[0] >0:
        latentNear = latent[:,:,indexNear]
        loss += (-0.001*torch.sum(latentNear[:,:,0:(latentNear.shape[2]-1)]*latentNear[:,:,1:latentNear.shape[2]]))#/(latentNear.shape[0])#*latentNear.shape[2]#+ 0.001*torch.norm((latentNear[:,:,0:(latentNear.shape[2]-2)]+latentNear[:,:,2:latentNear.shape[2]])/2-latentNear[:,:,1:(latentNear.shape[2]-1)])

    return loss

def LapLossV2_2(latent, timeindex):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    latent2 = latent[:,:,timeindex]
    indexNear = np.where(abs(timeindex - timeindex[:,0:1]) <= 5)
    indexFar = np.where(abs(timeindex - timeindex[:,0:1]) > 5)
    pwdist = torch.sum((latent*latent2.permute(3,0,1,2)).permute(1,2,3,0), [0,1])
    lossNear = torch.sum(pwdist[indexNear[0], indexNear[1]])/(latent.shape[0]*len(indexNear[0]))#
    lossFar = torch.sum(pwdist[indexFar[0], indexFar[1]])/(latent.shape[0]*len(indexFar[0]))#
    loss = lossFar - lossNear
    return loss.to(device), lossNear, lossFar

def LapLossV3(latent, timeindex, rayDInd):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    latent = latent[:,:,timeindex[np.argsort(timeindex)]]
    loss = (-0.1*torch.sum(latent[:,:,0:(latent.shape[2]-1)]*latent[:,:,1:latent.shape[2]]) + torch.norm((latent[:,:,0:(latent.shape[2]-2)]+latent[:,:,2:latent.shape[2]])/2-latent[:,:,1:(latent.shape[2]-1)]+1e-8))/(latent.shape[0]*latent.shape[2])
    for f in range(latent.shape[2]):
        indexFar = np.where(abs(rayDInd[f,0:10] - timeindex[f]) > 5)
        indexFar = indexFar[0]
        if indexFar.shape[0] >0:
            latentFar = latent[:,:,rayDInd[f,indexFar]]
            loss += torch.sum(latentFar*latent[:,:,f:f+1])/(latentFar.shape[0]*latentFar.shape[2])/latent.shape[3]

    return loss