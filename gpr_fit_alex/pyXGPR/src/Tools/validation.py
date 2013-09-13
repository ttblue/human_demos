#===============================================================================
#    Copyright (C) 2009  
#    Marion Neumann [marion dot neumann at iais dot fraunhofer dot de]
#    Zhao Xu [zhao dot xu at iais dot fraunhofer dot de]
#    Supervisor: Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyXGPR.
# 
#    pyXGPR is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyXGPR is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License 
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================
'''
Created on 09.09.2009

@author: Marion Neumann (last update 08/01/10)
'''

from numpy import *
import general
from GPR import gpr

def k_fold_cv(k, loghyper, covfunc, X, id, y, R, w, trafo=None):
    '''Do k-fold cross validation for 
       tuning weight parameter of SUM of 
       parameterized cov **function** AND cov **martix**. '''
    
    ## TRAINING the parameters
    logtheta = gpr.gp_train(loghyper, covfunc, X, y, R, w)
    #print exp(logtheta)
        
    MU = zeros((y.shape))
    S2 = zeros((y.shape))
    
    n = id.shape[0]
    s = floor(n/k)
    for j in range(0,k):
        # prepare X, y and R according to fold
        if j==0:    # first fold
            X_tr     = X[s:n,:]
            y_tr     = y[s:n,:]
            id_tr    = id[s:n,:]              
            X_val    = X[0:s,:]
            id_val   = id[0:s,:]
        elif 0<j<k-1:
            X_tr     = X[0:j*s,:]
            X_tr     = vstack((X_tr,X[j*s+s:n,:]))
            y_tr     = y[0:j*s,:]
            y_tr     = vstack((y_tr,y[j*s+s:n,:]))
            id_tr    = id[0:j*s,:]
            id_tr    = vstack((id_tr,id[j*s+s:n,:])) 
            X_val    = X[j*s:j*s+s,:]
            id_val   = id[j*s:j*s+s,:]         
        else:   # last fold
            s1 = s+fmod(n,k)
            X_tr     = X[0:n-s1,:]
            y_tr     = y[0:n-s1,:]
            id_tr    = id[0:n-s1,:]   
            X_val    = X[n-s1:n,:]
            id_val   = id[n-s1:n,:]
        

        loc_tr = general.get_index_vec(id_tr[:,0], id[:,0])
        R_tr = R.take(loc_tr.astype(int),axis=0).take(loc_tr.astype(int),axis=1)
        
        loc_val = general.get_index_vec(id_val[:,0], id[:,0])
        R_tr_val = R.take(loc_tr.astype(int),axis=0).take(loc_val.astype(int),axis=1)

        ## GET self covariances for the test cases (needed in covMatrix)
        R_val = R.take(loc_val.astype(int),axis=0).take(loc_val.astype(int),axis=1)
        d = diag(R_val).transpose()   
        R_tr_val = vstack((R_tr_val, d))
        
        ## PREDICTION 
        [MU_temp, S2_temp] = gpr.gp_pred(logtheta, covfunc, X_tr, y_tr, X_val, R_tr, w, R_tr_val)
        
        if j==0:  # first fold
            MU[0:s,:] = MU_temp
            S2[0:s,:] = S2_temp
        elif 0<j<k-1:
            MU[j*s:j*s+s,:] = MU_temp
            S2[j*s:j*s+s,:] = S2_temp      
        else:     # last fold
            MU[n-s1:n,:] = MU_temp
            S2[n-s1:n,:] = S2_temp 
            
    ## VALIDATION for predictions on validation dataset
    ## DO NOT resacle predictions - rescaling is done in valid_fig if needed!!!
    M = valid_fig(y, MU, trafo=trafo)
    print w
    print M
    return hstack((M, logtheta))


def valid(loghyper, covfunc, X, id, y, X_val, y_val, R_tr, w, R_tr_val, R_val, trafo=None):
    '''Do validation on validation dataset 
       tuning weight parameter of SUM of 
       parameterized cov **function** AND cov **martix**. '''
    
    ## TRAINING the parameters
    logtheta = gpr.gp_train(loghyper, covfunc, X, y, R_tr, w)

    ## GET self covariances for the test cases (needed in covMatrix)
    d = diag(R_val).transpose()   
    R_tr_val = vstack((R_tr_val, d))
    
    ## PREDICTION 
    [MU, S2] = gpr.gp_pred(logtheta, covfunc, X, y, X_val, R_tr, w, R_tr_val)
        
            
    ## VALIDATION for predictions on validation dataset
    ## DO NOT resacle predictions - rescaling is done in valid_fig if needed!!!
    M = valid_fig(y_val, MU, trafo=trafo)
    print w
    print M
    return hstack((M, logtheta))


def k_fold_cv_woLogthetaOpt(k, logtheta, covfunc, X, id, y, R, w, trafo=None):
    '''Do k-fold cross validation for 
       tuning weight parameter of SUM of 
       parameterized cov **function** AND cov **martix**. '''
    MU = zeros((y.shape))
    S2 = zeros((y.shape))
    
    n = id.shape[0]
    s = floor(n/k)
    for j in range(0,k):
        # prepare X, y and R according to fold
        if j==0:    # first fold
            X_tr     = X[s:n,:]
            y_tr     = y[s:n,:]
            id_tr    = id[s:n,:]              
            X_val    = X[0:s,:]
            id_val   = id[0:s,:]
        elif 0<j<k-1:
            X_tr     = X[0:j*s,:]
            X_tr     = vstack((X_tr,X[j*s+s:n,:]))
            y_tr     = y[0:j*s,:]
            y_tr     = vstack((y_tr,y[j*s+s:n,:]))
            id_tr    = id[0:j*s,:]
            id_tr    = vstack((id_tr,id[j*s+s:n,:])) 
            X_val    = X[j*s:j*s+s,:]
            id_val   = id[j*s:j*s+s,:]         
        else:   # last fold
            s1 = s+fmod(n,k)
            X_tr     = X[0:n-s1,:]
            y_tr     = y[0:n-s1,:]
            id_tr    = id[0:n-s1,:]   
            X_val    = X[n-s1:n,:]
            id_val   = id[n-s1:n,:]
        
        loc_tr = general.get_index_vec(id_tr[:,0], id[:,0])
        R_tr = R.take(loc_tr.astype(int),axis=0).take(loc_tr.astype(int),axis=1)
        
        loc_val = general.get_index_vec(id_val[:,0], id[:,0])
        R_tr_val = R.take(loc_tr.astype(int),axis=0).take(loc_val.astype(int),axis=1)

        ## GET self covariances for the test cases (needed in covMatrix)
        R_val = R.take(loc_val.astype(int),axis=0).take(loc_val.astype(int),axis=1)
        d = diag(R_val).transpose()   
        R_tr_val = vstack((R_tr_val, d))
        
        ## PREDICTION 
        [MU_temp, S2_temp] = gpr.gp_pred(logtheta, covfunc, X_tr, y_tr, X_val, R_tr, w, R_tr_val)
        
        if j==0:  # first fold
            MU[0:s,:] = MU_temp
            S2[0:s,:] = S2_temp
        elif 0<j<k-1:
            MU[j*s:j*s+s,:] = MU_temp
            S2[j*s:j*s+s,:] = S2_temp      
        else:     # last fold
            MU[n-s1:n,:] = MU_temp
            S2[n-s1:n,:] = S2_temp 
            
    ## VALIDATION for predictions on validation dataset
    M = valid_fig(y, MU, trafo=trafo)
    print M
    return hstack((M, logtheta))



def valid_fig(y, MU, S2=None, trafo=None):
    '''calculate evaluation measures
                 R^2
                 MAE
                 RMSE
        INPUT: trafo_O = [ymax, ymin, offset, log_tf_O]. '''   
    if trafo!=None:
        ## RESCALE BACK predictions  
        MU = MU + trafo[2]
        MU = MU*(trafo[0]-trafo[1])+ trafo[1]
        if trafo[3]:
            MU = exp(MU)  # -> median (?? median = mean!) 

        ## RESCALE BACK y_val
        y = y + trafo[2]
        y = y*(trafo[0]-trafo[1])+ trafo[1]
        if trafo[3]:
            y = exp(y)
    
    n = MU.shape[0]
    
    ## R-squared
    corr = sum((y-mean(y))*(MU-mean(MU)))/((n-1)*std(y)*std(MU)) # element-wise mult
    R_sq = corr**2
    
    ## RMSE
    sse = (y-MU)**2     # element-wise power
    sse = sum(sse)
    RMSE = sqrt(sse/n)
    
    ## MAE
    MAE = sum(abs(y-MU))/n
    
    return [R_sq, MAE, RMSE]


def valid_fig_NLPD(y, MU, S2):
    '''Calculate evaluation measure NLPD in transformed observation space.
       
       INPUT   y     observed targets
               MU    vector of predictions/predicted means
               S2    vector of 'self' variances
               
       OUTPUT  nlpd  Negative Log Predictive Density.'''
       
    n = MU.shape[0]
    nlpd = 0.5*log(2*math.pi*S2) + 0.5*((y-MU)**2)/S2 
    print nlpd.shape
    nlpd = sum(nlpd)/n
    return nlpd
    
    