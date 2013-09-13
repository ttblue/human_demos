
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
Created on 04.03.2010

Demo_XGP: Relational GP (XGP) to demonstrate prediction (and training) for generated noisy data (plus relations).

@author: marion neumann
'''


from GPR import gpr, kernels
from Tools import general, validation
from numpy import *
from matplotlib import pyplot

# DATA:
# necessary data for   XGPR X, y, Xstar, id_lab, id_unl, G ([weighted] adjacency matrix of relational graph)

## GENERATE data from a noisy XGP
l = 20      # number of labeled/training data 
u = 201     # number of unlabeled/test data

### GENERATE ATTRIBUTES (here: one-dimensional)
X = array(15*(random.random((l,1))-0.5))
X = msort(X)
### GENERATE id vector
id_lab = array([arange(1,l+1)]).transpose()

### GENERATE RELATIONS ([weighted] adjacency matrix of relational graph)
R = random.binomial(1,0.1,[l+u,l+u])    # generate sparse adjacency matrix    
R = R + R.transpose()                   # force symmetry

### CREATE covariance matrix based on relations 
### via regularized Laplacian graph kernel [NOTE: other graph kernels can be used HERE]
s2   = 1000**2
beta = 1
K_R = kernels.regLapKernel(R, beta, s2) 

'''      A   B
    K = 
         C   D 
    where A = train by train (l x l)
          B = train by test  (l x u)
          C = test by train  (u x l)
          D = test by test   (u x u). ''' 
          
A = K_R[0:l, 0:l]                       # cov matrix for labeled (i.e. training) data (needed to tune weight w)
B = K_R[0:l, l:l+u]                     # cov matrix between labeled and unlabeled data (needed to do xgp prediction)
d = diag(K_R[l:l+u,l:l+u]).transpose()  # self convariances for unlabeled data (needed in kernels.covMatrix)
B = vstack((B, d))

## SET COVARIANCE FUNCTION
covfunc = ['kernels.covSumMat', ['kernels.covSEiso','kernels.covNoise','kernels.covMatrix']]  # covMatrix -> no hyperparameters to optimize!!
## SET (hyper)parameters, e.g.:
#logtheta = array([log(0.3), log(1.08), log(5e-5)])
#logtheta = array([log(3), log(1.16), log(0.89)])
logtheta = array([log(1), log(1), log(sqrt(0.01))])


w_used = round(random.random(),1)
print 'generated mixture weight: ', w_used

### GENERATE sample observations from the XGP 
y = dot(linalg.cholesky(general.feval(covfunc, logtheta, X, A, w_used)).transpose(),random.standard_normal((l,1)))
### TEST POINTS
Xstar = array([linspace(-7.5,7.5,u)]).transpose()   # u test points evenly distributed in the interval [-7.5, 7.5]



#_________________________________
## Relational GP (XGP)

# ***UNCOMMENT THE FOLLOWING LINES TO DO TRAINING OF HYPERPARAMETERS AND MIXTURE WEIGHT***
### TRAINING XGP (learn hyperparameters of GP and tune mixture weight)
## INITIALIZE (hyper)parameters by -1
#d = X.shape[1]
#init = -1*ones((d,1))
#loghyper = array([[-1], [-1]])
#loghyper = vstack((init, loghyper))[:,0]
#print 'initial hyperparameters: ', exp(loghyper)
#
#w = array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])  # set vector for grid search
#M = zeros((size(w),3+size(loghyper)))  # matrix to collect[r_sq_B, RMSE_B, MAE_B] for all weights w(i)
#print 'XGP: ...training (PT)'
#print 'XGP: ...tune weight parameter w (via CV)'
#for i in range(0,size(w)):
#    M[i,:] = validation.k_fold_cv(3, loghyper, covfunc, X, id_lab, y.copy(), A, w[i])      
#        
## SELECT weight and corresponding hyperparameters according to highest R_sq value
#loc = M[:,0].argmax(0)  
#w_used = w[loc]
#print 'selected weight: '+str(w_used)
#logtheta = array(M[loc,3:])



# XGP PREDICTION  
print 'XGP: ...prediction (PT)'
[MU, S2] = gpr.gp_pred(logtheta, covfunc, X, y, Xstar, A, w_used, B)

#print MU

## plot results
pyplot.suptitle('logtheta:', fontsize=12)
pyplot.title(logtheta)

pyplot.plot(Xstar,MU, 'g^', X,y, 'ro')
pyplot.show()
