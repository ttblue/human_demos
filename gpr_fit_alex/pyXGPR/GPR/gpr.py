
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
Created on 31/08/2009

    This implementation follows the matlab GP implementation by Rasmussen, 
    which is Copyright (c) 2005 - 2007 by Carl Edward Rasmussen and Chris Williams. 
    It invokes minimize.py which is implemented in python by Roland Memisevic 2008, 
    following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen.

    This module provides functions for 
    - training the hyperparameters of a Standard GP (parameterised covariance function)                                 
    - prediction in the Standard GP framework
    
    (note: these also work for the XGP framework but mixture weights have to be adjusted separately!)
    
    INPUT:
    loghyper       column vector of log hyperparameters (of the covariance function)  
    covfunc        is the covariance function (string)
    X              is the n by D matrix of training inputs
    y              is the column vector of targets (size n)
                   NOTE: y must have zero mean!!
    Xstar          is the nn by D matrix of test inputs
    
    R, w, Rstar    are inputs for XGP regression
    
    OUTPUT:
    logtheta       are the learnt hyperparameters for the given cov func
    out1           is the vector of predicted means
    out2           is the vector of predicted variances
    

@author: Marion Neumann (last update 08/01/10)
'''

from numpy import linalg, array, dot, zeros, size, log, diag, pi, eye
#import scipy.linalg
from Tools.general import feval
import minimize


def gp_train(loghyper, covfunc, X, y, R=None, w=None):
    ''' gp_train() returns the learnt hyperparameters.
    Following chapter 5.4.1 in Rasmussen and Williams: GPs for ML (2006).
    The original version (MATLAB implementation) of used optimizer minimize.m 
    is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
    The used python adaptation is by Roland Memisevic 2008.
    
    Input R and w is needed for XGP regression! '''
    if R==None: 
        print '        gp_train()'
    else:
        print '        xgp_train()'

    [logtheta, fvals, iter] = minimize.run(loghyper, nlml, dnlml, [covfunc, X, y, R, w], maxnumfuneval=100)
    
    return logtheta

    
def gp_pred(logtheta, covfunc, X, y, Xstar, R=None, w=None, Rstar=None):
    if R==None: 
        print '        gp_pred()'
    #else:
    #    print '        xgp_pred()'
        
    # compute training set covariance matrix (K) and
    # (marginal) test predictions (Kss = self-cov; Kstar = corss-cov)
    if R==None:
        K = feval(covfunc, logtheta, X)                     # training covariances
        [Kss, Kstar] = feval(covfunc, logtheta, X, Xstar)   # test covariances (Kss = self covariances, Kstar = cov between train and test cases)
    else:
        K = feval(covfunc, logtheta, X, R, w)               # training covariances
        [Kss, Kstar] = feval(covfunc, logtheta, X, R, w, Xstar, Rstar)   # test covariances
     
    L = linalg.cholesky(K)                      # cholesky factorization of cov (lower triangular matrix)
    alpha = solve_chol(L.transpose(),y)         # compute inv(K)*y
    out1 = dot(Kstar.transpose(),alpha)         # predicted means
    v = linalg.solve(L, Kstar)                  
    tmp=v*v                                     
    out2 = Kss - array([tmp.sum(axis=0)]).transpose()  # predicted variances  

    return [out1, out2]


def solve_chol(A,B):
    return linalg.solve(A,linalg.solve(A.transpose(),B))


def nlml(loghyper, covfunc, X, y, R=None, w=None):
 
    n = X.shape[0]
    # compute training set covariance matrix
    if R==None:
        K = feval(covfunc, loghyper, X)
    else:
        K = feval(covfunc, loghyper, X, R, w)     
        
    # cholesky factorization of the covariance
    L = linalg.cholesky(K)      # lower triangular matrix
    # compute inv(K)*y
    alpha = solve_chol(L.transpose(),y)
    # compute the negative log marginal likelihood
    return (0.5*dot(y.transpose(),alpha) + (log(diag(L))).sum(axis=0) + 0.5*n*log(2*pi))[0][0] 


def dnlml(loghyper, covfunc, X, y, R=None, w=None):
    out = zeros((loghyper.shape))
    W = get_W(loghyper, covfunc, X, y, R, w)

    if R==None:
        for i in range(0,size(out)):
            out[i] = (W*feval(covfunc, loghyper, X, i)).sum()/2
    else:
        for i in range(0,size(out)):
            out[i] = (W*feval(covfunc, loghyper, X, R, w, i)).sum()/2
 
    return out    
 
        
def get_W(loghyper, covfunc, X, y, R=None, w=None):
    '''Precompute W for convenience.'''
    n = X.shape[0]
    # compute training set covariance matrix
    if R==None:
        K = feval(covfunc, loghyper, X)
    else:
        K = feval(covfunc, loghyper, X, R, w)

    # cholesky factorization of the covariance
    L = linalg.cholesky(K)      # lower triangular matrix
    alpha = solve_chol(L.transpose(),y)
    W = linalg.solve(L.transpose(),linalg.solve(L,eye(n)))-dot(alpha,alpha.transpose())
    return W

