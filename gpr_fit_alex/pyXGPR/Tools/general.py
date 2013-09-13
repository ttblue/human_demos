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
Created on 08.09.2009

@author: Marion Neumann (last update 08/01/10)
'''
from numpy import *
from GPR import kernels

def feval(funcName, *args):
    if isinstance(funcName, list):
        return eval(funcName[0])(funcName[1],*args)
    else:
        return eval(funcName)(*args)
    
    
def get_index_vec(a,b):
    ''' returns vector of indices of b for the entries in a.
    
    example:
    
         1        2                2
    a =  2    b = 3    index_vec = 0
         3        1                1
    
    returns '-1' if entry in a is not an entry in b. '''
    index_vec = -1*ones(a.size)
    for i in range(a.size):
        for j in range(b.size):
            if a[i]==b[j]:
                index_vec[i]=j
                break    
    return index_vec


