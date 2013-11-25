===============================================================================
    Copyright (C) 2009  
    Marion Neumann [marion dot neumann at iais dot fraunhofer dot de]
    Zhao Xu [zhao dot xu at iais dot fraunhofer dot de]
    Supervisor: Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
 
    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
 
    This file is part of pyXGPR.
 
    pyXGPR is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.
 
    pyXGPR is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
 
    You should have received a copy of the GNU General Public License 
    along with this program; if not, see <http://www.gnu.org/licenses/>.
===============================================================================

pyXGPR provides an implementation of Standard and Relational Gaussian Process Regression,
as it is described in:

"M. Neumann, K. Kersting, Z. Xu, D. Schulz. 
Stacked Gaussian Process Learning. In H. Kargupta, W. Wang, editor(s), 
Proceedings of the 9th IEEE International Conference on Data Mining (ICDM-09), 
Miami, FL, USA, Dec. 6-9 2009" 


please cite as:
@inproceedings { neumann09icdm,
author = {M. Neumann and K. Kersting and Z. Xu and D. Schulz},
title = {Stacked Gaussian Process Learning},
year = {2009},
booktitle = {Proceedings of the 9th IEEE International Conference on Data Mining (ICDM{--}09)},
editor = {H. Kargupta, W. Wang},
month = {Dec. 6{--}9},
address = {Miami, FL, USA}
}


The current implementation has not been optimized yet and is work in progress. We appreciate any feedback.

It is mainly based on the matlab GP implementation by Rasmussen and Williams, 
which is Copyright (c) 2005 - 2007 by Carl Edward Rasmussen and Chris Williams. 
Further it invokes minimize.py which is implemented in python by Roland Memisevic 2008, 
following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen.


installing pyXGPR
------------------
Download the archive and extract it to any local directory.
Add the local directory to your PYTHONPATH:
	export PYTHONPATH=$PYTHONPATH:/path/to/local/directory/

requirements
--------------
Python 2.6
Scipy and Numpy: two of many open-source packages for scientific computing that use the Python programming language. 


