#!/usr/bin/env python
# coding: utf-8

# In[315]:


get_ipython().run_line_magic('matplotlib', 'notebook')
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')

import numpy as np
import math as m

from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

import matplotlib.pyplot as plt
from matplotlib import ticker


# ## Step 1: Function evaluations 
# Evaluation of the objective function and its gradient given a specific point $p$, number of segments $n$, segment lenghts $l$ and a set of configuration angles $v$. 

# In[381]:


def evalu(n,l,p0,v):
    
    cums = np.cumsum(v)
    f = np.zeros(2)
    g = np.zeros(2)
    ggh = np.zeros((n,2))
    grad_g = np.zeros((n,2))
    
    for i in range(n):
        
        c = m.cos(cums[i])
        s = m.sin(cums[i])
        
        f[0] = (l[i]*c)+f[0]
        f[1] = (l[i]*s)+f[1]
        
    
    for i in range(n):
        
            if i == 0:
                
                for j in range(n):
                    s = m.sin(cums[j])
                    c = m.cos(cums[j])
            
                    ggh[i,0] = l[j]*(-s)+ggh[i,0]
                    ggh[i,1] = l[j]*(c)+ggh[i,1]
                
            else:
                
                for j in range(i,n):
                        
                    s = m.sin(cums[j])
                    c = m.cos(cums[j])
            
                    ggh[i,0] = l[j]*(-s)+ggh[i,0]
                    ggh[i,1] = l[j]*(c)+ggh[i,1]
               
            
    g[0] = 0.5*(f[0]-p0[0])**2
    g[1] = 0.5*(f[1]-p0[1])**2
    g = g[0]+g[1]
    
    grad_g = ggh @ (f-p0)
    
    
    return g, grad_g



        
        
        



# ## Step 2:Line search
# The first thing we will do is to find the line search using "Backtracking Armijo".

# In[418]:


def BacktrackLineSearch(vk, gk, ggk, pk, ak, c, rho, nmaxls=100):
    
    pkggk = pk @ ggk
   
    g,ggk = evalu(n,l,p0,vk)
    
    for i in range (nmaxls):
        if g <= gk + c*rho*ak*pkggk:
            break
        ak *= rho
        vk = vk + np.multiply(ak, pk)
        g,ggk = evalu(n,l,p0,vk)
        
    return ak


# Then, we need to compute the search direction. For Gradient descent, this is simply $p_{k+1}=-\nabla f(x_{k+1})$, and we have already computed this during the first step.

# ## Step 3:Optimization
# Now we need to compute the search direction, perform line search along this direction and then perform the step.

# In[419]:


def optimize(n, l, p0, v0, c, rho, tol,nmax):
    
    vk = v0
    listofAng = np.zeros((nmax,3))
    listofAng[0] = v0;
    gk,ggk = evalu(n,l,p0,vk)
    pk = None
    ak = 1
    
    
    for k in range(1,nmax):
        
        ak = 1
        pk = -ggk   #The search direction for steepest descent is simply the gradient
        ak = BacktrackLineSearch(vk, gk, ggk, pk, ak, c, rho)
        listofAng[k] = vk + np.multiply(ak, pk)
        vk = listofAng[k]
        gk,ggk = evalu(n,l,p0,vk) 
        
        if np.linalg.norm(ggk) < tol:
            break
    
    return listofAng
    


# In[420]:


def ROBOT_PLT(n, l, p0, v):
    
    f = np.zeros(2)
    temp0 = np.zeros(n)
    temp = np.zeros(2)    
    
    c = m.cos(v[0])
    s = m.sin(v[0])
    f[0] = (l[0]*c)
    f[1] = (l[0]*s)
    temp0[0] = v[0]
    x_values = (0,f[0])
    y_values = (0,f[1])
    plt.plot(x_values, y_values)
    plt.scatter(f[0],f[1],s=300)


    for i in range(1,n):

            temp[0]=f[0]
            temp[1]=f[1]
            c = m.cos(v[i]+np.sum(temp0))
            s = m.sin(v[i]+np.sum(temp0))

            f[0] = (l[i]*c)+f[0]
            f[1] = (l[i]*s)+f[1]
            temp0[i] = v[i]
         
            x_values = (temp[0],f[0])
            y_values = (temp[1],f[1])
            plt.plot(x_values, y_values)
            plt.scatter(f[0],f[1],s=300)

    plt.scatter(0,0)        
    plt.scatter(3,2)
    plt.show()

    


# In[421]:


def OPT_PLT(n, l, p0, v):
    
    f = np.zeros(2)
    temp0 = np.zeros(n)
    temp = np.zeros(2)    
    
    c = m.cos(v[0])
    s = m.sin(v[0])
    f[0] = (l[0]*c)
    f[1] = (l[0]*s)
    temp0[0] = v[0]


    for i in range(1,n):

            temp[0]=f[0]
            temp[1]=f[1]
            c = m.cos(v[i]+np.sum(temp0))
            s = m.sin(v[i]+np.sum(temp0))

            f[0] = (l[i]*c)+f[0]
            f[1] = (l[i]*s)+f[1]
            temp0[i] = v[i]
            
            
    plt.scatter(f[0],f[1],marker='x',s=50)        
    plt.scatter(3,2,color='black')
    plt.show()

    


# In[422]:


#Input arguments
n = 3  #Number of segments
l = np.array([3, 2, 2])
p0 = np.array([3, 2])



#Initial conditions
#v0 = np.array([1.72, -1.54, -0.9])
v0 = np.array([0,1,1])




#Algorithm parameters
c = 0.01
rho = 0.5
tol = 1e-8
nmax = 1000

vgd = optimize(n, l, p0, v0, c, rho, tol,nmax)


topi = 2*m.pi
for i in range(nmax):
    for j in range(n):
        if vgd[i,j]>topi:
            fac = int(vgd[i,j]/(topi))
            vgd[i,j] = vgd[i,j]-(topi*fac)
        elif vgd[i,j]<0:
            fac = int(vgd[i,j]/(topi))
            vgd[i,j] = vgd[i,j]-(topi*fac)
            vgd[i,j] = -vgd[i,j]
            
###############################################################################################################################""



#print(vgd)


################################################################################################################################
#          PLOT
################################################################################################################################


# for i in range(nmax):
#     OPT_PLT(n, l, p0, vgd[i])



# SOLUTION = np.array([1.72, -1.54, -0.9])
# plt.figure()
# ROBOT_PLT(n, l, p0, SOLUTION)

# print(vgd[nmax-1])

# SOLUTION = vgd[nmax-1]
# plt.figure()
# ROBOT_PLT(n, l, p0, SOLUTION)







# In[ ]:





# In[ ]:




