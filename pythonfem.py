#Python Main FEM Code for the Heat Conduction Problem

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy.linalg import inv
from scipy import interpolate
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from datetime import datetime

startTime = datetime.now()


#####################################################################################################################Define the Meshing Function
def cmesh(n):  #Number of Nodes in Each Direction For Meshing Function

    n_nodes = int(n**3)
    n_elem = int((n-1)**3)
    
    #This code generates the connectivity table for the mesh

    conn = np.zeros(((n-1)**3, 8), dtype = np.int)

    #First row of connectivity table
    conn[0,0] = 1
    conn[0,1] = 2
    conn[0,2] = n + 2
    conn[0,3] = n + 1
    conn[0,4] = n**2 + 1
    conn[0,5] = n**2 + 2
    conn[0,6] = n**2 + n + 2
    conn[0,7] = n**2 + n + 1


    counter  = 0 #counter for row of connectivity table
    for ii in range(0, n-1): #outer jumps in z + n**2
        if ii < n-1:
            conn[counter] = conn[counter-(n-1)**2] + n**2
            conn[0,0] = 1
            conn[0,1] = 2
            conn[0,2] = n+2
            conn[0,3] = n+1
            conn[0,4] = n**2 + 1
            conn[0,5] = n**2 + 2
            conn[0,6] = n**2 + n + 2
            conn[0,7] = n**2 + n + 1
            counter = counter + 1
            jj = 0
        else:
            jj = 0
        for jj in range(0, n-1): #outer jumps in x + 1
            if jj < n - 2:
                conn[counter] = conn[counter-1] + 1
                counter = counter + 1
                kk = 0
            else:
                kk = 0
                for kk in range(0, n-2): #inner jumps in y + 2
                    conn[counter] = conn[counter-1] + 2
                    counter = counter + 1
                    for jj in range(0, n-2): #outer jumps in x + 1
                        conn[counter] = conn[counter-1] + 1
                        counter = counter + 1
        

    #Adjust the indexes to 0-base
    conn = conn - 1

    for r in range(0, len(conn)):
        for col in range(0, 4):
            conn[r,col] = int(conn[r,col]) #Make the entries integers

    #print conn
        
    #Now I need to generate the nodes corresponding to the connectivity table

    nodes = np.zeros((n**3,3))
    nodes2 = np.zeros((n**3,3))

    space = np.linspace(0, 1, n)

    step = space[1] - space[0]

    cc1 = 0

    for zz in range(0, n):
        nodes[cc1,2] = space[zz]
        nodes[cc1,1] = 0
        nodes[cc1,0] = 0
        #cc1 = cc1 + 1
        nodes[0] = 0
        for yy in range(0, n):
            nodes[cc1,2] = space[zz]
            nodes[cc1,1] = space[yy]
            #cc1 = cc1 + 1
            for xx in range(0, n):
                nodes[cc1,2] = space[zz]
                nodes[cc1,1] = space[yy]
                nodes[cc1,0] = space[xx]
                cc1 = cc1 + 1


    return nodes, conn

#####################################################################################################################
#Mapping Functions

# The shape functions
def phi0(z1,z2,z3):
    phi0 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)*(1.0 - z3)
    return phi0

def phi1(z1,z2,z3):
    phi1 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)*(1.0 - z3)
    return phi1

def phi2(z1,z2,z3):
    phi2 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)*(1.0 - z3)
    return phi2

def phi3(z1,z2,z3):
    phi3 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)*(1.0 - z3)
    return phi3

def phi4(z1,z2,z3):
    phi4 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)*(1.0 + z3)
    return phi4

def phi5(z1,z2,z3):
    phi5 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)*(1.0 + z3)
    return phi5

def phi6(z1,z2,z3):
    phi6 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)*(1.0 + z3)
    return phi6

def phi7(z1,z2,z3):
    phi7 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)*(1.0 + z3)
    return phi7

#Derivatives of the shape function with respect to z1

def dphi0dz1(z1,z2,z3):
    dphi0dz1 = (1.0/8.0)*(-1.0)*(1.0 - z2)*(1.0 - z3)
    return dphi0dz1

def dphi1dz1(z1,z2,z3):
    dphi1dz1 = (1.0/8.0)*(1.0 - z2)*(1.0 - z3)
    return dphi1dz1

def dphi2dz1(z1,z2,z3):
    dphi2dz1 = (1.0/8.0)*(1.0 + z2)*(1.0 - z3)
    return dphi2dz1

def dphi3dz1(z1,z2,z3):
    dphi3dz1 = (1.0/8.0)*(-1.0)*(1.0 + z2)*(1.0 - z3)
    return dphi3dz1

def dphi4dz1(z1,z2,z3):
    dphi4dz1 = (1.0/8.0)*(-1.0)*(1.0 - z2)*(1.0 + z3)
    return dphi4dz1

def dphi5dz1(z1,z2,z3):
    dphi5dz1 = (1.0/8.0)*(1.0 - z2)*(1.0 + z3)
    return dphi5dz1

def dphi6dz1(z1,z2,z3):
    dphi6dz1 = (1.0/8.0)*(1.0 + z2)*(1.0 + z3)
    return dphi6dz1

def dphi7dz1(z1,z2,z3):
    dphi7dz1 = (1.0/8.0)*(-1.0)*(1.0 + z2)*(1.0 + z3)
    return dphi7dz1


#Derivatives of the shape function with respect to z2

def dphi0dz2(z1,z2,z3):
    dphi0dz2 = (1.0/8.0)*(1.0 - z1)*(-1.0)*(1.0 - z3)
    return dphi0dz2

def dphi1dz2(z1,z2,z3):
    dphi1dz2 = (1.0/8.0)*(1.0 + z1)*(-1.0)*(1.0 - z3)
    return dphi1dz2

def dphi2dz2(z1,z2,z3):
    dphi2dz2 = (1.0/8.0)*(1.0 + z1)*(1.0 - z3)
    return dphi2dz2

def dphi3dz2(z1,z2,z3):
    dphi3dz2 = (1.0/8.0)*(1.0 - z1)*(1.0 - z3)
    return dphi3dz2

def dphi4dz2(z1,z2,z3):
    dphi4dz2 = (1.0/8.0)*(1.0 - z1)*(-1.0)*(1.0 + z3)
    return dphi4dz2

def dphi5dz2(z1,z2,z3):
    dphi5dz2 = (1.0/8.0)*(1.0 + z1)*(-1.0)*(1.0 + z3)
    return dphi5dz2

def dphi6dz2(z1,z2,z3):
    dphi6dz2 = (1.0/8.0)*(1.0 + z1)*(1.0 + z3)
    return dphi6dz2

def dphi7dz2(z1,z2,z3):
    dphi7dz2 = (1.0/8.0)*(1.0 - z1)*(1.0 + z3)
    return dphi7dz2


#Derivatives of the shape function with respect to z3

def dphi0dz3(z1,z2,z3):
    dphi0dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)*(-1.0)
    return dphi0dz3

def dphi1dz3(z1,z2,z3):
    dphi1dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)*(-1.0)
    return dphi1dz3

def dphi2dz3(z1,z2,z3):
    dphi2dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)*(-1.0)
    return dphi2dz3

def dphi3dz3(z1,z2,z3):
    dphi3dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)*(-1.0)
    return dphi3dz3

def dphi4dz3(z1,z2,z3):
    dphi4dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 - z2)
    return dphi4dz3

def dphi5dz3(z1,z2,z3):
    dphi5dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 - z2)
    return dphi5dz3

def dphi6dz3(z1,z2,z3):
    dphi6dz3 = (1.0/8.0)*(1.0 + z1)*(1.0 + z2)
    return dphi6dz3

def dphi7dz3(z1,z2,z3):
    dphi7dz3 = (1.0/8.0)*(1.0 - z1)*(1.0 + z2)
    return dphi7dz3


#####################################################################################################################
#Generate the mesh
Nx = 6
[nodes, conn] = cmesh(Nx)

n_nodes = int(Nx**3)
n_elem = int((Nx-1)**3)

Ne = len(conn)
NP = len(nodes)

S = sps.lil_matrix((NP, NP))
fS1 = sps.lil_matrix((NP, NP))
fS2 = sps.lil_matrix((NP, NP))
fS3 = sps.lil_matrix((NP, NP))

R = np.zeros(NP)
fR1 = np.zeros(8)
fR2 = np.zeros(8)
fR3 = np.zeros(8)

F = np.zeros((3,3))

x1 = np.zeros(8)
x2 = np.zeros(8)
x3 = np.zeros(8)

dphatdz1vec = np.zeros(8)
dphatdz2vec = np.zeros(8)
dphatdz3vec = np.zeros(8)
phatvec = np.zeros(8)

inter1 = np.zeros(3)
inter2 = np.zeros(3)

#Conductivity and source, flux termx

Kt = 25.0

zsource = 100.0

qflux = 0.0

Tbardown = 200.0

Tbarup = 800.0


#Surface Normals and Penalty Terms

zdown = -1.0
Nzdown = np.array([0.0, 0.0, zdown])

zup = 1.0
Nzup = np.array([0.0, 0.0, zup])
Pstar = 10000000.0



#Gauss Vector
w5 = np.array([0.568888888888889, 0.478628670499366, 0.478628670499366, 0.236926885056189, 0.236926885056189])
gauss5 = np.array([0.000000000000000, 0.538469310105683, -0.538469310105683, 0.906179845938664, -0.906179845938664])



#####################################################################################################################
# Main Loop

print 'Integral 1 start'
print datetime.now() - startTime

for e in range(0,Ne):

    x1[:] = np.array(nodes[conn[e,:],0])
    x2[:] = np.array(nodes[conn[e,:],1])
    x3[:] = np.array(nodes[conn[e,:],2])


    
#Computation of Integral 1 - Stiffness Matrix Term
    
    for A in range(0,8):
        for B in range(0,8):
            count = 0
            inter = np.zeros((125,1))
            for ii in range(0,5):
                for jj in range(0,5):
                    for kk in range(0,5):
                        F11 = x1[0]*dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[1]*dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[2]*dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[3]*dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[4]*dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[5]*dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[6]*dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[7]*dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])

                        F12 = x1[0]*dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[1]*dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[2]*dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[3]*dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[4]*dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[5]*dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[6]*dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[7]*dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])

                        F13 = x1[0]*dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[1]*dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[2]*dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[3]*dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[4]*dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[5]*dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[6]*dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[7]*dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])

                        F21 = x2[0]*dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[1]*dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[2]*dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[3]*dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[4]*dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[5]*dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[6]*dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[7]*dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])

                        F22 = x2[0]*dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[1]*dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[2]*dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[3]*dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[4]*dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[5]*dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[6]*dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[7]*dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])

                        F23 = x2[0]*dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[1]*dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[2]*dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[3]*dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[4]*dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[5]*dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[6]*dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[7]*dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])

                        F31 = x3[0]*dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[1]*dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[2]*dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[3]*dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[4]*dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[5]*dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[6]*dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[7]*dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])

                        F32 = x3[0]*dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[1]*dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[2]*dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[3]*dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[4]*dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[5]*dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[6]*dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[7]*dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])

                        F33 = x3[0]*dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[1]*dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[2]*dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[3]*dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[4]*dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[5]*dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[6]*dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[7]*dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])

                        F = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

                        dphatdz1vec = [dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]), dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]), dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]), dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]), dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])]

                        dphatdz2vec = [dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]), dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]), dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]), dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]), dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])]

                        dphatdz3vec = [dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]), dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]), dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]), dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                       dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]), dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])]

                        Finv = inv(F)
                    
                        inter1 = np.dot(Finv.transpose(),[[dphatdz1vec[A]], [dphatdz2vec[A]], [dphatdz3vec[A]]])
                        inter2 = np.dot(Finv.transpose(),[[dphatdz1vec[B]], [dphatdz2vec[B]], [dphatdz3vec[B]]])

                        inter3 = inter1[0]*inter2[0] + inter1[1]*inter2[1] + inter1[2]*inter2[2]

                        inter[count] = w5[ii]*w5[jj]*w5[kk]*inter3*(np.linalg.det(F))*Kt

                        fS1[A,B] = sum(inter)

                        count =  count + 1

    for i in range(0,8):
        for j in range(0,8):
            S[conn[e,i],conn[e,j]] = S[conn[e,i],conn[e,j]] + fS1[i,j]

print 'Integral 1 end'
print datetime.now() - startTime

###############################################
#Computation of Integral 2 - Stiffness Matrix Term Boundary Term (Dirichlet)

# Dirichlet term on bottom z surface - z normal = -1

print 'Integral 2 start'
print datetime.now() - startTime
                
                
for e in range(0,(Nx-1)**2):

    x1[:] = np.array(nodes[conn[e,:],0])
    x2[:] = np.array(nodes[conn[e,:],1])
    x3[:] = np.array(nodes[conn[e,:],2])

    
    for A in range(0,8):
        for B in range(0,8):
            count = 0
            inter = np.zeros((25,1))
            for ii in range(0,5):
                for jj in range(0,5):
                        F11 = x1[0]*dphi0dz1(gauss5[ii],gauss5[jj],zdown) + x1[1]*dphi1dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x1[2]*dphi2dz1(gauss5[ii],gauss5[jj],zdown) + x1[3]*dphi3dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x1[4]*dphi4dz1(gauss5[ii],gauss5[jj],zdown) + x1[5]*dphi5dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x1[6]*dphi6dz1(gauss5[ii],gauss5[jj],zdown) + x1[7]*dphi7dz1(gauss5[ii],gauss5[jj],zdown)

                        F12 = x1[0]*dphi0dz2(gauss5[ii],gauss5[jj],zdown) + x1[1]*dphi1dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x1[2]*dphi2dz2(gauss5[ii],gauss5[jj],zdown) + x1[3]*dphi3dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x1[4]*dphi4dz2(gauss5[ii],gauss5[jj],zdown) + x1[5]*dphi5dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x1[6]*dphi6dz2(gauss5[ii],gauss5[jj],zdown) + x1[7]*dphi7dz2(gauss5[ii],gauss5[jj],zdown)

                        F13 = x1[0]*dphi0dz3(gauss5[ii],gauss5[jj],zdown) + x1[1]*dphi1dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x1[2]*dphi2dz3(gauss5[ii],gauss5[jj],zdown) + x1[3]*dphi3dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x1[4]*dphi4dz3(gauss5[ii],gauss5[jj],zdown) + x1[5]*dphi5dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x1[6]*dphi6dz3(gauss5[ii],gauss5[jj],zdown) + x1[7]*dphi7dz3(gauss5[ii],gauss5[jj],zdown)

                        F21 = x2[0]*dphi0dz1(gauss5[ii],gauss5[jj],zdown) + x2[1]*dphi1dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x2[2]*dphi2dz1(gauss5[ii],gauss5[jj],zdown) + x2[3]*dphi3dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x2[4]*dphi4dz1(gauss5[ii],gauss5[jj],zdown) + x2[5]*dphi5dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x2[6]*dphi6dz1(gauss5[ii],gauss5[jj],zdown) + x2[7]*dphi7dz1(gauss5[ii],gauss5[jj],zdown)

                        F22 = x2[0]*dphi0dz2(gauss5[ii],gauss5[jj],zdown) + x2[1]*dphi1dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x2[2]*dphi2dz2(gauss5[ii],gauss5[jj],zdown) + x2[3]*dphi3dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x2[4]*dphi4dz2(gauss5[ii],gauss5[jj],zdown) + x2[5]*dphi5dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x2[6]*dphi6dz2(gauss5[ii],gauss5[jj],zdown) + x2[7]*dphi7dz2(gauss5[ii],gauss5[jj],zdown)

                        F23 = x2[0]*dphi0dz3(gauss5[ii],gauss5[jj],zdown) + x2[1]*dphi1dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x2[2]*dphi2dz3(gauss5[ii],gauss5[jj],zdown) + x2[3]*dphi3dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x2[4]*dphi4dz3(gauss5[ii],gauss5[jj],zdown) + x2[5]*dphi5dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x2[6]*dphi6dz3(gauss5[ii],gauss5[jj],zdown) + x2[7]*dphi7dz3(gauss5[ii],gauss5[jj],zdown)

                        F31 = x3[0]*dphi0dz1(gauss5[ii],gauss5[jj],zdown) + x3[1]*dphi1dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x3[2]*dphi2dz1(gauss5[ii],gauss5[jj],zdown) + x3[3]*dphi3dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x3[4]*dphi4dz1(gauss5[ii],gauss5[jj],zdown) + x3[5]*dphi5dz1(gauss5[ii],gauss5[jj],zdown) + \
                              x3[6]*dphi6dz1(gauss5[ii],gauss5[jj],zdown) + x3[7]*dphi7dz1(gauss5[ii],gauss5[jj],zdown)

                        F32 = x3[0]*dphi0dz2(gauss5[ii],gauss5[jj],zdown) + x3[1]*dphi1dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x3[2]*dphi2dz2(gauss5[ii],gauss5[jj],zdown) + x3[3]*dphi3dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x3[4]*dphi4dz2(gauss5[ii],gauss5[jj],zdown) + x3[5]*dphi5dz2(gauss5[ii],gauss5[jj],zdown) + \
                              x3[6]*dphi6dz2(gauss5[ii],gauss5[jj],zdown) + x3[7]*dphi7dz2(gauss5[ii],gauss5[jj],zdown)

                        F33 = x3[0]*dphi0dz3(gauss5[ii],gauss5[jj],zdown) + x3[1]*dphi1dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x3[2]*dphi2dz3(gauss5[ii],gauss5[jj],zdown) + x3[3]*dphi3dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x3[4]*dphi4dz3(gauss5[ii],gauss5[jj],zdown) + x3[5]*dphi5dz3(gauss5[ii],gauss5[jj],zdown) + \
                              x3[6]*dphi6dz3(gauss5[ii],gauss5[jj],zdown) + x3[7]*dphi7dz3(gauss5[ii],gauss5[jj],zdown)

                        F = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

                        phatvec = [phi0(gauss5[ii],gauss5[jj],zdown), phi1(gauss5[ii],gauss5[jj],zdown), phi2(gauss5[ii],gauss5[jj],zdown), \
                                   phi3(gauss5[ii],gauss5[jj],zdown), phi4(gauss5[ii],gauss5[jj],zdown), phi5(gauss5[ii],gauss5[jj],zdown), \
                                   phi6(gauss5[ii],gauss5[jj],zdown), phi7(gauss5[ii],gauss5[jj],zdown)]

                        Finv = inv(F)

                        inter1 = phatvec[A]*Pstar*phatvec[B];
                        inter21 = np.dot(Nzdown, Finv)
                        inter22 = np.dot(Finv.transpose(), Nzdown)
                        inter2 = (np.dot(inter21,inter22))**0.5
                        
                 

                        inter[count] = w5[ii]*w5[jj]*inter1*inter2*(np.linalg.det(F))

                        fS2[A,B] = sum(inter)

                        count =  count + 1

    for i in range(0,8):
        for j in range(0,8):
            S[conn[e,i],conn[e,j]] = S[conn[e,i],conn[e,j]] + fS2[i,j]


###############################################
#Computation of Integral 2 - Stiffness Matrix Term Boundary Term (Dirichlet)

# Dirichlet term on top z surface - z normal = 1
                
                
for e in range(Ne-(Nx-1)**2,Ne):

    x1[:] = np.array(nodes[conn[e,:],0])
    x2[:] = np.array(nodes[conn[e,:],1])
    x3[:] = np.array(nodes[conn[e,:],2])

    
    for A in range(0,8):
        for B in range(0,8):
            count = 0
            inter = np.zeros((25,1))
            for ii in range(0,5):
                for jj in range(0,5):
                        F11 = x1[0]*dphi0dz1(gauss5[ii],gauss5[jj],zup) + x1[1]*dphi1dz1(gauss5[ii],gauss5[jj],zup) + \
                              x1[2]*dphi2dz1(gauss5[ii],gauss5[jj],zup) + x1[3]*dphi3dz1(gauss5[ii],gauss5[jj],zup) + \
                              x1[4]*dphi4dz1(gauss5[ii],gauss5[jj],zup) + x1[5]*dphi5dz1(gauss5[ii],gauss5[jj],zup) + \
                              x1[6]*dphi6dz1(gauss5[ii],gauss5[jj],zup) + x1[7]*dphi7dz1(gauss5[ii],gauss5[jj],zup)

                        F12 = x1[0]*dphi0dz2(gauss5[ii],gauss5[jj],zup) + x1[1]*dphi1dz2(gauss5[ii],gauss5[jj],zup) + \
                              x1[2]*dphi2dz2(gauss5[ii],gauss5[jj],zup) + x1[3]*dphi3dz2(gauss5[ii],gauss5[jj],zup) + \
                              x1[4]*dphi4dz2(gauss5[ii],gauss5[jj],zup) + x1[5]*dphi5dz2(gauss5[ii],gauss5[jj],zup) + \
                              x1[6]*dphi6dz2(gauss5[ii],gauss5[jj],zup) + x1[7]*dphi7dz2(gauss5[ii],gauss5[jj],zup)

                        F13 = x1[0]*dphi0dz3(gauss5[ii],gauss5[jj],zup) + x1[1]*dphi1dz3(gauss5[ii],gauss5[jj],zup) + \
                              x1[2]*dphi2dz3(gauss5[ii],gauss5[jj],zup) + x1[3]*dphi3dz3(gauss5[ii],gauss5[jj],zup) + \
                              x1[4]*dphi4dz3(gauss5[ii],gauss5[jj],zup) + x1[5]*dphi5dz3(gauss5[ii],gauss5[jj],zup) + \
                              x1[6]*dphi6dz3(gauss5[ii],gauss5[jj],zup) + x1[7]*dphi7dz3(gauss5[ii],gauss5[jj],zup)

                        F21 = x2[0]*dphi0dz1(gauss5[ii],gauss5[jj],zup) + x2[1]*dphi1dz1(gauss5[ii],gauss5[jj],zup) + \
                              x2[2]*dphi2dz1(gauss5[ii],gauss5[jj],zup) + x2[3]*dphi3dz1(gauss5[ii],gauss5[jj],zup) + \
                              x2[4]*dphi4dz1(gauss5[ii],gauss5[jj],zup) + x2[5]*dphi5dz1(gauss5[ii],gauss5[jj],zup) + \
                              x2[6]*dphi6dz1(gauss5[ii],gauss5[jj],zup) + x2[7]*dphi7dz1(gauss5[ii],gauss5[jj],zup)

                        F22 = x2[0]*dphi0dz2(gauss5[ii],gauss5[jj],zup) + x2[1]*dphi1dz2(gauss5[ii],gauss5[jj],zup) + \
                              x2[2]*dphi2dz2(gauss5[ii],gauss5[jj],zup) + x2[3]*dphi3dz2(gauss5[ii],gauss5[jj],zup) + \
                              x2[4]*dphi4dz2(gauss5[ii],gauss5[jj],zup) + x2[5]*dphi5dz2(gauss5[ii],gauss5[jj],zup) + \
                              x2[6]*dphi6dz2(gauss5[ii],gauss5[jj],zup) + x2[7]*dphi7dz2(gauss5[ii],gauss5[jj],zup)

                        F23 = x2[0]*dphi0dz3(gauss5[ii],gauss5[jj],zup) + x2[1]*dphi1dz3(gauss5[ii],gauss5[jj],zup) + \
                              x2[2]*dphi2dz3(gauss5[ii],gauss5[jj],zup) + x2[3]*dphi3dz3(gauss5[ii],gauss5[jj],zup) + \
                              x2[4]*dphi4dz3(gauss5[ii],gauss5[jj],zup) + x2[5]*dphi5dz3(gauss5[ii],gauss5[jj],zup) + \
                              x2[6]*dphi6dz3(gauss5[ii],gauss5[jj],zup) + x2[7]*dphi7dz3(gauss5[ii],gauss5[jj],zup)

                        F31 = x3[0]*dphi0dz1(gauss5[ii],gauss5[jj],zup) + x3[1]*dphi1dz1(gauss5[ii],gauss5[jj],zup) + \
                              x3[2]*dphi2dz1(gauss5[ii],gauss5[jj],zup) + x3[3]*dphi3dz1(gauss5[ii],gauss5[jj],zup) + \
                              x3[4]*dphi4dz1(gauss5[ii],gauss5[jj],zup) + x3[5]*dphi5dz1(gauss5[ii],gauss5[jj],zup) + \
                              x3[6]*dphi6dz1(gauss5[ii],gauss5[jj],zup) + x3[7]*dphi7dz1(gauss5[ii],gauss5[jj],zup)

                        F32 = x3[0]*dphi0dz2(gauss5[ii],gauss5[jj],zup) + x3[1]*dphi1dz2(gauss5[ii],gauss5[jj],zup) + \
                              x3[2]*dphi2dz2(gauss5[ii],gauss5[jj],zup) + x3[3]*dphi3dz2(gauss5[ii],gauss5[jj],zup) + \
                              x3[4]*dphi4dz2(gauss5[ii],gauss5[jj],zup) + x3[5]*dphi5dz2(gauss5[ii],gauss5[jj],zup) + \
                              x3[6]*dphi6dz2(gauss5[ii],gauss5[jj],zup) + x3[7]*dphi7dz2(gauss5[ii],gauss5[jj],zup)

                        F33 = x3[0]*dphi0dz3(gauss5[ii],gauss5[jj],zup) + x3[1]*dphi1dz3(gauss5[ii],gauss5[jj],zup) + \
                              x3[2]*dphi2dz3(gauss5[ii],gauss5[jj],zup) + x3[3]*dphi3dz3(gauss5[ii],gauss5[jj],zup) + \
                              x3[4]*dphi4dz3(gauss5[ii],gauss5[jj],zup) + x3[5]*dphi5dz3(gauss5[ii],gauss5[jj],zup) + \
                              x3[6]*dphi6dz3(gauss5[ii],gauss5[jj],zup) + x3[7]*dphi7dz3(gauss5[ii],gauss5[jj],zup)

                        F = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

                        phatvec = [phi0(gauss5[ii],gauss5[jj],zup), phi1(gauss5[ii],gauss5[jj],zup), phi2(gauss5[ii],gauss5[jj],zup), \
                                   phi3(gauss5[ii],gauss5[jj],zup), phi4(gauss5[ii],gauss5[jj],zup), phi5(gauss5[ii],gauss5[jj],zup), \
                                   phi6(gauss5[ii],gauss5[jj],zup), phi7(gauss5[ii],gauss5[jj],zup)]

                        Finv = inv(F)

                        inter1 = phatvec[A]*Pstar*phatvec[B];
                        inter21 = np.dot(Nzup, Finv)
                        inter22 = np.dot(Finv.transpose(), Nzup)
                        inter2 = (np.dot(inter21,inter22))**0.5
                        
                 

                        inter[count] = w5[ii]*w5[jj]*inter1*inter2*(np.linalg.det(F))

                        fS3[A,B] = sum(inter)

                        count =  count + 1

    for i in range(0,8):
        for j in range(0,8):
            S[conn[e,i],conn[e,j]] = S[conn[e,i],conn[e,j]] + fS3[i,j]


print 'Integral 2 end'
print datetime.now() - startTime

#####################################################################################################################
#Computation of the Load Vector (RHS)

#Intergral 3

print 'Integral 3 start'
print datetime.now() - startTime

for e in range(0,Ne):

    x1[:] = np.array(nodes[conn[e,:],0])
    x2[:] = np.array(nodes[conn[e,:],1])
    x3[:] = np.array(nodes[conn[e,:],2])

    
    for A in range(0,8):
            count = 0
            inter = np.zeros((125,1))
            for ii in range(0,5):
                for jj in range(0,5):
                    for kk in range(0,5):
                        F11 = x1[0]*dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[1]*dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                             x1[2]*dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[3]*dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                             x1[4]*dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[5]*dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                             x1[6]*dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[7]*dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])

                        F12 = x1[0]*dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[1]*dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[2]*dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[3]*dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[4]*dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[5]*dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[6]*dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[7]*dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])

                        F13 = x1[0]*dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[1]*dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[2]*dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[3]*dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[4]*dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[5]*dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x1[6]*dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x1[7]*dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])

                        F21 = x2[0]*dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[1]*dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[2]*dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[3]*dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[4]*dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[5]*dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[6]*dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[7]*dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])

                        F22 = x2[0]*dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[1]*dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[2]*dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[3]*dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[4]*dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[5]*dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[6]*dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[7]*dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])

                        F23 = x2[0]*dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[1]*dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[2]*dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[3]*dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[4]*dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[5]*dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x2[6]*dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x2[7]*dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])

                        F31 = x3[0]*dphi0dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[1]*dphi1dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[2]*dphi2dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[3]*dphi3dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[4]*dphi4dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[5]*dphi5dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[6]*dphi6dz1(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[7]*dphi7dz1(gauss5[ii],gauss5[jj],gauss5[kk])

                        F32 = x3[0]*dphi0dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[1]*dphi1dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[2]*dphi2dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[3]*dphi3dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[4]*dphi4dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[5]*dphi5dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[6]*dphi6dz2(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[7]*dphi7dz2(gauss5[ii],gauss5[jj],gauss5[kk])

                        F33 = x3[0]*dphi0dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[1]*dphi1dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[2]*dphi2dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[3]*dphi3dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[4]*dphi4dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[5]*dphi5dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + \
                              x3[6]*dphi6dz3(gauss5[ii],gauss5[jj],gauss5[kk]) + x3[7]*dphi7dz3(gauss5[ii],gauss5[jj],gauss5[kk])

                        F = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

                        phatvec = [phi0(gauss5[ii],gauss5[jj],gauss5[kk]), phi1(gauss5[ii],gauss5[jj],gauss5[kk]), phi2(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                   phi3(gauss5[ii],gauss5[jj],gauss5[kk]), phi4(gauss5[ii],gauss5[jj],gauss5[kk]), phi5(gauss5[ii],gauss5[jj],gauss5[kk]), \
                                   phi6(gauss5[ii],gauss5[jj],gauss5[kk]), phi7(gauss5[ii],gauss5[jj],gauss5[kk])]

                        Finv = inv(F)

                        inter1 = phatvec[A]*zsource

                        inter[count] = w5[ii]*w5[jj]*w5[kk]*inter1*(np.linalg.det(F))

                        fR1[A] = sum(inter)

                        count =  count + 1

    for i in range(0,8):
        R[conn[e,i]] = R[conn[e,i]] + fR1[i]

print 'Integral 3 end'
print datetime.now() - startTime


#####################################
#Integral 5

#Bottom Surface Dirichlet Boundary

print 'Integral 5 start'
print datetime.now() - startTime

for e in range(0,(Nx-1)**2):

    x1[:] = np.array(nodes[conn[e,:],0])
    x2[:] = np.array(nodes[conn[e,:],1])
    x3[:] = np.array(nodes[conn[e,:],2])

    
    for A in range(0,8):
        count = 0
        inter = np.zeros((25,1))
        for ii in range(0,5):
            for jj in range(0,5):
                F11 = x1[0]*dphi0dz1(gauss5[ii],gauss5[jj],zdown) + x1[1]*dphi1dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x1[2]*dphi2dz1(gauss5[ii],gauss5[jj],zdown) + x1[3]*dphi3dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x1[4]*dphi4dz1(gauss5[ii],gauss5[jj],zdown) + x1[5]*dphi5dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x1[6]*dphi6dz1(gauss5[ii],gauss5[jj],zdown) + x1[7]*dphi7dz1(gauss5[ii],gauss5[jj],zdown)

                F12 = x1[0]*dphi0dz2(gauss5[ii],gauss5[jj],zdown) + x1[1]*dphi1dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x1[2]*dphi2dz2(gauss5[ii],gauss5[jj],zdown) + x1[3]*dphi3dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x1[4]*dphi4dz2(gauss5[ii],gauss5[jj],zdown) + x1[5]*dphi5dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x1[6]*dphi6dz2(gauss5[ii],gauss5[jj],zdown) + x1[7]*dphi7dz2(gauss5[ii],gauss5[jj],zdown)

                F13 = x1[0]*dphi0dz3(gauss5[ii],gauss5[jj],zdown) + x1[1]*dphi1dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x1[2]*dphi2dz3(gauss5[ii],gauss5[jj],zdown) + x1[3]*dphi3dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x1[4]*dphi4dz3(gauss5[ii],gauss5[jj],zdown) + x1[5]*dphi5dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x1[6]*dphi6dz3(gauss5[ii],gauss5[jj],zdown) + x1[7]*dphi7dz3(gauss5[ii],gauss5[jj],zdown)

                F21 = x2[0]*dphi0dz1(gauss5[ii],gauss5[jj],zdown) + x2[1]*dphi1dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x2[2]*dphi2dz1(gauss5[ii],gauss5[jj],zdown) + x2[3]*dphi3dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x2[4]*dphi4dz1(gauss5[ii],gauss5[jj],zdown) + x2[5]*dphi5dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x2[6]*dphi6dz1(gauss5[ii],gauss5[jj],zdown) + x2[7]*dphi7dz1(gauss5[ii],gauss5[jj],zdown)

                F22 = x2[0]*dphi0dz2(gauss5[ii],gauss5[jj],zdown) + x2[1]*dphi1dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x2[2]*dphi2dz2(gauss5[ii],gauss5[jj],zdown) + x2[3]*dphi3dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x2[4]*dphi4dz2(gauss5[ii],gauss5[jj],zdown) + x2[5]*dphi5dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x2[6]*dphi6dz2(gauss5[ii],gauss5[jj],zdown) + x2[7]*dphi7dz2(gauss5[ii],gauss5[jj],zdown)

                F23 = x2[0]*dphi0dz3(gauss5[ii],gauss5[jj],zdown) + x2[1]*dphi1dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x2[2]*dphi2dz3(gauss5[ii],gauss5[jj],zdown) + x2[3]*dphi3dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x2[4]*dphi4dz3(gauss5[ii],gauss5[jj],zdown) + x2[5]*dphi5dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x2[6]*dphi6dz3(gauss5[ii],gauss5[jj],zdown) + x2[7]*dphi7dz3(gauss5[ii],gauss5[jj],zdown)

                F31 = x3[0]*dphi0dz1(gauss5[ii],gauss5[jj],zdown) + x3[1]*dphi1dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x3[2]*dphi2dz1(gauss5[ii],gauss5[jj],zdown) + x3[3]*dphi3dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x3[4]*dphi4dz1(gauss5[ii],gauss5[jj],zdown) + x3[5]*dphi5dz1(gauss5[ii],gauss5[jj],zdown) + \
                      x3[6]*dphi6dz1(gauss5[ii],gauss5[jj],zdown) + x3[7]*dphi7dz1(gauss5[ii],gauss5[jj],zdown)

                F32 = x3[0]*dphi0dz2(gauss5[ii],gauss5[jj],zdown) + x3[1]*dphi1dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x3[2]*dphi2dz2(gauss5[ii],gauss5[jj],zdown) + x3[3]*dphi3dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x3[4]*dphi4dz2(gauss5[ii],gauss5[jj],zdown) + x3[5]*dphi5dz2(gauss5[ii],gauss5[jj],zdown) + \
                      x3[6]*dphi6dz2(gauss5[ii],gauss5[jj],zdown) + x3[7]*dphi7dz2(gauss5[ii],gauss5[jj],zdown)

                F33 = x3[0]*dphi0dz3(gauss5[ii],gauss5[jj],zdown) + x3[1]*dphi1dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x3[2]*dphi2dz3(gauss5[ii],gauss5[jj],zdown) + x3[3]*dphi3dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x3[4]*dphi4dz3(gauss5[ii],gauss5[jj],zdown) + x3[5]*dphi5dz3(gauss5[ii],gauss5[jj],zdown) + \
                      x3[6]*dphi6dz3(gauss5[ii],gauss5[jj],zdown) + x3[7]*dphi7dz3(gauss5[ii],gauss5[jj],zdown)

                F = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

                phatvec = [phi0(gauss5[ii],gauss5[jj],zdown), phi1(gauss5[ii],gauss5[jj],zdown), phi2(gauss5[ii],gauss5[jj],zdown), \
                           phi3(gauss5[ii],gauss5[jj],zdown), phi4(gauss5[ii],gauss5[jj],zdown), phi5(gauss5[ii],gauss5[jj],zdown), \
                           phi6(gauss5[ii],gauss5[jj],zdown), phi7(gauss5[ii],gauss5[jj],zdown)]

                Finv = inv(F)

                inter1 = phatvec[A]*Pstar*Tbardown;
                inter21 = np.dot(Nzdown, Finv)
                inter22 = np.dot(Finv.transpose(), Nzdown)
                inter2 = (np.dot(inter21,inter22))**0.5

                        
                inter[count] = w5[ii]*w5[jj]*inter1*inter2*(np.linalg.det(F))

                fR2[A] = sum(inter)

                count =  count + 1

    for i in range(0,8):
        R[conn[e,i]] = R[conn[e,i]] + fR2[i]


#####################################
#Integral 5

#Top Surface Dirichlet Boundary

for e in range(Ne-(Nx-1)**2,Ne):

    x1[:] = np.array(nodes[conn[e,:],0])
    x2[:] = np.array(nodes[conn[e,:],1])
    x3[:] = np.array(nodes[conn[e,:],2])

    
    for A in range(0,8):
        count = 0
        inter = np.zeros((25,1))
        for ii in range(0,5):
            for jj in range(0,5):
                F11 = x1[0]*dphi0dz1(gauss5[ii],gauss5[jj],zup) + x1[1]*dphi1dz1(gauss5[ii],gauss5[jj],zup) + \
                      x1[2]*dphi2dz1(gauss5[ii],gauss5[jj],zup) + x1[3]*dphi3dz1(gauss5[ii],gauss5[jj],zup) + \
                      x1[4]*dphi4dz1(gauss5[ii],gauss5[jj],zup) + x1[5]*dphi5dz1(gauss5[ii],gauss5[jj],zup) + \
                      x1[6]*dphi6dz1(gauss5[ii],gauss5[jj],zup) + x1[7]*dphi7dz1(gauss5[ii],gauss5[jj],zup)

                F12 = x1[0]*dphi0dz2(gauss5[ii],gauss5[jj],zup) + x1[1]*dphi1dz2(gauss5[ii],gauss5[jj],zup) + \
                      x1[2]*dphi2dz2(gauss5[ii],gauss5[jj],zup) + x1[3]*dphi3dz2(gauss5[ii],gauss5[jj],zup) + \
                      x1[4]*dphi4dz2(gauss5[ii],gauss5[jj],zup) + x1[5]*dphi5dz2(gauss5[ii],gauss5[jj],zup) + \
                      x1[6]*dphi6dz2(gauss5[ii],gauss5[jj],zup) + x1[7]*dphi7dz2(gauss5[ii],gauss5[jj],zup)

                F13 = x1[0]*dphi0dz3(gauss5[ii],gauss5[jj],zup) + x1[1]*dphi1dz3(gauss5[ii],gauss5[jj],zup) + \
                      x1[2]*dphi2dz3(gauss5[ii],gauss5[jj],zup) + x1[3]*dphi3dz3(gauss5[ii],gauss5[jj],zup) + \
                      x1[4]*dphi4dz3(gauss5[ii],gauss5[jj],zup) + x1[5]*dphi5dz3(gauss5[ii],gauss5[jj],zup) + \
                      x1[6]*dphi6dz3(gauss5[ii],gauss5[jj],zup) + x1[7]*dphi7dz3(gauss5[ii],gauss5[jj],zup)

                F21 = x2[0]*dphi0dz1(gauss5[ii],gauss5[jj],zup) + x2[1]*dphi1dz1(gauss5[ii],gauss5[jj],zup) + \
                      x2[2]*dphi2dz1(gauss5[ii],gauss5[jj],zup) + x2[3]*dphi3dz1(gauss5[ii],gauss5[jj],zup) + \
                      x2[4]*dphi4dz1(gauss5[ii],gauss5[jj],zup) + x2[5]*dphi5dz1(gauss5[ii],gauss5[jj],zup) + \
                      x2[6]*dphi6dz1(gauss5[ii],gauss5[jj],zup) + x2[7]*dphi7dz1(gauss5[ii],gauss5[jj],zup)

                F22 = x2[0]*dphi0dz2(gauss5[ii],gauss5[jj],zup) + x2[1]*dphi1dz2(gauss5[ii],gauss5[jj],zup) + \
                      x2[2]*dphi2dz2(gauss5[ii],gauss5[jj],zup) + x2[3]*dphi3dz2(gauss5[ii],gauss5[jj],zup) + \
                      x2[4]*dphi4dz2(gauss5[ii],gauss5[jj],zup) + x2[5]*dphi5dz2(gauss5[ii],gauss5[jj],zup) + \
                      x2[6]*dphi6dz2(gauss5[ii],gauss5[jj],zup) + x2[7]*dphi7dz2(gauss5[ii],gauss5[jj],zup)

                F23 = x2[0]*dphi0dz3(gauss5[ii],gauss5[jj],zup) + x2[1]*dphi1dz3(gauss5[ii],gauss5[jj],zup) + \
                      x2[2]*dphi2dz3(gauss5[ii],gauss5[jj],zup) + x2[3]*dphi3dz3(gauss5[ii],gauss5[jj],zup) + \
                      x2[4]*dphi4dz3(gauss5[ii],gauss5[jj],zup) + x2[5]*dphi5dz3(gauss5[ii],gauss5[jj],zup) + \
                      x2[6]*dphi6dz3(gauss5[ii],gauss5[jj],zup) + x2[7]*dphi7dz3(gauss5[ii],gauss5[jj],zup)

                F31 = x3[0]*dphi0dz1(gauss5[ii],gauss5[jj],zup) + x3[1]*dphi1dz1(gauss5[ii],gauss5[jj],zup) + \
                      x3[2]*dphi2dz1(gauss5[ii],gauss5[jj],zup) + x3[3]*dphi3dz1(gauss5[ii],gauss5[jj],zup) + \
                      x3[4]*dphi4dz1(gauss5[ii],gauss5[jj],zup) + x3[5]*dphi5dz1(gauss5[ii],gauss5[jj],zup) + \
                      x3[6]*dphi6dz1(gauss5[ii],gauss5[jj],zup) + x3[7]*dphi7dz1(gauss5[ii],gauss5[jj],zup)

                F32 = x3[0]*dphi0dz2(gauss5[ii],gauss5[jj],zup) + x3[1]*dphi1dz2(gauss5[ii],gauss5[jj],zup) + \
                      x3[2]*dphi2dz2(gauss5[ii],gauss5[jj],zup) + x3[3]*dphi3dz2(gauss5[ii],gauss5[jj],zup) + \
                      x3[4]*dphi4dz2(gauss5[ii],gauss5[jj],zup) + x3[5]*dphi5dz2(gauss5[ii],gauss5[jj],zup) + \
                      x3[6]*dphi6dz2(gauss5[ii],gauss5[jj],zup) + x3[7]*dphi7dz2(gauss5[ii],gauss5[jj],zup)

                F33 = x3[0]*dphi0dz3(gauss5[ii],gauss5[jj],zup) + x3[1]*dphi1dz3(gauss5[ii],gauss5[jj],zup) + \
                      x3[2]*dphi2dz3(gauss5[ii],gauss5[jj],zup) + x3[3]*dphi3dz3(gauss5[ii],gauss5[jj],zup) + \
                      x3[4]*dphi4dz3(gauss5[ii],gauss5[jj],zup) + x3[5]*dphi5dz3(gauss5[ii],gauss5[jj],zup) + \
                      x3[6]*dphi6dz3(gauss5[ii],gauss5[jj],zup) + x3[7]*dphi7dz3(gauss5[ii],gauss5[jj],zup)

                F = np.array([[F11, F12, F13], [F21, F22, F23], [F31, F32, F33]])

                phatvec = [phi0(gauss5[ii],gauss5[jj],zup), phi1(gauss5[ii],gauss5[jj],zup), phi2(gauss5[ii],gauss5[jj],zup), \
                           phi3(gauss5[ii],gauss5[jj],zup), phi4(gauss5[ii],gauss5[jj],zup), phi5(gauss5[ii],gauss5[jj],zup), \
                           phi6(gauss5[ii],gauss5[jj],zup), phi7(gauss5[ii],gauss5[jj],zup)]

                Finv = inv(F)

                inter1 = phatvec[A]*Pstar*Tbarup;
                inter21 = np.dot(Nzup, Finv)
                inter22 = np.dot(Finv.transpose(), Nzup)
                inter2 = (np.dot(inter21,inter22))**0.5

                        
                inter[count] = w5[ii]*w5[jj]*inter1*inter2*(np.linalg.det(F))

                fR3[A] = sum(inter)

                count =  count + 1

    for i in range(0,8):
        R[conn[e,i]] = R[conn[e,i]] + fR3[i]

print 'Integral 5 end'
print datetime.now() - startTime

#####################################################################################################################
#Solve the linear sytem

print 'solver'

print datetime.now() - startTime

T = spsolve(S.tocsc(), R)

print 'solver end'

####################################################################################################################
#Write the VTK file for Paraview


temptest125element = open('temptest125element2.vtu', 'w')

temptest125element.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian"> \n')
temptest125element.write("  <UnstructuredGrid> \n")
temptest125element.write('    <Piece NumberOfPoints="{0}" NumberOfCells="{1}"> \n'.format(n_nodes, n_elem))
temptest125element.write('     <PointData Scalars="scalars"> \n')
temptest125element.write('      <DataArray type="Float32" Name="Temperature" format="ascii"> \n')

for L in T:
    temptest125element.write("\t {0} \n".format(L))

temptest125element.write('\n')

temptest125element.write('      </DataArray> \n')
temptest125element.write('     </PointData> \n')

temptest125element.write('\t<Points> \n')
temptest125element.write('\t  <DataArray type="Float32" NumberOfComponents="3" format="ascii"> \n')

for L in nodes:
    temptest125element.write("\t    {0} {1} {2} \n".format(*L))

temptest125element.write('\t  </DataArray> \n')
temptest125element.write('\t</Points> \n')
temptest125element.write('\t<Cells> \n')

temptest125element.write('\t  <DataArray type="Int32" Name="connectivity" format="ascii"> \n')

for L in conn:
    temptest125element.write("{0} {1} {2} {3} {4} {5} {6} {7} \n".format(*L))

temptest125element.write('\t  </DataArray> \n')
temptest125element.write('\t      <DataArray type="Int32" Name="offsets" format="ascii"> \n')

qq = 0
for q in range(0, len(conn)):
    qq = qq + len(conn[q])
    temptest125element.write('\t    {0}'.format(qq))

temptest125element.write('\n')
temptest125element.write('\t  </DataArray> \n')
temptest125element.write('\n')
temptest125element.write('\t      <DataArray type="UInt8" Name="types" format="ascii"> \n')
for w in range(0, len(conn)):
    ww = 12
    temptest125element.write('\t    {0}'.format(ww))

temptest125element.write('\n')
temptest125element.write('\t  </DataArray> \n')
temptest125element.write('\n')
temptest125element.write('\t</Cells> \n')
temptest125element.write('\n')

temptest125element.write('    </Piece> \n')
temptest125element.write('\n')
temptest125element.write('  </UnstructuredGrid> \n')
temptest125element.write('\n')
temptest125element.write('</VTKFile> \n')

temptest125element.close()

#print nodes
#print conn
#print cc1
#print space

print datetime.now() - startTime


