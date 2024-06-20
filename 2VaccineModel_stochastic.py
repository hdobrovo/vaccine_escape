# -*- coding: utf-8 -*-
"""
Created on Sun May 30 16:14:46 2021

@author: Tarun Mohan
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib
import matplotlib.pyplot as plt

beta_0 = 1.12
beta_1 = beta_0
beta_2 = beta_0
beta_e = beta_0
# delta -> Natural Birth rate / Natural Death Rate
delta = 3.3 * 10**-5
# alpha -> Recovery Rate
alpha = 1.0 / 7.0
# gamma -> rate of vaccination (.001 -> 1) (*logspace*)
gamma_1 = .01
gamma_2 = gamma_1
#epsilon -> vaccine efficacy
eps_1 = .5
eps_2 = eps_1
# n -> total population
n = 328.2*10**6
#delta_v -> rate of death due to virus
delta_v = 0.00366
#omega -> rate of waning immunity
omega = 1.0 / 30.0
# s -> # suspectible people, iw-> # infected w. wild type virus
# ie -> # infected w. escaped virus, # v -> # vaccinated, r-> # recovered
# fixed points - set time derivatives equal to zero

def stoc_eqs_tauleap(INP):
    X = INP
    Rate = np.zeros((50))##propensity function [b1*T*V1 k1*E1 d1*I1 p1*I1 c1*V1 b2*T*V2 k2*E2 d2*I2 p2*I2 c2*V2]
    Transitions = np.zeros((50,15))##stoichiometric matrix, each row of which is a transition vector
    Rate[0] = delta*(n-X[14]); Transitions[0,:]=([+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[1] = beta_0*X[0]*X[1]/n;  Transitions[1,:]=([-1, +1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[2] = beta_1*X[0]*(X[2]+X[3])/n;  Transitions[2,:]=([-1, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[3] = beta_2*X[0]*(X[4]+X[5])/n;  Transitions[3,:]=([-1, 0, 0, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[4] = beta_e*X[0]*(X[6]+X[7]+X[8])/n;  Transitions[4,:]=([-1, 0, 0, 0, 0, 0, +1, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[5] = omega*X[11]; Transitions[5,:]=([+1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0])
    Rate[6] = gamma_1*X[0]; Transitions[6,:]=([-1, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0, 0, 0])
    Rate[7] = gamma_2*X[0]; Transitions[7,:]=([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0, 0])
    Rate[8] = delta*X[0]; Transitions[8,:]=([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[9] = delta*X[1];  Transitions[9,:]=([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[10] = alpha*X[1];  Transitions[10,:]=([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0])
    Rate[11] = delta_v*X[1];  Transitions[11,:]=([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1])
    Rate[12] = delta*X[2];  Transitions[12,:]=([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[13] = alpha*X[2];  Transitions[13,:]=([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0])
    Rate[14] = delta_v*X[2];  Transitions[14,:]=([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1])   
    Rate[15] = delta*X[3];  Transitions[15,:]=([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[16] = alpha*X[3];  Transitions[16,:]=([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0])
    Rate[17] = delta_v*X[3];  Transitions[17,:]=([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1])
    Rate[18] = delta*X[4];  Transitions[18,:]=([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[19] = alpha*X[4];  Transitions[19,:]=([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, +1, 0, 0, 0])
    Rate[20] = delta_v*X[4];  Transitions[20,:]=([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1])  
    Rate[21] = delta*X[5];  Transitions[21,:]=([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[22] = alpha*X[5];  Transitions[22,:]=([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, +1, 0])
    Rate[23] = delta_v*X[5];  Transitions[23,:]=([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, +1]) 
    Rate[24] = delta*X[6];  Transitions[24,:]=([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0])
    Rate[25] = alpha*X[6];  Transitions[25,:]=([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, +1, 0, 0, 0])
    Rate[26] = delta_v*X[6];  Transitions[26,:]=([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, +1])   
    Rate[27] = delta*X[7];  Transitions[27,:]=([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0])
    Rate[28] = alpha*X[7];  Transitions[28,:]=([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, +1, 0, 0])
    Rate[29] = delta_v*X[7];  Transitions[29,:]=([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, +1])           
    Rate[30] = delta*X[8];  Transitions[30,:]=([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0])
    Rate[31] = alpha*X[8];  Transitions[31,:]=([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, +1, 0])
    Rate[32] = delta_v*X[8];  Transitions[32,:]=([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, +1]) 
    Rate[33] = beta_1*X[9]*(X[2]+X[3])/n; Transitions[33,:]=([0, 0, 0, +1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
    Rate[34] = (1-epsilon)*beta_0*X[9]*X[1]/n; Transitions[34,:]=([0, 0, 0, +1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])    
    Rate[35] = beta_e*X[9]*(X[6]+X[7]+X[8])/n; Transitions[35,:]=([0, 0, 0, 0, 0, 0, 0, +1, 0, -1, 0, 0, 0, 0, 0])  
    Rate[36] = (1-epsilon)*beta_1*X[9]*(X[2]+X[3])/n; Transitions[36,:]=([0, 0, 0, 0, 0, 0, 0, 0, +1, -1, 0, 0, 0, 0, 0])   
    Rate[37] = beta_2*X[10]*(X[4]+X[5])/n; Transitions[37,:]=([0, 0, 0, 0, 0, +1, 0, 0, 0, 0, -1, 0, 0, 0, 0])
    Rate[38] = (1-epsilon)*beta_0*X[10]*X[1]/n; Transitions[38,:]=([0, 0, 0, 0, 0, +1, 0, 0, 0, 0, -1, 0, 0, 0, 0])    
    Rate[39] = beta_e*X[10]*(X[6]+X[7]+X[8])/n; Transitions[39,:]=([0, 0, 0, 0, 0, 0, 0, 0, +1, 0, -1, 0, 0, 0, 0])  
    Rate[40] = (1-epsilon)*beta_1*X[10]*(X[4]+X[5])/n; Transitions[40,:]=([0, 0, 0, 0, 0, 0, 0, +1, 0, 0, -1, 0, 0, 0, 0])            
    Rate[41] = delta*X[9];  Transitions[41,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0])
    Rate[42] = omega*X[12];  Transitions[42,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, -1, 0, 0])
    Rate[43] = gamma_1*X[11];  Transitions[43,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, -1, 0, 0, 0])
    Rate[44] = delta*X[10];  Transitions[44,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0])
    Rate[45] = omega*X[13];  Transitions[45,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, 0, -1, 0])
    Rate[46] = gamma_2*X[12];  Transitions[46,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, +1, 0, -1, 0, 0])  
    Rate[47] = delta*X[11];  Transitions[47,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0])
    Rate[48] = delta*X[12];  Transitions[48,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0])
    Rate[49] = delta*X[13];  Transitions[49,:]=([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0])              
    for k in np.arange(1,50):
        leap=np.random.poisson(Rate[k]*tau);#no of times each transition occurs in the time interval dt or tau
    		## To avoid negative numbers
        Use=min([leap, X[np.where(Transitions[k,:]<0)[0][0]]]);
        X=X+Transitions[k,:]*Use;
    return X

def Stoch_Iteration(INPUT):
    s = [0]
    i0 = [0]
    i10 = [0]
    i11 = [0]
    i20 = [0]
    i22 = [0]
    ie0 = [0]
    ie1 = [0]
    ie2 = [0]
    v1 = [0]
    v2 = [0]
    r0 = [0]
    r1 = [0]
    r2 = [0]
    d = [0]
    for tt in time:
        res=stoc_eqs_tauleap(INPUT)
        s.append(INPUT[0])
        i0.append(INPUT[1])
        i10.append(INPUT[2])
        i11.append(INPUT[3])
        i20.append(INPUT[4])
        i22.append(INPUT[5])
        ie0.append(INPUT[6])
        ie1.append(INPUT[7])
        ie2.append(INPUT[8])
        v1.append(INPUT[9])
        v2.append(INPUT[10])
        r0.append(INPUT[11])
        r1.append(INPUT[12])
        r2.append(INPUT[13])
        d.append(INPUT[14])      
        INPUT=res
    return ([s, i0, i10, i11, i20, i22, ie0, ie1, ie2, v1, v2, r0, r1, r2, d])    
        
INPUT = [n-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0]
tau = 0.01 
Max = 150
time=np.arange(0.0, Max, tau)
#[s, iw, ieu, ie, v, r, rv, d] = Stoch_Iteration(INPUT)

gam = np.logspace(-6,0,100)
eps = np.linspace(1,0,100)
esc = (100,100)
esc = np.zeros(esc)
n_simulations = 10

for i in range(100):
    for j in range(100):
        tmp = 0;
        for g in range(n_simulations+1):
            epsilon = eps[i]
            gamma_1 = gam[j]
            gamma_2 = gam[j]
            [s, i0, i10, i11, i20, i22, ie0, ie1, ie2, v1, v2, r0, r1, r2, d] = Stoch_Iteration(INPUT)
            tmp = tmp + (sum(ie1) + sum(ie2) + sum(ie0)) / (sum(ie1) + sum(ie2) + sum(ie0) + sum(i0) + sum(i10) + sum(i11) + sum(i20) + sum(i22))
        esc[i,j] = tmp/n_simulations
        print(esc[i,j])

np.savetxt("2v_model.dat",esc)
