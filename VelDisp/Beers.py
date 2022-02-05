#!/usr/bin/env python

import numpy as np 

def z2v(z_i,c):
    return 3e5 * (z_i - c)/(1. + c)

def center_cluster_biweight(z_i):
    M = np.median(z_i)
    for __ in range(10): 
        MAD = np.median(abs(z_i-M))
        u_i = (z_i-M)/(6.0*MAD)
        idx = abs(u_i)<1
        C_bi = M + (np.sum((z_i[idx]-M) * (1-u_i[idx]**2)**2)
                    /np.sum((1-u_i[idx]**2)**2))
        M = C_bi
        if abs(M-C_bi)/C_bi<0.01:
            break
    return C_bi


def sigma_bi(v_i):
    for __ in range(10):
        v_avg = np.mean(v_i)
        n = len(v_i)
        MAD = np.median(abs(v_i-v_avg))
        u_i = (v_i-v_avg)/(9*MAD)
        
        idx = abs(u_i)<1
        
        D = np.sum((1-u_i[idx]**2)*(1-5*u_i[idx]**2))
        sig_bi = np.sqrt(n * np.sum((1-u_i[idx]**2)**4*(v_i[idx]-v_avg)**2)
                         /(D*(D-1)))
        
        v_i = v_i[abs(v_i)<3*sig_bi]

    return sig_bi


def gapper(v_i):
    for __ in range(10):
        n = len(v_i)
        v_i = np.sort(v_i)
        gw = 0
        for i in range(n-1):
            g_i = v_i[i+1]-v_i[i]
            w_i = i*(n-i)
            gw += g_i*w_i
        s_g = (np.sqrt(np.pi)/(n*(n-1)))*gw
        v_i = v_i[abs(v_i)<3*s_g]
        
        
    return s_g