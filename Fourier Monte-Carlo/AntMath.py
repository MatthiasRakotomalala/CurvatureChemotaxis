#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:46:27 2024

@author: Matthias Rakotomalala
"""

from os import system
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta


def B(th, p1 = 0., p2 = 0., A11 = -1., A12 = 0., A22 = 0. , tau = 1.):
    """
        Evaluate the scalar field for the orientation given the derivatives of the chemical field
    """
    Hesspart = ((A22 - A11)*np.cos(th)*np.sin(th) + A12*(np.cos(th)**2 - np.sin(th)**2))
    Gradpart = -p1*np.sin(th) + p2*np.cos(th)
    return Gradpart + tau*Hesspart

def FreqMsh(N):
    k = np.arange(-N, (N+1))
    l = np.arange(-N, (N+1))
    return np.meshgrid(k,l, indexing='ij')

def Eval(x, y, ca, cb, N, P = 1.):
    Kevalcff, Levalcff = FreqMsh(N)
    return Evalwc(x, y, ca, cb, Kevalcff, Levalcff, P = P)

def EvalMltpl(x, y, ca, cb, N, P = 1.):
    Kevalcff, Levalcff = FreqMsh(N)
    return Evalwc(x, y, ca, cb, Kevalcff, Levalcff, P = P)

def OpD(ca, cb, N, P = 1.):
    """
        First derivative operator on Fourier coefficients
    """
    K, L = FreqMsh(N)
    return OpD_wc(ca, cb, K, L, P = P)

def OpDD(ca, cb, N, P = 1.):
    """
        Second derivative operator on Fourier coefficients
    """
    K, L = FreqMsh(N)
    Ksq = K**2
    Lsq = L**2
    KL = K*L
    return OpDD_wc(ca, cb, Ksq, KL, Lsq, P = P)

def OpD_wc(ca, cb, K, L, P = 1.):
    domscl = 2.*np.pi/P
    Dxca = -domscl*cb*K
    Dxcb =  domscl*ca*K
    Dyca = -domscl*cb*L
    Dycb =  domscl*ca*L
    return Dxca, Dxcb, Dyca, Dycb

def OpDD_wc(ca, cb, Ksq, KL, Lsq, P = 1.):
    domsclsq = (2.*np.pi/P)**2
    Dxxca = -domsclsq*ca*Ksq
    Dxxcb = -domsclsq*cb*Ksq
    Dyyca = -domsclsq*ca*Lsq
    Dyycb = -domsclsq*cb*Lsq
    Dxyca = -domsclsq*ca*KL
    Dxycb = -domsclsq*cb*KL
    return Dxxca, Dxxcb, Dxyca, Dxycb, Dyyca, Dyycb


def Evalwc(x, y, ca, cb, Kevalcff, Levalcff, P = 1.):
    """
        Eval C over multiple points X, given Fourier coefficients Ca(real part), Cb(complex part)
    """
    expmultDim = (...,) + (None,)*len(x.shape)
    Zeta = (Kevalcff[expmultDim]*x+Levalcff[expmultDim]*y)
    Mods = np.cos(Zeta*(2.*np.pi/P))*ca[expmultDim] - np.sin(Zeta*(2.*np.pi/P))*cb[expmultDim]
    return np.sum(Mods, axis = (0,1))


def EvalMltplwc(x, y, ca, cb, Kevalcff, Levalcff, P = 1.):
    """
    Eval multiple C over multiple points X

    Ca, Cb must be of shape (N, NFr+1, 2NFr + 1)
    where N is the number of evaluation points and NFr is the truncature

    X and Y must be of shape (N,)
    """

    Zeta = x[:,np.newaxis,np.newaxis]*Kevalcff+y[:,np.newaxis,np.newaxis]*Levalcff
    Mods = np.cos(Zeta*(2.*np.pi/P))*ca - np.sin(Zeta*(2.*np.pi/P))*cb

    return np.sum(Mods, axis = (1,2))


def LaplaceImpFourier(N, dt = .1, sigma = .1, gamma = .5, P = 1.):
    """
        Parabolic equation discrete  implicit scheme operator on the Fourier coefficients
    """
    k = np.arange(-N, (N+1))
    l = np.arange(-N, (N+1))
    domP= (2.*np.pi/P)**2
    A = 1./(1.+(-gamma + sigma*domP*(k[:,np.newaxis]**2+l**2))*dt)
    return A

def FundSolHeatFourierCoeff_multd(x0, y0, N, dt = .1, sigma = .1, gamma = -.5, P = 1.):
    """
        Fourier coefficents of the Green function of the heat equation on the Torus
    """
    k, l = FreqMsh(N)
    domP= 2.*np.pi/P

    expmultDim = (None,)*len(x0.shape) + (...,)

    decay = np.exp(dt*(-domP**2*sigma*(k**2+l**2)+gamma))

    zeta0 = x0[..., np.newaxis, np.newaxis]*k+y0[...,np.newaxis, np.newaxis]*l
    ca = np.cos(domP*zeta0)
    cb = -np.sin(domP*zeta0)
    ca *= decay[expmultDim]
    cb *= decay[expmultDim]
    return ca, cb

def FsimEngine(Xinit, Yinit, Thinit, Cainit, Cbinit,
               N = 100, Nt = 500, dt = .1, sigth = .1,
               sigx = .001, lmb = .8, Xi = .7, tau = 1.,
               sigc = .01, gamma = -.5, mu = .5, srceps = .0001, NFr = 30):

    X = np.zeros((Nt, N))
    Y = np.zeros((Nt, N))
    Theta = np.zeros((Nt, N))
    
    CNa = np.zeros((Nt, 2*NFr+1, 2*NFr+1))
    CNb = np.zeros((Nt, 2*NFr+1, 2*NFr+1))
    
    LplOp = LaplaceImpFourier(NFr, dt = dt, sigma = sigc, gamma = gamma)

    K, L = FreqMsh(NFr)
    Ksq, KL, Lsq = K**2, K*L, L**2

    X[0,:] = Xinit
    Y[0,:] = Yinit
    Theta[0,:] = Thinit
    
    Ca_last = Cainit
    Cb_last = Cbinit
    
    CNa[0,...] = Cainit.mean(axis = 0)
    CNb[0,...] = Cbinit.mean(axis = 0)

    orderN = N * Nt * NFr * NFr
    
    loadbar = 0
    print('.'*100)
    start_time = datetime.now()
    
    for k in range(1, Nt):
        X[k,:] = X[k-1,:] + lmb*np.cos(Theta[k-1,:])*dt + np.random.normal(0,np.sqrt(dt)*sigx, N)
        Y[k,:] = Y[k-1,:] + lmb*np.sin(Theta[k-1,:])*dt + np.random.normal(0,np.sqrt(dt)*sigx, N)

        X[k,:] = (X[k,:]+.5)%1.-.5
        Y[k,:] = (Y[k,:]+.5)%1.-.5

        faN, fbN = FundSolHeatFourierCoeff_multd(X[k,:], Y[k,:], NFr, dt = 1., sigma = sigc*srceps, gamma = 0.)
        Ca_last = LplOp[np.newaxis,...]*(Ca_last + mu*dt*faN)
        Cb_last = LplOp[np.newaxis,...]*(Cb_last + mu*dt*fbN)

        P1  = np.zeros(N)
        P2  = np.zeros(N)
        A11 = np.zeros(N)
        A22 = np.zeros(N)
        A12 = np.zeros(N)

        #excluded Average
        Ca_last_sum = Ca_last.sum(axis = 0)
        Cb_last_sum = Cb_last.sum(axis = 0)
        
        Ca_exAv =  (Ca_last_sum[np.newaxis, ...] - Ca_last)/(N-1)
        Cb_exAv =  (Cb_last_sum[np.newaxis, ...] - Cb_last)/(N-1)
        
        CNa[k,...] = Ca_last_sum/N
        CNb[k,...] = Cb_last_sum/N
    
        Dxca, Dxcb, Dyca, Dycb = OpD_wc(Ca_exAv, Cb_exAv, K, L, P = 1.)
        Dxxca, Dxxcb, Dxyca, Dxycb, Dyyca, Dyycb = OpDD_wc(Ca_exAv, Cb_exAv, Ksq, KL, Lsq, P = 1.)

        P1  = EvalMltplwc(X[k,:], Y[k,:], Dxca, Dxcb, K, L)
        P2  = EvalMltplwc(X[k,:], Y[k,:], Dyca, Dycb, K, L)
        A11 = EvalMltplwc(X[k,:], Y[k,:], Dxxca, Dxxcb, K, L)
        A12 = EvalMltplwc(X[k,:], Y[k,:], Dxyca, Dxycb, K, L)
        A22 = EvalMltplwc(X[k,:], Y[k,:], Dyyca, Dyycb, K, L)

        Theta[k,:] = Theta[k-1,:] + Xi*B(Theta[k-1,:], p1 = P1, p2 = P2,  A11 = A11, A12 = A12,  A22 = A22, tau = tau)*dt + np.random.normal(0,np.sqrt(dt)*sigth, N)
        if loadbar < 100*k/Nt:
            loadbar +=1
            system("clear")
            elapsed_time = datetime.now() - start_time
            remaining_time = elapsed_time * (Nt/k - 1.)
            
            print(
                "|" * loadbar + "." * (100 - loadbar) + "{}%".format(loadbar)+\
                " , Order O(N*Nt*NFr^2) = {:.2E}".format(Decimal(orderN))+' Estimated Remaing Time : '+str(remaining_time),
                flush=True,
            )
    print('Total Run Time : ' + str(datetime.now() - start_time))
    return X,Y, Theta, CNa, CNb
