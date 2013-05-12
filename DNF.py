#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Dymamic Neural Field with finite transmission speed
# Copyright (C) 2010 Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
# 
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
# 
# You should have received a copy of the GNU General Public License along with
# this program. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------------
#
# Dependencies:
#
#     python > 2.6 (required): http://www.python.org
#     numpy        (required): http://numpy.scipy.org
#     matplotlib   (optional): http://matplotlib.sourceforge.net
#
# -----------------------------------------------------------------------------
# Contributors:
#
#     Nicolas P. Rougier
#     Axel Hutt
#     Cyril Noël
#
# Contact Information:
#
#     Axel Hutt / Nicolas P. Rougier
#     INRIA Nancy - Grand Est research center
#     CS 20101
#     54603 Villers les Nancy Cedex France
#
# References:
#
#     Axel Hutt and Nicolas P. Rougier
#     "Activity spread and breathers induced by finite transmission
#      speeds in two-dimensional neural fields"
#     Physical Review Letter E, 2010, to appear.
#
# -----------------------------------------------------------------------------
'''
Numerical integration of dynamic neural fields with finite propagation speed

This script implements the numerical integration of a dynamic neural fields
with finite (or infinite) propagation speed:

  ∂V(x,t)                     ⌠                       |x-y|
τ ------- = I(x,t) - V(x,t) + ⎮  K(|x-y|) S( V(y, t - -----) ) d²y
    ∂t                        ⌡Ω                        c

where # V(x,t) is the potential of a neural population at position x and time t
      # Ω is the domain of integration of size lxl (mm²)
      # K(x) is a neighborhood function from [0,√2l] -> ℝ
      # S(x) is the firing rate of a single neuron from  ℝ⁺ -> ℝ
      # c is the velocity of an action potential (mm/s)
      # τ is the temporal decay of the synapse
      # I(x,t) is the input at position x and time t

Numerical parameters:
      # n  : space discretisation
      # dt : temporal discretisation (s)
      # t  : duration of the simulation (s)

The integration is made over the finite 2d domain [-l/2,+l/2]x[-l/2,+l/2]
discretized into n x n elements considered as a toric surface, during a period
of t seconds.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2,ifft2,fftshift,ifftshift


def disc(shape=(256,256), center=None, radius = 64):
    ''' Generate a numpy array containing a disc.

    :Parameters:
        `shape` : (int,int)
            Shape of the output array
        `center`: (int,int)
            Disc center
         `radius`: int
             Disc radius (if radius = 0 -> disc is 1 point)
    '''
    if not center:
        center = (shape[0]//2,shape[1]//2)    
    def distance(x,y):
        return np.sqrt((x-center[0])**2+(y-center[1])**2)
    D = np.fromfunction(distance,shape)
    return np.where(D<=radius,True,False).astype(np.float32)

def peel(Z, center=None, r=8):
    ''' Peel an array Z into several 'onion rings' of width r.

    :Parameters:
        `Z`: numpy.ndarray
            Array to be peeled
        `center`: (int,int)
            Center of the 'onion'
        `r` : int
            ring radius
    :Returns:
        `out` : [numpy.ndarray,...]
            List of n Z-onion rings with n ≥ 1
    '''
    if r <= 0 :
        raise exceptions.ValueError('Radius must be > 0')
    if not center:
        center = (Z.shape[0]//2,Z.shape[1]//2)
    if  (center[0] >= Z.shape[0] or center[1] >= Z.shape[1] or \
        center[0] < 0 or center[1] < 0 ) : 
        raise exceptions.ValueError('Center must be in the matrix')

    # Compute the maximum diameter to get number of rings
    dx = float(max(Z.shape[0]-center[0],center[0]))
    dy = float(max(Z.shape[1]-center[1],center[1]))
    radius = np.sqrt(dx**2+dy**2)

    # Generate 1+int(d/r) rings
    L = []
    K = Z.copy()
    n = 1+int(radius/r)
    for i in range(n):
        r1 = (i  )*r/2
        r2 = (i+1)*r/2
        K = (disc(Z.shape,center,2*r2) - disc(Z.shape,center,2*r1))*Z
        L.append(K)
    L[0][center[0],center[1]] = Z[center[0],center[1]]
    return L

def gaussian(x, sigma=1.0):
    ''' Gaussian function of the form exp(-x²/σ²)/(π.σ²) '''
    return 1.0/(sigma**2*np.pi)*np.exp(-x**2/(sigma**2))

def g(x, sigma=1.0):
    ''' Gaussian function of the form exp(-x²/2σ²)) '''
    return np.exp(-0.5*(x/sigma)**2)

def sigmoid(x):
    ''' Sigmoid function of the form 1/(1+exp(-x)) '''
    return 1.0/(1+np.exp(-x))


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Parameters
    # ----------
    l     = 10.00  # size of the field (mm)
    n     = 256    # space discretization
    c     = 10.0   # velocity of an action potential (m/s)
    t     = 1.450  # duration of simulation (in seconds)
    dt    = 0.010  # temporal discretisation (in seconds)
    tau   = 1.0    # temporal decay of the synapse
 
    # Input
    I = 2
    I0 = 1.0
    sigma_i = 0.2
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    y_inf, y_sup, cy, dy = -l/2, +l/2, 0, l/float(n)
    nx, ny = (x_sup-x_inf)/dx, (y_sup-y_inf)/dy
    X,Y = np.meshgrid(np.arange(x_inf,x_sup,dx), np.arange(y_inf,y_sup,dy))
    D = np.sqrt(X**2+Y**2)
    I_ext = I0*gaussian(D,sigma_i)

    # Initial state (t ≤ 0)
    V0 = 2.00083
    V = np.ones((n,n))*V0

    # Neighborhood function
    def K(X,Y):
        phi_0 = 0*np.pi/3.0
        phi_1 = 1*np.pi/3.0
        phi_2 = 2*np.pi/3.0
        K0    = 0.1
        k_c   = 10*np.pi/l
        sigma = 10
        return K0*(np.cos(k_c*(X*np.cos(phi_0)+Y*np.sin(phi_0))) + \
                   np.cos(k_c*(X*np.cos(phi_1)+Y*np.sin(phi_1))) + \
                   np.cos(k_c*(X*np.cos(phi_2)+Y*np.sin(phi_2)))) * \
                   np.exp(-np.sqrt(X*X+Y*Y)/sigma)

    # Firing rate function
    def S(X):
        return 2.0/(1.0+np.exp(-5.5*(X-3)))

    # Generate kernel rings
    x_inf, x_sup, cx, dx = -l/2, +l/2, 0, l/float(n)
    y_inf, y_sup, cy, dy = -l/2, +l/2, 0, l/float(n)
    nx, ny = (x_sup-x_inf)/dx, (y_sup-y_inf)/dy
    X,Y = np.meshgrid(np.arange(x_inf,x_sup,dx), np.arange(y_inf,y_sup,dy))
    K_ = K(X,Y)*dx*dy
    r = max(1,c*dt*n/l)
    Ki = peel(K_, center=(n//2,n//2), r=r)

    nrings = len(Ki) # Number of rings
    # Precompute Fourier transform for each kernel ring since they're
    # only used in the Fourier domain
    Ki = [fft2(fftshift(Ki[i])) for i in range(nrings)]

    # Print parameters
    print '---------------------'
    print 'Simulation parameters'
    print '---------------------'
    print 'Size of the field        : %.1fmm×%.1fmm' % (l,l)
    print 'Action potential velocity: %.1fmm/s' % c
    print 'Tau                      : %.2f' % tau
    print 'Space discretisation     : %d×%d' % (n,n)
    print 'Time discretisation      : %.2f ms' % (dt)
    print 'Simulation duration      : %.2f s' % t
    print 'Number of rings          : %d' % nrings
    print 'K sum                    : %f' % K_.sum()


    # Initialisation
    # ---------------
    # Initialisation of past S(V) values (from t=-Tmax to t=0, where Tmax =
    # nrings*dt) Since we're working in the Fourier domain, past values are
    # directly stored using their Fourier transform
    U = [fft2(S(V)),]*nrings

    t = 1.45
    V_schedule = [0.5, 0.75, 1.0, 1.25]
    V_copy = []
    # Run simulation
    # --------------
    for i in range(0,int(t/dt)):
        print 'Time %.3fms:' % (i*dt)
        print '   V_min = %.8f' % V.min()
        print '   V_max = %.8f' % V.max()
        
        L = Ki[0]*U[0]
        for j in range(1,nrings):
            L += Ki[j]*U[j]
        L = ifft2(L).real
        if (i < 60):   dV = dt/tau*(-V+L+I)
        else:          dV = dt/tau*(-V+L+I+I_ext)
        V += dV
        U = [fft2(S(V)),] + U[:-1]
        if (i*dt in V_schedule):
            V_copy.append(V.copy())
    
    n = len(V_copy)
    fig = plt.figure(figsize=(n*4.5,4))
    for i in range(n):
        plt.subplot(1,n,i+1)
        plt.imshow(V_copy[i], vmin=2.00, vmax=2.025, interpolation='bicubic')
        plt.yticks([])
        plt.xticks([])
        plt.title("t=%.2fms" % V_schedule[i])
    fig.savefig('figure-1.pdf')
    plt.show()


