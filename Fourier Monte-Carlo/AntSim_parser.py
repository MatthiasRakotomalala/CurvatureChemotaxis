#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:47:50 2024

@author: Matthias Rakotomalala
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
from decimal import Decimal
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation
from AntMath import *

ROOT_PATH = ''

parser = argparse.ArgumentParser(description="AntSim")
# Add the argument parsers with default values
parser.add_argument("--N", type=int, default=5, help="Number of particles")
parser.add_argument("--Nt", type=int, default=100, help="Number of time steps")
parser.add_argument("--NFr", type=int, default=20, help="Number of Fourier coefficients")
parser.add_argument("--T", type=float, default=2.0, help="Total time")
parser.add_argument("--sigc", type=float, default=0.1, help="Sigma c parameter")
parser.add_argument("--tau", type=float, default=0.1, help="Tau parameter")

# Add parsers for the missing values with their default values
parser.add_argument("--gamma", type=float, default=-5.0, help="Gamma parameter")
parser.add_argument("--mu", type=float, default=5.0, help="Mu parameter")
parser.add_argument("--sigth", type=float, default=0.3, help="Sigma theta parameter")
parser.add_argument("--sigx", type=float, default=0.001, help="Sigma x parameter")
parser.add_argument("--lmb", type=float, default=0.7, help="Lambda parameter")
parser.add_argument("--Xi", type=float, default=0.7, help="Xi parameter")
parser.add_argument("--srceps", type=float, default=0.001, help="Source epsilon parameter")

parser.add_argument('--Uniform', action='store_true')
parser.add_argument('--DiracMass', dest='Uniform', action='store_false')
parser.set_defaults(Uniform=True)

parser.add_argument('--createVideo', action='store_true')
parser.add_argument('--noVideo', dest='createVideo', action='store_false')
parser.set_defaults(createVideo=False)

# Parse the arguments
args = parser.parse_args()

# Assign the parsed arguments to variables
N = args.N
Nt = args.Nt
NFr = args.NFr
T = args.T
sigc = args.sigc
tau = args.tau
gamma = args.gamma
mu = args.mu
sigth = args.sigth
sigx = args.sigx
lmb = args.lmb
Xi = args.Xi
srceps = args.srceps

# Calculate dt
dt = T / Nt

# Create a dictionary to hold all variables
simparams = {
    'N': N,
    'T': T,
    'Nt': Nt,
    'NFr': NFr,
    'sigc':sigc,
    'tau' : tau,
    'gamma' : gamma,
    'mu':mu,
    'sigth':sigth,
    'sigx':sigx,
    'lmb':lmb,
    'Xi':Xi,
    'srceps':srceps
}

current_time = datetime.now().strftime("%m%d%H%M%S")

run_id = "{}".format(current_time)

outdir = ROOT_PATH + "data/{}/".format(run_id)
figuresdir = ROOT_PATH + "Figures/{}/".format(run_id)
Path(outdir).mkdir(parents=True, exist_ok=True)
Path(figuresdir).mkdir(parents=True, exist_ok=True)

# Save the dictionary to a file
with open(outdir+'parameters.pkl', 'wb') as f:
    pickle.dump(simparams, f)


s_particles = 0.2
scale_arrows = 90

title = title = (
    r"$N = $"
    + str(N)
    + r"$, T = $"
    + str(T)
    + r"$, \sigma_{\theta} = $"
    + str(sigth)
    + r"$, \sigma_x = $"
    + str(sigx)
    + ",\n"
    + r"$\lambda = $"
    + str(lmb)
    + r"$, \chi = $"
    + str(Xi)
    + r"$, \tau = $"
    + str(round(tau, 2))
    + r"$, \sigma_c = $"
    + str(sigc)
    + r"$, \gamma = $"
    + str(gamma)
    + ",\n"
    + r"$\mu = $"
    + str(mu)
    + r".$N_F =$"
    + str(NFr)
    + r"$, N_t = $"
    + str(Nt)
    + r"$, \epsilon = {:.1E}.$".format(Decimal(srceps))
)

title_oneline = title.replace("\n", "")

Xinit = np.zeros(N)
Yinit = np.zeros(N)
Thinit = np.random.uniform(-np.pi, np.pi, N)

if args.Uniform :
    Xinit = np.random.uniform(-0.5, 0.5, N)
    Yinit = np.random.uniform(-0.5, 0.5, N)

Cainit = np.zeros((N, 2 * NFr + 1, 2 * NFr + 1))
Cbinit = np.zeros((N, 2 * NFr + 1, 2 * NFr + 1))

X, Y, Theta, Ca, Cb = FsimEngine(
    Xinit,
    Yinit,
    Thinit,
    Cainit,
    Cbinit,
    N=N,
    Nt=Nt,
    dt=dt,
    sigth=sigth,
    sigx=sigx,
    lmb=lmb,
    Xi=Xi,
    tau=tau,
    sigc=sigc,
    gamma=gamma,
    mu=mu,
    NFr=NFr,
    srceps=srceps,
)

##Save Data
PartPost = np.zeros((Nt, N, 3))
PartPost[..., 0] = X
PartPost[..., 1] = Y
PartPost[..., 2] = Theta

FieldC = np.zeros((Nt, 2 * NFr + 1, 2 * NFr + 1))
FieldC = Ca + 1.0j * Cb

np.save(outdir + "ParticlePost.npy", PartPost)
np.save(outdir + "FieldCFourier.npy", FieldC)

## FIRST PRINT
plt.ioff()
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(15, 7))

timesp = 0
alphapart = 0.5
xylin = np.linspace(-0.5, 0.5, 50)
Xgrid, Ygrid = np.meshgrid(xylin, xylin)

plt.subplots_adjust(right=0.85)
for j in range(8):
    Z = Eval(Xgrid,Ygrid, Ca[timesp,:], Cb[timesp,:], NFr)
    cntr = axs[j//4,j%4].contourf(Xgrid, Ygrid, Z, levels = 20, cmap = 'Blues', zorder = 0)
    axs[j//4,j%4].scatter(X[timesp,:], Y[timesp,:], marker = '.', color = 'orange', alpha = alphapart, s = s_particles, zorder=2)
    axs[j//4,j%4].quiver(X[timesp,:], Y[timesp,:], np.cos(Theta[timesp,:]), np.sin(Theta[timesp,:]),alpha = alphapart, scale =scale_arrows, zorder=1, color = 'orange')
    axs[j//4,j%4].title.set_text('t='+str(round(timesp*dt,2)))
    timesp += Nt//8

plt.subplots_adjust(wspace=0)
_ = fig.suptitle(title_oneline)

filename = "MonteCarloParticleFourierSimMltpTimes"
fig.savefig(figuresdir + filename + ".pdf", bbox_inches="tight", dpi=300, format="pdf")
plt.close(fig)

## SECOND PRINT
xylin = np.linspace(-0.5, 0.5, 100)
plt.ioff()
Xgrid, Ygrid = np.meshgrid(xylin, xylin)
it = -1
Z = Eval(Xgrid, Ygrid, Ca[it, :], Cb[it, :], NFr)

fig, ax = plt.subplots()

alphapart = 0.5
contourLast = ax.contourf(Xgrid, Ygrid, Z, levels=20, cmap="Blues")
ax.scatter(
    X[it, :], Y[it, :], marker=".", color="orange", alpha=alphapart, s=s_particles
)
ax.quiver(
    X[it, :],
    Y[it, :],
    np.cos(Theta[it, :]),
    np.sin(Theta[it, :]),
    alpha=alphapart,
    scale=scale_arrows,
    color="orange",
)

plt.title(title)

ax.set_aspect("equal", adjustable="box")
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
_ = plt.axis("off")


filename = "MonteCarloParticleFourierSimLastTime"
plt.savefig(
    figuresdir + filename + ".pdf",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
    format="pdf",
)
plt.close(fig)

## THIRD PRINT
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["figure.dpi"] = 150
plt.ioff()

fig, ax = plt.subplots()

sclt = 1
xylin = np.linspace(-0.5, 0.5, 50)
Xgrid, Ygrid = np.meshgrid(xylin, xylin)

Antsct = ax.scatter(
    X[0, :], Y[0, :], marker=".", color="orange", alpha=alphapart, s=s_particles
)
AntQuiv = ax.quiver(
    X[0, :],
    Y[0, :],
    np.cos(Theta[0, :]),
    np.sin(Theta[0, :]),
    alpha=alphapart,
    scale=scale_arrows,
    zorder=0,
    color="orange",
)

levelsContourf = contourLast.levels
Z = Eval(Xgrid, Ygrid, Ca[0, :], Cb[0, :], NFr)
contour = ax.contourf(Xgrid, Ygrid, Z, levels=levelsContourf, cmap="Blues", zorder=-1)


def update(t):
    global contour

    Antsct.set_offsets(np.c_[X[sclt * t, :], Y[sclt * t, :]])
    AntQuiv.set_offsets(np.c_[X[sclt * t, :], Y[sclt * t, :]])
    AntQuiv.set_UVC(np.cos(Theta[sclt * t, :]), np.sin(Theta[sclt * t, :]))

    Z = Eval(Xgrid, Ygrid, Ca[sclt * t, :], Cb[sclt * t, :], NFr)
    Z = np.clip(Z, None, levelsContourf[-1])
    for c in contour.collections:
        c.remove()  # Remove old contours
    contour = ax.contourf(
        Xgrid, Ygrid, Z, levels=levelsContourf, cmap="Blues", zorder=-1
    )

    return (
        contour.collections,
        Antsct,
        AntQuiv,
    )

fig.set_size_inches(5., 5.)

plt.xlim((-0.5, 0.5))
plt.ylim((-0.5, 0.5))
plt.axis("equal")
plt.axis("off")
fig.tight_layout(pad=0.1)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=Nt)

# To display the animation without the static plot
if args.createVideo:
    print('Building video')
    writer = matplotlib.animation.PillowWriter(fps=12)
    ani.save(
        figuresdir + "animationSimulation.gif",
        dpi=300,
        writer= writer,
    )
plt.close(fig)
