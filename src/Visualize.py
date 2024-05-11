#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import torch 
#import torch.nn.functional as F

import math
import os

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = '35'
plt.rcParams['font.family'] = 'DejaVu Serif'
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
import mpl_toolkits.mplot3d as Axes3D


def error_plot(error, length, width):

    plt.rcParams['font.size'] = '30'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=35)
    Nx, Ny = error.shape
    X, Y = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny), indexing='ij')
    bound = np.max(abs(error))
    levels = np.linspace(-bound, bound, 50)
    cset = ax.contourf(X, Y, error, levels, cmap="coolwarm")#"Spectral_r")     
    plt.xlim([0, Nx]); plt.ylim([0, Ny])
    plt.xticks([0,Nx//3,Nx//3*2,Nx], [0, np.round(length/3, 4),np.round(length/3*2, 4), np.round(length, 4)])
    plt.yticks([Ny//3,Ny//3*2,Ny], [np.round(width/3, 4),np.round(width/3*2, 4), np.round(width, 4)])

    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    bound *= 0.99
    cbar.set_ticks(np.round([-bound, -bound/2, 0, bound/2, bound], 2))
    cbar.ax.set_title('$\Delta$T(℃)', fontsize=30, pad=25)
    plt.show()

def temp_plot(Temps, length, width, reso=1, rdbu=False, name=None):

    plt.rcParams['font.size'] = '30'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=35)
    Nx, Ny = Temps.shape
    X, Y = np.meshgrid(np.arange(0, Nx, reso), np.arange(0, Ny, reso), indexing='ij')
    levels = np.linspace(np.floor(Temps.min()*10)/10, np.ceil(Temps.max()*10)/10+0.1, 50)
    if rdbu:
        cset = ax.contourf(X, Y, Temps, levels, cmap="RdBu")    
    else:
        cset = ax.contourf(X, Y, Temps, levels, cmap=plt.cm.jet)    
    plt.xlim([0, Nx]); plt.ylim([0, Ny])
    plt.xticks([0,Nx//3,Nx//3*2,Nx], [0, np.round(length/3, 4),np.round(length/3*2, 4), np.round(length, 4)])
    plt.yticks([Ny//3,Ny//3*2,Ny], [np.round(width/3, 4),np.round(width/3*2, 4), np.round(width, 4)])
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    cbar.set_ticks([np.round(val, 1) for val in np.linspace(levels[0], levels[-1], 6).tolist()])
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=30)
    cbar.ax.set_title('T(℃)', fontsize=30, pad=25)
    plt.show()

def power_plot(flp_df):
    from matplotlib.ticker import FuncFormatter
    plt.rcParams['font.size'] = '30'
    plt.rcParams['font.family'] = 'DejaVu Serif'
    
    length = np.max(flp_df['X'] + flp_df['Length (m)'])
    width = np.max(flp_df['Y'] + flp_df['Width (m)'])
    flp_df["Powerdens"] = (flp_df["Power_dyn"]+flp_df["Power_leak"])/flp_df["Length (m)"]/flp_df["Width (m)"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.tick_params(labelsize=24)
    cmap = plt.cm.get_cmap("turbo")

    for i in range(flp_df.shape[0]):
        name = flp_df.iloc[i]['UnitName']
        x, y = flp_df.iloc[i]["X"], flp_df.iloc[i]["Y"]
        rect = plt.Rectangle((x, y), flp_df.iloc[i]["Length (m)"], flp_df.iloc[i]["Width (m)"], 
                linewidth=0, facecolor=cmap(flp_df.iloc[i]["Powerdens"]/flp_df["Powerdens"].max()))
        ax.add_patch(rect)

    ax.set_xlim(0, length); ax.set_xticks([0,length/3,length/3*2,length])
    ax.set_ylim(0, width); ax.set_yticks([width/3,width/3*2,width])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([]) # 此行是为了使ScalarMappable正常工作

    position = fig.add_axes([0.99, 0.15, 0.02, 0.65])
    cbar = plt.colorbar(sm, ax=ax, pad=0, fraction=0, cax=position)

#    cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_format_func))
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbarvals = np.linspace(0, np.ceil(flp_df["Powerdens"].max()), 6)
    cbar.set_ticklabels(['{:.1e}'.format(val) for val in cbarvals])
    #cbar.set_ticklabels([sci_format_func(val, None) for val in cbarvals])
    #cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), fontsize=30)
    cbar.ax.set_title('W/m$^2$', fontsize=30, pad=15 )
    plt.show()

def Temp_with_FP(Temps, reso, Trange, flp_df, x_scale, y_scale, barname='Temp(°C)'):
    
    fig, ax = plt.subplots(figsize=(10, 9))
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    plt.tick_params(labelsize=20)
    X, Y = np.meshgrid(np.arange(0, Temps.shape[0], reso),
                       np.arange(0, Temps.shape[1], reso), indexing='ij')
    levels = np.linspace(Trange[0], Trange[1], 50)
    cmap = cm.jet if Trange[0]>1 else "OrRd"
    cset = ax.contourf(X, Y, Temps, levels, cmap=cmap)   
    for i in range(flp_df.shape[0]):
        name = flp_df.iloc[i]['UnitName']
        x = flp_df.iloc[i]["X"]*x_scale
        y = flp_df.iloc[i]["Y"]*y_scale
        length = flp_df.iloc[i]["Length (m)"]*x_scale
        width = flp_df.iloc[i]["Width (m)"]*y_scale
        rect = plt.Rectangle((x, y), length, width, linestyle="--", linewidth=1, edgecolor='grey', facecolor='none')
        ax.add_patch(rect)
        ax.text(x + 0.5*length, y+0.5*width, name, ha='center', va='center', alpha=0.8, fontsize=26)

    Nx, Ny = Temps.shape
    plt.xlim([-1, Nx+1]); plt.ylim([-1, Ny+1])
    plt.xticks([]); plt.yticks([])
    position = fig.add_axes([0.95, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, orientation="vertical", pad=0, fraction=0, cax=position)
    cbar.set_label(barname,fontsize=24,x=0.5)
    cbar.set_ticks((np.linspace(levels[0], levels[-1], 5)*100//10/10).tolist())
    plt.show()

def Plot3d(data, z_reso=5):
    if data.max()<=data.min() or np.nan in data:
        return "The maximum of data must be larger than the minimum"
    # Create 3D plot
    scale = min(3,100/max(data.shape))
    data = F.interpolate(torch.Tensor(data).reshape(1,1,*data.shape), size=(int(scale*data.shape[0]),
                int(scale*data.shape[1]),data.shape[-1]), mode='trilinear', align_corners=False,).numpy().squeeze()
    Lx, Ly, Lz = data.shape
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    zcord = np.linspace(0,Lz-1,z_reso,dtype="int")
    X,Y,Z = np.meshgrid(np.arange(Lx), np.arange(Ly), zcord, indexing='ij')
    im = ax.scatter(X,Y,Z, c=data[...,zcord], cmap=cm.coolwarm, alpha=0.2)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1., 1.0, 1.0, 0.0))
    ax.set_xlim([0, Lx])
    ax.xaxis.set_ticks([Lx//4, Lx//4*3])
    ax.xaxis.set_ticklabels(['Left', 'Right'])
    ax.set_xlabel('X', labelpad=10)
    
    ax.set_ylim([0, Ly])
    ax.yaxis.set_ticks([Ly//4, Ly//4*3])
    ax.yaxis.set_ticklabels(['Front', 'Back'])
    ax.set_ylabel('Y', labelpad=10)
    
    ax.set_zlim([0, Lz])
    ax.zaxis.set_ticklabels([])
    ax.set_zlabel('Z')
    
    ax.grid(True)               # remove grid lines
    ax.view_init(elev=30, azim=-120)  # adjust view angle to show 3D structure
    
    # Set axis labels as unseen
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')

    cbar = plt.colorbar(im, shrink=0.8, aspect=16)
    cbar.set_label('Temp', fontsize=22)
    cbar.set_ticks(np.linspace(data.min(), data.max(), 5)*1000//10/100, fontsize=24)
    #cbar.set_ticklabels(['0', '0.25', '0.5', '0.75', '1'] )
    
    plt.show()
    
def Temp2d_plot(Temps, reso=1, rdbu=False, name=None):

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.tick_params(labelsize=20)
    X, Y = np.meshgrid(np.arange(0, Temps.shape[0], reso),
                       np.arange(0, Temps.shape[1], reso), indexing='ij')
    levels = np.linspace(Temps.min(), Temps.max(), 50)
    if rdbu:
        cset = ax.contourf(X, Y, Temps, levels, cmap="RdBu")    
    else:
        cset = ax.contourf(X, Y, Temps, levels, cmap=cm.jet)    
    ax.set_title('{} profile, Reso={:.0f}'.format(name, reso))
    Nx, Ny = Temps.shape
    plt.xlim([0, Nx]); plt.ylim([0, Ny])
    plt.xticks([0,Nx//2,Nx]); plt.yticks([Ny//2,Ny])
    position = fig.add_axes([0.99, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(cset, pad=0, fraction=0, cax=position)
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5).tolist())
    plt.show()
    
def FFT_plot(Values, kmax=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = '20'
    plt.rcParams['font.family'] = 'DejaVu Serif'

    plt.tick_params(labelsize=20)
    
    if isinstance(Values, torch.Tensor):
        Values = Values.cpu().detach().numpy()
    while len(Values.shape)>2:
        Values = np.sum(Values,0)
    
    X = np.fft.fftfreq(Values.shape[0], 1)
    Y = np.fft.fftfreq(Values.shape[1], 1)
    X, Y = np.meshgrid(X, Y)
    levels=np.linspace(min(0, Values.min()),max(1, Values.max()),20)
    cset1 = ax.contourf(X, Y, Values, levels, cmap=cm.jet)    
    ax.set_title('FFT Results')
    kmax = kmax if kmax is not None else X.max()
    plt.xlim(-kmax,kmax); plt.ylim(-kmax, kmax)
    cbar = fig.colorbar(cset1)
    cbar.set_ticks(np.linspace(levels[0], levels[-1], 5).tolist())
    plt.show()
    
    