#!/usr/bin/env python
# coding: utf-8


import os
#os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
#os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
#os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

import sys
import argparse
import time
import math
import multiprocessing
from multiprocessing import Pool


import pandas as pd
import numpy as np

from scipy import interpolate
import matplotlib.pyplot as plt
from utils import concate_arrys, Local_layers, compare_res, memory_check

from Read_parser import Read_parser
from ChipStack import ChipStack
from GridManager import GridManager
from TSVManager import TSVManager
from CoarseSolver import CoarseSolver
from GridSolver import FVM_solver
from Visualize import Temp2d_plot, temp_plot, power_plot, error_plot

home_path = '../Mono3D/'
lcfFile = home_path + "Mono3D_lcf.csv"
ConfigFile = home_path + "Mono3D.config"
SimParamsFile = home_path + "SimParms.config",
gridSteadyFile = home_path + "results/trial.out"
thickness_layers, lcf_df, defaultConfig, SimParams, label_dict = Read_parser(lcfFile, ConfigFile, SimParamsFile)

val, unit = defaultConfig['Temperature']['init'].split()
if (unit == 'K' or unit == 'Kelvin'):
    initTemp = float(val)
elif (unit == 'C' or 'Celsius'):
    initTemp = float(val) + 273.15
    
if os.path.exists(SimParams._sections["TSV"]['tsv_path']):
    tsvmanager = TSVManager(defaultConfig, SimParams._sections["TSV"]["tsv_path"], 
                                        SimParams._sections["TSV"]["tsv_shape"])
    tsvmanager.add_TSV2Grids(chipstack, gridmanager)

SimParams._sections['Grid']['depth'] = 1
SimParams._sections['Grid']['granularity'] = 1 #128 #16

processes = 16
iter_leaks = 1
Local_layer_set = Local_layers(lcf_df.iloc[:-1], defaultConfig)
Power_idxs = lcf_df['PowerFile'].notna().values.nonzero()[0]

SimParams._sections['Grid']['rows'] = 50
SimParams._sections['Grid']['cols'] = 50
for layer_idx in Local_layer_set:
    if layer_idx in Power_idxs:
        lcf_df.loc[layer_idx, 'Clip_num_x'] = 4 #int(400/reso)
        lcf_df.loc[layer_idx, 'Clip_num_y'] = 4 #int(400/reso)
    #else:
    #    lcf_df.loc[layer_idx, 'Clip_num_x'] = int(200/reso)
    #    lcf_df.loc[layer_idx, 'Clip_num_y'] = int(200/reso)

print(lcf_df['Clip_num_x'])
start = time.time()

chipstack = ChipStack(lcf_df, defaultConfig, initTemp)
gridmanager = GridManager(SimParams._sections['Grid'], defaultConfig)
gridmanager.createGrids(chipstack)

grid_rows = int(SimParams._sections['Grid'].get('rows'))
grid_cols = int(SimParams._sections['Grid'].get('cols'))
grid_length, grid_width = gridmanager.grid_length, gridmanager.grid_width
granularity = int(SimParams._sections['Grid'].get('granularity'))
num_layers = chipstack.num_layers
solver_properties = {'rows': grid_rows*granularity, 'cols': grid_cols*granularity, 
                     'num_layers': num_layers, 
                     'OMP_NUM_THREADS': int(SimParams._sections["Solver"].get("number_of_core"))}
Solver = CoarseSolver(solver_properties)

x_pad = (np.arange(grid_cols*granularity+2)-0.5)/(granularity)
y_pad = (np.arange(grid_rows*granularity+2)-0.5)/(granularity)

T_old = None
Layers = chipstack.Layers_data
Layers.iloc[:-1].apply(lambda x: x.set_init_temps(initTemp))
Layers.iloc[:-1].apply(lambda x: x.calc_power())

for iter_num in range(iter_leaks):
    Layers.iloc[:-1].apply(lambda x: x.calc_power())
    Layers.apply(lambda x: gridmanager.calc_R_and_I(x, chipstack.num_layers))
    grid_temp = Solver.getTemperature(Layers)
    #print("Coarse Solve takes: ", time.time()-start)

    for layer_idx in Local_layer_set:
        t1 = time.time()
        layer = Layers[layer_idx]
        for i in range(grid_rows):
            for j in range(grid_cols):
                layer.Grid_array[i][j].set_init_temps(np.mean(grid_temp[j*granularity:(j+1)*granularity, 
                        i*granularity:(i+1)*granularity,layer.layer_num:layer.layer_num+layer.Rz.shape[0]+1]))

        z_pad = np.linspace(0, layer.Clip_num_z, SimParams._sections['Grid']['depth']+1)
        x_grids = (np.arange(grid_cols*layer.Clip_num_x)+0.5)/(layer.Clip_num_x)
        y_grids = (np.arange(grid_rows*layer.Clip_num_y)+0.5)/(layer.Clip_num_y)
        z_grids = (np.arange(layer.Clip_num_z)+0.5)
        Temp_pad = np.pad(grid_temp[:,:,layer.layer_num: layer.layer_num+layer.Rz.shape[0]+1],
                         ((1,1),(1,1),(0,0)), 'reflect')
        interp = interpolate.RegularGridInterpolator((x_pad, y_pad, z_pad), Temp_pad,
                                                     bounds_error=False, method="linear")
        x,y,z = np.meshgrid(x_grids, y_grids, np.array([0]), indexing="ij")
        T_bot = interp((x,y,z))
        x,y,z = np.meshgrid(x_grids, y_grids, np.array([layer.Clip_num_z]), indexing="ij")
        T_top = interp((x,y,z))

        g_x = np.empty((grid_rows, grid_cols+1), dtype=object)
        g_y = np.empty((grid_rows+1, grid_cols), dtype=object)
        for j in range(grid_cols+1):
            x,y,z = np.meshgrid(np.array([j]), y_grids, z_grids, indexing="ij")
            psi_temp = interp((x,y,z))[0,:,:]
            for i in range(grid_rows):
                g_x[i][j] = psi_temp[i*layer.Clip_num_y:(i+1)*layer.Clip_num_y]
        for i in range(grid_rows+1):
            x,y,z = np.meshgrid(x_grids, np.array([i]), z_grids, indexing="ij")
            psi_temp = interp((x,y,z))[:,0,:]
            for j in range(grid_cols):
                g_y[i][j] = psi_temp[j*layer.Clip_num_x:(j+1)*layer.Clip_num_x]
                if i>0:
                    x_min, x_max = j*layer.Clip_num_x, (j+1)*layer.Clip_num_x
                    y_min, y_max = (i-1)*layer.Clip_num_y, i*layer.Clip_num_y
                    layer.Grid_array[i-1][j].set_boundary([g_x[i-1][j], g_x[i-1][j+1], 
                                                        g_y[i-1][j], g_y[i][j], 
                                                        T_bot[x_min:x_max, y_min:y_max, 0], 
                                                        T_top[x_min:x_max, y_min:y_max, 0]], BVtype=1)    
        #print("Prepare takes: ", time.time()-t1)
        t2 = time.time()        
        pool = Pool(processes=processes)
        result=[]
        for i in range(grid_rows):
            for j in range(grid_cols):
                result.append(pool.apply_async(FVM_solver, args=(
                                    layer.Grid_array[i][j].power, layer.Grid_array[i][j].calc_kappa_init(), 
                                    layer.Grid_array[i][j].BV_psi, layer.Grid_array[i][j].dxyz)))      
        pool.close()
        pool.join()
        #print("Fine solve takes: ", time.time()-t2)
        for i in range(grid_rows):
            for j in range(grid_cols):
                layer.Grid_array[i][j].psi_2_temp(result[int(i*grid_cols+j)].get())
    memory_check()
print(time.time()-start)
memory_check()