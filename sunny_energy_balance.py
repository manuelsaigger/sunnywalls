#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:12:29 2021

@author: manuel
"""

import numpy as np
import matplotlib.pyplot as plt

def get_constants():
    
    constants = {'sigma': 5.67e-8,               # Stefan-Bolzman-Constant (W m-2 K-4)
                 
                 'limestone':{
                     'rho': 2750,                # density (kg m-3)
                     'cp' :840,                  # specific heat capacity (J kg-1 K-1)
                     'k': 1.3,                   # heat conductivity (W m-1 K-1)
                     
                     'epsilon': 1,               # emissivity
                     'alpha': 0.5,               # albedo
                     }
                 }
    
    return constants

def make_ebalance_day(ds_icon, az_sun_h, el_sun_h, el_ter_h, el_terrain, cragname, crags_meta):
    
    constants_rock = get_constants()
      
    
    # short wave stuff
    
    # inclination angle
    
    # hard coded stuff...
    dir_wall = crags_meta[cragname]['wall_dir']
    angle_wall = crags_meta[cragname]['wall_angl']
    rocktype = crags_meta[cragname]['rocktype']
    # dir_wall = 180
    # angle_wall = 80
    # rocktype = 'limestone'
    
    factor_incl = get_factor_sw_in_direct(dir_wall, angle_wall, az_sun_h, el_sun_h, el_ter_h)
    
    # sky view factor
    svf = get_skyviewfactor(el_terrain)
    
    alpha = constants_rock[rocktype]['alpha']
    
    swnet_eff = alpha * (ds_icon['sw_dir'].values * factor_incl + ds_icon['sw_dif'].values * svf)
    
    
    
    
    # surfac model stuff
    t_all = len(ds_icon.t_h.values)
    # t_all = 3
    
    x_all = 50
    dx = 0.2
    dt = 3600
    
    rho = constants_rock[rocktype]['rho']
    cp = constants_rock[rocktype]['cp']
    k = constants_rock[rocktype]['k']
    epsilon = constants_rock[rocktype]['epsilon']
    sigma = constants_rock['sigma']
    
    
    T_g = np.zeros((x_all, t_all))
    
    # ic: whole profile with t_g
    T_g[:, 0] = T_g[:, 0] + ds_icon.t_g.values[0]
    
    lwout_all = []
    shf_all = []
    dF_all = []
    
    for ii in range(1, t_all):
        # print('--------------------')
        
        # print(ii)
        
        swnet_ii = swnet_eff[ii-1]
        # print(swnet_ii)
                
        lwin_ii = ds_icon.lw_in.values[ii-1]
        # print(lwin_ii)
        
        lwout_ii = get_lwout(T_g[0, ii-1], epsilon, sigma)
        # print(lwout_ii)
        
        shf_ii = get_shf(T_g[0, ii-1], ds_icon['t_2m'].values[ii-1], ds_icon['ff_10m'].values[ii-1], method_k='const')
        # print(T_g[0, ii-1])
        # print(ds_icon['t_2m'].values[ii-1])
        # print(shf_ii)        
        
        dF = swnet_ii + lwin_ii - lwout_ii# + shf_ii
        # print(dF)
        
        lwout_all.append(lwout_ii)
        shf_all.append(shf_ii)
        dF_all.append(dF)
        
        T_g_ii = make_timestep_ghf(T_g[:, ii-1], dF, dt, dx, rho, cp, k)
        
        T_g[:, ii] = T_g_ii
        
        
        
        
    
    return T_g[0,:]
    
    # plt.figure()
    # plt.pcolormesh(T_g)
    # plt.colorbar()
    
    # return lwout_all, shf_all, dF_all, T_g
    
    
    
def get_shf(T_surf, T_air, ff, method_k='const'):
    
    rho = 1.25
    cp = 1005
    
    if method_k == 'const':
        kh = 5
    
    dz = 2
    dT = T_air - T_surf
    
    shf = rho * cp *  kh * (dT/dz)
    
    return shf
    
    
    
    
    
    
    





def get_factor_sw_in_direct(dir_wall, angle_wall, az_sun, el_sun, el_terrain):
    
    
    # gamma: vertical angle of sun inclination
    gamma = angle_wall + el_sun
    
    
    # delta: horizontal angle of sun inclination
    delta = az_sun - dir_wall
    
    
    factor_incl = np.sin(np.deg2rad(gamma)) * np.cos(np.deg2rad(delta))
    
    
    factor_incl[el_sun < el_terrain] = 0
        
    
    return factor_incl

def get_skyviewfactor(el_terrain):
    
    el_terrain = np.array(el_terrain)    
    
    svf = np.mean((90 - el_terrain) / 90)
    
    return svf


def get_lwout(T_surf, epsilon, sigma):
    
    lwout = epsilon * sigma * T_surf**4
    
    return lwout


def make_timestep_ghf(Tg_i, dF, dt, dx, rho, cp, k):
    
    Tg_ip = np.zeros(np.shape(Tg_i))
    
    Tg_ip[0] = Tg_i[0] + (dF * dt)/(rho*dx*cp) # upper BC
    Tg_ip[-1] = Tg_i[-1]                       # lower BC
    
    index_i = np.arange(1, len(Tg_i)-1)
    index_ip = np.arange(2, len(Tg_i))
    index_im = np.arange(0, len(Tg_i)-2)
    
    Tg_ip[index_i] = Tg_i[index_i] - dt * k/(rho*cp) * (Tg_i[index_ip] - Tg_i[index_im])/dx**2
    
    return Tg_ip
    

def test_diffusion():
    
    dF = 10
    dt = 3600
    dx = 0.2
    rho = 2750
    cp = 840
    k = 1.3
    
    xx = 5
    l_x = int(xx/dx)
    
    
    
    
    alpha = k/(rho*cp)
    print(dt*alpha/dx**2)
    
    t_end = 1200
    t = np.arange(0, t_end)
    dF_t = dF * np.sin(2*np.pi*(t/24))
    
    # plt.plot(dF_t)
    
    Tg = np.zeros((l_x, t_end))
    
    for ii in np.arange(1, t_end):
        # print(ii)
        Tg_ip = make_timestep_ghf(Tg[:, ii-1], dF_t[ii], dt, dx, rho, cp, k)
        Tg[:, ii] = Tg_ip

    plt.pcolormesh(Tg)    
    plt.colorbar()
    
    
    
    