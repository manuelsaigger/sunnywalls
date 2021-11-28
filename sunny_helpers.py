#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:48:51 2021

@author: manuel


################
Helping functions for sunny_walls

"""


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import pvlib
import json
from PIL import Image

import os

def make_plot_full(cragname, ds_icon, date_str, t_day, el_sun, el_terrain_time, t_sunstart, t_sunend, include_energy_balance=False, T_rock=None):
    
    
    # time array hourly
    t_day_h = pd.date_range(start=t_day[0], end=t_day[-1], freq='h')
    
    # time array hourly plus 1 for pcolor
    t_day_hp = pd.date_range(start=t_day[0], end=t_day[-1]+np.timedelta64(1,'h'), freq='h')
    t_day_hp = np.reshape(t_day_hp, (1, len(t_day_hp)))
    
    tlim = [t_day[0], t_day[-1]]
    
    # position vector for pcolor
    yy = np.vstack((np.ones(np.shape(t_day_hp)), np.zeros(np.shape(t_day_hp))))
    
    
    # configure axes layout
    fig = plt.figure(figsize=(16,10))
    
    dx_l = .01
    dx_sym = .07
    dx = .89
    dx_cb = .2
    dxx_cb = .03
    
    dy_u = .05
    dy_cb = .02
    dy_sm = .05
    dy_bg = .2
    dy_m = .02
    
    ax_cbcl = fig.add_axes([dx_l+dx_sym, dy_u, dx_cb, dy_cb])
    ax_cbrr = fig.add_axes([dx_l+dx_sym+dx_cb+dxx_cb, dy_u, dx_cb, dy_cb])
    ax_cbff = fig.add_axes([dx_l+dx_sym+2*(dx_cb+dxx_cb), dy_u, dx_cb, dy_cb])
    ax_cbtr = fig.add_axes([dx_l+dx_sym+3*(dx_cb+dxx_cb), dy_u, dx_cb, dy_cb])
    
    dy_tmp = dy_u + dy_cb + 0.07
    ax_tr = fig.add_axes([dx_l+dx_sym, dy_tmp, dx, dy_sm])
    
    dy_tmp = dy_tmp + dy_sm + dy_m
    ax_rh = fig.add_axes([dx_l+dx_sym, dy_tmp, dx, dy_bg])
    ax_temp = ax_rh.twinx()
    
    dy_tmp = dy_tmp + dy_bg + dy_m
    ax_wind = fig.add_axes([dx_l+dx_sym, dy_tmp, dx, dy_sm])
    
    dy_tmp = dy_tmp + dy_sm + dy_m
    ax_rr = fig.add_axes([dx_l+dx_sym, dy_tmp, dx, dy_sm])
    
    dy_tmp = dy_tmp + dy_sm + dy_m
    ax_cl = fig.add_axes([dx_l+dx_sym, dy_tmp, dx, dy_sm])
    ax_cm = fig.add_axes([dx_l+dx_sym, dy_tmp+dy_sm, dx, dy_sm])
    ax_ch = fig.add_axes([dx_l+dx_sym, dy_tmp+2*dy_sm, dx, dy_sm])
    
    dy_tmp = dy_tmp + 3*dy_sm + dy_m
    ax_sun = fig.add_axes([dx_l+dx_sym, dy_tmp, dx, dy_bg])
    
    
    # configure colormaps
    values_cl = np.arange(0, 101, 20)
    values_rr = np.array([0, 0.01, 0.1, 1, 10, 100])
    values_ff = np.array([0, 1, 5, 10, 20, 50])
    values_tr = np.arange(-20, 21, 2)
    
    cmap_rr = mpl.colors.ListedColormap(["#FFFFFF","#C1E1EF","#8BAECF","#6470A8","#512082"])
    cmap_cl = mpl.colors.ListedColormap(["#F9F9F9","#DADADA","#AEAEAE","#7E7E7E","#4C4C4C"])
    cmap_ff = mpl.colors.ListedColormap(["#FFE9D0","#BFD48F","#44B09E","#5178A0","#73386B"])
    
    # cmap_cl = mpl.cm.get_cmap('Greys')
    # cmap_rr = mpl.cm.get_cmap('BuPu')
    # cmap_ff = mpl.cm.get_cmap('YlGnBu')
    cmap_tr = mpl.cm.get_cmap('RdBu_r')

    norm_cl = mpl.colors.BoundaryNorm(values_cl, cmap_cl.N)      
    norm_rr = mpl.colors.BoundaryNorm(values_rr, cmap_rr.N)   
    norm_ff = mpl.colors.BoundaryNorm(values_ff, cmap_ff.N)   
    norm_tr = mpl.colors.BoundaryNorm(values_tr, cmap_tr.N)   
    
    
    
    clcl = np.nan * np.zeros(np.shape(t_day_h))
    clcl_tmp= ds_icon.clcl.values
    clcl[0:len(clcl_tmp)] = clcl_tmp
    clcl = np.reshape(clcl, (1, len(clcl)))
    
    cf_cl = ax_cl.pcolormesh(t_day_hp, yy, clcl, cmap=cmap_cl, norm=norm_cl)
    cb_cl = plt.colorbar(cf_cl, cax=ax_cbcl, orientation='horizontal')
    cb_cl.set_label('Wolkenbedeckung (%)')
    
    
    clcm = np.nan * np.zeros(np.shape(t_day_h))
    clcm_tmp= ds_icon.clcm.values
    clcm[0:len(clcm_tmp)] = clcm_tmp
    clcm = np.reshape(clcm, (1, len(clcm)))
    
    ax_cm.pcolormesh(t_day_hp, yy, clcm, cmap=cmap_cl, norm=norm_cl)
    
    clch = np.nan * np.zeros(np.shape(t_day_h))
    clch_tmp= ds_icon.clch.values
    clch[0:len(clch_tmp)] = clch_tmp
    clch = np.reshape(clch, (1, len(clch)))
    
    ax_ch.pcolormesh(t_day_hp, yy, clch, cmap=cmap_cl, norm=norm_cl)
    
    ax_cl.set_xlim(tlim)
    ax_cm.set_xlim(tlim)
    ax_ch.set_xlim(tlim)
    ax_cl.set_xticklabels([])
    ax_cm.set_xticklabels([])
    ax_ch.set_xticklabels([])
    ax_cl.set_yticks([])
    ax_cm.set_yticks([])
    ax_ch.set_yticks([])
    
    ax_cl.text(-0.01, 0.5, 'T', verticalalignment='center', horizontalalignment='left', transform=ax_cl.transAxes)
    ax_cm.text(-0.01, 0.5, 'M', verticalalignment='center', horizontalalignment='left', transform=ax_cm.transAxes)
    ax_ch.text(-0.01, 0.5, 'H', verticalalignment='center', horizontalalignment='left', transform=ax_ch.transAxes)
    
    ax_cl.set_xticks(t_day_h)
    ax_cm.set_xticks(t_day_h)
    ax_ch.set_xticks(t_day_h)
    
    rr = np.nan * np.zeros(np.shape(t_day_h))
    rr_tmp = ds_icon.precip_intens.values
    rr[0:len(rr_tmp)] = rr_tmp
    rr = np.reshape(rr, (1, len(rr)))
    
    cf_rr = ax_rr.pcolormesh(t_day_hp, yy, rr, cmap=cmap_rr, norm=norm_rr)
    cb_rr = plt.colorbar(cf_rr, cax=ax_cbrr, orientation='horizontal', ticks=[0, 0.01, 0.1, 1, 10])
    cb_rr.set_label('Niederschlag (mm h$^{-1}$)')    
    cb_rr.set_ticklabels(['0', '0.01', '0.1', '1', '10'])
    
    ax_rr.set_xlim(tlim)
    ax_rr.set_xticklabels([])
    ax_rr.set_yticks([])

    ax_rr.set_xticks(t_day_h)

    
    ff = np.nan * np.zeros(np.shape(t_day_h))
    ff_tmp = ds_icon.ff_10m.values
    ff[0:len(ff_tmp)] = ff_tmp
    ff = np.reshape(ff, (1, len(ff)))
    
    dd = np.nan * np.zeros(np.shape(t_day_h))
    dd_tmp = ds_icon.dd_10m.values
    dd[0:len(dd_tmp)] = dd_tmp
    dd = np.reshape(dd, (1, len(dd)))
    
    ff1 = 0.5*np.ones(np.shape(ff))
    uu1, vv1 = ddff2uv(dd, ff1)
    
    t_day_h_quiver = t_day_h + np.timedelta64(30, 'm',width=.1)
    y_quiver = 0.5 + np.zeros(np.shape(ff))
    
    cf_ff = ax_wind.pcolormesh(t_day_hp, yy, ff, cmap=cmap_ff, norm=norm_ff)
    ax_wind.quiver(t_day_h_quiver, y_quiver, uu1, vv1, pivot='middle', scale=20, width=0.002)
    
    cb_ff = plt.colorbar(cf_ff, cax=ax_cbff, orientation='horizontal', ticks=[0, 1, 5, 10, 20])
    cb_ff.set_label('Wind (m s$^{-1}$)')
    
    
    ax_wind.set_xlim(tlim)
    ax_wind.set_xticklabels([])
    ax_wind.set_yticks([])
    ax_wind.set_xticks(t_day_h)
    
    
    rh = np.nan * np.zeros(np.shape(t_day_h))
    rh_tmp = ds_icon.relhum_2m.values
    rh[0:len(rh_tmp)] = rh_tmp
    
    ax_rh.fill_between(t_day_h, rh, color='lightgreen')
    
    t2 = np.nan * np.zeros(np.shape(t_day_h))
    t2_tmp = ds_icon.t_2m.values
    t2[0:len(t2_tmp)] = t2_tmp - 273.15
    
    ax_temp.plot(t_day_h, t2, 'r-', linewidth=2)
    
    ax_rh.set_xlim(tlim)
    ax_temp.set_xlim(tlim)
    ax_rh.set_xticklabels([])
    ax_temp.set_xticklabels([])
    
    ax_rh.set_ylim([0, 100])
    ax_rh.set_yticks([])
    
    for yi in np.arange(-20, 40, 5):
        ax_temp.axhline(y=yi, color='lightgrey', linewidth=.5)
    ax_temp.axhline(y=0, color='lightgrey', linewidth=1)
    
    ax_temp.set_yticks(np.arange(-20, 40, 5))
    ax_temp.set_ylim([np.nanmin(t2)-4, np.nanmax(t2)+4])
    ax_temp.yaxis.set_label_position("left")
    ax_temp.yaxis.tick_left()
    ax_temp.set_xticks(t_day_h)
    ax_rh.set_xticks(t_day_h)
    
    
    
    
    
    
    ax_sun.fill_between(t_day, el_sun, 0, color='darkorange')
    ax_sun.fill_between(t_day, el_terrain_time, 0, color='darkgrey')
    ax_sun.plot(t_day, el_sun, '--', color='darkorange', linewidth=2)
    
    ax_sun.set_ylim([0, 90])
    ax_sun.set_xlim(tlim)
    ax_sun.set_xticklabels([])
    ax_sun.set_yticks([])
    ax_sun.set_xticks(t_day_h)
    
    
    if include_energy_balance:
        T_rock = np.reshape(T_rock, (1, len(T_rock))) - 273.15
    else:
        T_rock = np.nan*ff
    print(np.shape(ff))        
    print(np.shape(T_rock))
    cf_tr = ax_tr.pcolormesh(t_day_hp, yy, T_rock, cmap=cmap_tr, norm=norm_tr)
    cb_tr = plt.colorbar(cf_tr, cax=ax_cbtr, orientation='horizontal')
    cb_tr.set_label('Felstemperatur (°C)')
    
    ax_tr.set_yticks([])
    if not include_energy_balance:
        ax_tr.text(0.5, 0.5, 'Hier könnte Ihre Felstemperatur stehen!', verticalalignment='center', horizontalalignment='center',
               transform=ax_tr.transAxes)
    
    
    ax_tr.set_xlim(tlim)
    
    ax_tr.set_xticks(t_day_h)
    # tticks = []
    ax_tr.set_xticklabels(list(np.arange(0, 24)))
    ax_tr.set_xlabel('Uhrzeit (UTC)')
    
    
    wd_sym = .06
    
    img_sun = plt.imread('sonne.png')
    ax_sym_sun = fig.add_axes([dx_l, .79, wd_sym, wd_sym])
    ax_sym_sun.imshow(img_sun)
    ax_sym_sun.axis('off')
    
    img_cloud = plt.imread('wolke.png')
    ax_sym_cloud = fig.add_axes([dx_l, .61, wd_sym, wd_sym])
    ax_sym_cloud.imshow(img_cloud)
    ax_sym_cloud.axis('off')
    
    img_rain = plt.imread('regen.png')
    ax_sym_rain = fig.add_axes([dx_l, .5, wd_sym, wd_sym])
    ax_sym_rain.imshow(img_rain)
    ax_sym_rain.axis('off')
    
    img_wind = plt.imread('wind.png')
    ax_sym_wind = fig.add_axes([dx_l, .42, wd_sym, wd_sym])
    ax_sym_wind.imshow(img_wind)
    ax_sym_wind.axis('off')
    
    img_temp = plt.imread('temperatur.png')
    ax_sym_temp = fig.add_axes([dx_l, .3, wd_sym, wd_sym])
    ax_sym_temp.imshow(img_temp)
    ax_sym_temp.axis('off')
    
    img_moist = plt.imread('feuchte.png')
    ax_sym_moist = fig.add_axes([dx_l, .24, wd_sym, wd_sym])
    ax_sym_moist.imshow(img_moist)
    ax_sym_moist.axis('off')
    
    # img_cl = plt.imread('wolke.png')
    # img_rain = plt.imread('regen.png')
    # img_wind = plt.imread('wind.png')
    # img_temp = plt.imread('thermometer.jpg')
    
    
    # ax_clsym.imshow(img_cl)
    # ax_rrsym.imshow(img_rain)
    # ax_windsym.imshow(img_wind)
    # ax_tempsym.imshow(img_temp)
    
    hh_start = pd.to_datetime(t_sunstart).hour
    if hh_start < 10:
        hh_start_s = '0{}'.format(hh_start)
    else:
        hh_start_s = str(hh_start)
    mm_start = pd.to_datetime(t_sunstart).minute
    if mm_start < 10:
        mm_start_s = '0{}'.format(mm_start)
    else:
        mm_start_s = str(mm_start)
    hh_end = pd.to_datetime(t_sunend).hour
    if hh_end < 10:
        hh_end_s = '0{}'.format(hh_end)
    else:
        hh_end_s = str(hh_end)
    mm_end = pd.to_datetime(t_sunend).minute
    if mm_end < 10:
        mm_end_s = '0{}'.format(mm_end)
    else:
        mm_end_s = str(mm_end)
    ax_sun.text(0.01, 0.9, 'Direkte Sonne möglich von {}:{} bis {}:{} UTC'.format(hh_start_s, mm_start_s, hh_end_s, mm_end_s),
                horizontalalignment='left', verticalalignment='center', transform=ax_sun.transAxes, fontsize=14, backgroundcolor='white')
    
    day = date_str[6:8]
    month = date_str[4:6]
    year = date_str[:4]
    
    cragname_title = cragname.replace('_', ' ')
    ax_sun.set_title('{}, {}.{}.{}'.format(cragname_title, day, month, year), fontsize=14, fontweight='bold')
    
    fig.savefig('forecast_{}_{}.png'.format(cragname, date_str), dpi=300)
    
    




def get_data_icon(cragname, date_str, crags_meta, start_h=0, t_all=24, include_energy_balance=False):
    
    
    lat_crag = crags_meta[cragname]['lat']
    lon_crag = crags_meta[cragname]['lon']
    
    vars_icon = ['clcl', 'clcm', 'clch', 'tot_prec', 't_2m', 'relhum_2m', 'u_10m', 'v_10m']
    vars_icon_ds = ['ccl', 'ccl', 'ccl', 'tp', 't2m', 'r2', 'u10', 'v10']
    
    if include_energy_balance:
        vars_icon = vars_icon + ['aswdifd_s', 'aswdir_s', 't_g', 'athb_s']
        vars_icon_ds = vars_icon_ds + ['ASWDIFD_S', 'ASWDIR_S', 't', 'nlwrf']
    t_h = np.arange(start_h, start_h+t_all)
    
    
    data_icon = xr.Dataset(coords={'t_h':t_h})
    
    
    
    
    
    for var_icon, var_icon_ds in zip(vars_icon, vars_icon_ds):
        
        data_var_point = []
        
        if var_icon in ['clct', 'clcl', 'clcm', 'clch', 'tot_prec', 'aswdifd_s', 'aswdir_s']:
            use_exact_point = False
        else:
            use_exact_point = True
        
        for ii, ti in enumerate(t_h):
            filename_icon_tmp, url_icon_tmp = create_names_icon(var_icon, date_str, start_h, ti)
            
            download_data_icon(url_icon_tmp, filename_icon_tmp)
            
            d_icon_tmp_ti = get_data_point_icon(filename_icon_tmp, var_icon, var_icon_ds, lat_crag, lon_crag, use_exact_point=use_exact_point)
            
            # if var_icon in ['rain_con', 'rain_gsp', 'snow_con', 'snow_gsp']:
            #     if ii > 0:
            #         d_icon_tmp_ti = d_icon_tmp_ti - data_var_point[-1]
                        
            data_var_point.append(d_icon_tmp_ti)
            
            
        
        data_var_point = np.array(data_var_point)
        
        data_icon[var_icon] = ('t_h', data_var_point)
        
    
    if 'tot_prec' in vars_icon:
        precip_intens = np.diff(data_icon.tot_prec.values)
        precip_intens = np.append(precip_intens, np.nan)
        
        data_icon['precip_intens'] = ('t_h', precip_intens)
    
    if 'u_10m' in vars_icon:
        dd, ff = uv2ddff(data_icon.u_10m.values, data_icon.v_10m.values)
        data_icon['dd_10m'] = ('t_h', dd)
        data_icon['ff_10m'] = ('t_h', ff)
        
    if 'aswdifd_s' in vars_icon:
        dif_avg = data_icon['aswdifd_s'].values
        dir_avg = data_icon['aswdir_s'].values
        lw_avg = data_icon['athb_s'].values
        
        dif_inst = np.zeros(np.shape(dif_avg))
        dir_inst = np.zeros(np.shape(dif_avg))
        lw_inst = np.zeros(np.shape(dif_avg))
        
        for ii in range(1, len(dif_avg)):
            dif_inst[ii] = dif_avg[ii] * ii - dif_avg[ii-1] * (ii-1)
            dir_inst[ii] = dir_avg[ii] * ii - dir_avg[ii-1] * (ii-1)
            lw_inst[ii] = lw_avg[ii] * ii - lw_avg[ii-1] * (ii-1)
            
        data_icon['sw_dif'] = ('t_h', dif_inst)
        data_icon['sw_dir'] = ('t_h', dir_inst)
        data_icon['lw_net'] = ('t_h', lw_inst)
        
        sigma = 5.67e-8
        data_icon['lw_out'] = sigma * data_icon['t_g']**4
        data_icon['lw_in'] = data_icon['lw_net'] + data_icon['lw_out']
    
    
    
    return data_icon


def get_data_point_icon(filename_icon, var_icon, var_icon_ds, lat_i, lon_i, use_exact_point=True, dlatlon=0.1):
    
    ds = xr.open_dataset(filename_icon, engine='cfgrib')
    
    # print(ds)

    if use_exact_point:
        d_point = float(ds.sel(latitude=lat_i, method='nearest').sel(longitude=lon_i, method='nearest')[var_icon_ds].mean().values)
    else:
        d_point = float(ds.sel(latitude=slice(lat_i-dlatlon, lat_i+dlatlon)).sel(longitude=slice(lon_i-dlatlon, lon_i+dlatlon))[var_icon_ds].mean().values)

    return d_point


def download_data_icon(url_icon, filename_icon):
    
    # check if file already exists
    dir_list = os.listdir()
    if filename_icon in dir_list:
        print('ICON data already exists..')
        
    else:
        print('ICON data is downloaded..')
        print(url_icon)
    
        command_wget = 'wget ' + url_icon
    
        try:
            os.system(command_wget)
        
        except:
            raise RuntimeError('Error while downloading ICON data')
    
        filename_icon_bz2 = filename_icon + '.bz2'
        command_bz = 'bzip2 -d ' + filename_icon_bz2
  
        os.system(command_bz)
    


def create_names_icon(var_icon, date, start_h, t_step):
    
    if t_step < 10:
        t_step_str = '0' + str(t_step)
    else:
        t_step_str = str(t_step)
    
    if start_h < 10:
        start_h_str = '0' + str(start_h)
    else:
        start_h_str = str(start_h)
    
    filename_icon = 'icon-d2_germany_regular-lat-lon_single-level_{}{}_0{}_2d_{}.grib2'.format(date, start_h_str, t_step_str, var_icon)
    
    
    url_icon = 'https://opendata.dwd.de/weather/nwp/icon-d2/grib/{}/{}/{}.bz2'.format(start_h_str, var_icon, filename_icon)
    
    return filename_icon, url_icon




def make_plot_basic(cragname, date, az_terrain, el_terrain, az_sun, el_sun):
    
    
    plt.figure(figsize=(8,5))
    
    plt.plot(az_terrain, el_terrain, 'k-', label='terrain')
    plt.plot(az_sun, el_sun, 'g-', label='sun')
    
    plt.ylim([0,90])
    plt.xlim([0, 360])
    plt.xticks(np.arange(0,361,90))
    plt.xlabel('Azimuth')
    plt.ylabel('Elevation')
    plt.legend(loc='lower left')
    plt.title('Horizon line and sun path for {} on {}'.format(cragname, date))


def get_sun_height_duration(crags_meta, cragname, date, dt=5, path_to_meta='./crags_meta.json', horizon_type='srtm'):
    
    
    if horizon_type == 'srtm':
        el_terrain = crags_meta[cragname]['horizon_elev']
    elif horizon_type == 'tiris_far':
        el_terrain = crags_meta[cragname]['horizon_tiris_far']
    elif horizon_type == 'tiris':
        el_terrain = crags_meta[cragname]['horizon_tiris']
    
    az_terrain = crags_meta[cragname]['azimuth']
    
    
    # create time array for whole day with frequency dt
    freq_sun = '{}min'.format(dt)
    t_day = pd.date_range(start=date, freq=freq_sun, periods=24*60/dt)
    
    # calculate sun elevation angle for every time step at crag position
    sun = pvlib.solarposition.get_solarposition(t_day, crags_meta[cragname]['lat'], crags_meta[cragname]['lon'])
    
    az_sun = np.array(sun.azimuth)
    el_sun = np.array(sun.elevation)
    
    elev_interp_ter = np.interp(az_sun, az_terrain, el_terrain)

    # get times when sun elevation > horizon elevation
    t = sun.index
    mask_sun = el_sun > elev_interp_ter
    if any(mask_sun):
        t_sunstart = t[mask_sun][0]
        t_sunend = t[mask_sun][-1]
    
    return t_day, az_sun, el_sun, az_terrain, el_terrain, t_sunstart, t_sunend, elev_interp_ter


def add_tiris_to_meta(cragname, path_to_meta='./crags_meta.json'):
    
    with open(path_to_meta) as data_file:
        crags_meta = json.load(data_file)
        
    infile_tiris = 'horizon_tiris_{}.txt'.format(cragname)
    data_horizon = np.loadtxt(infile_tiris, skiprows=3)
    az_terrain = data_horizon[:,0]
    el_tiris_far = data_horizon[:,2]
    el_tiris = data_horizon[:,3]
    
    el_tiris_far = list(el_tiris_far)
    el_tiris = list(el_tiris)
    
    crags_meta[cragname]['horizon_tiris'] = el_tiris
    crags_meta[cragname]['horizon_tiris_far'] = el_tiris_far
        
        
        # return crags_meta
                  
    json_file = json.dumps(crags_meta)
    
    f = open('crags_meta.json', 'w')
    f.write(json_file)
    f.close()
    

def make_new_crag(cragname, lat_crag, lon_crag, path_to_srtm='./srtm_39_03.tif', path_to_meta='./crags_meta.json',
                  wall_dir=0, wall_angl=0, rocktype='limestone', tree=False, make_horizon_tiris=False, url_to_tiris=None):
    
    """create horizon line for new crag, default: SRTM horizon, optional wall direction, tiris horizon""" 
    
    
    with open(path_to_meta) as data_file:
        crags_meta = json.load(data_file)
        
    go_on = 'y'
    if cragname in list(crags_meta.keys()):
        go_on = input('This crag already exists, do you want to go on anyway? (y/n)  ')
        
    if go_on == 'n':
       return
    elif go_on == 'y':
        
        ###############
        # SRTM Stuff
        ###############
        
        print('loading DEM...')

        dem = Image.open(path_to_srtm)
            
        dem_np = np.fliplr(np.transpose(np.array(dem)))
            
        # some hard-coded stuff, should be replaced...
        lat_0_dem = 45
        lat_1_dem = 50

        lon_0_dem = 10
        lon_1_dem = 15

        number_of_points_lat = dem_np.shape[1]
        number_of_points_lon = dem_np.shape[0]

        lats_dem = np.linspace(lat_0_dem, lat_1_dem, number_of_points_lat)
        lons_dem = np.linspace(lon_0_dem, lon_1_dem, number_of_points_lon)
            
        lats, lons = np.meshgrid(lats_dem, lons_dem)
            
        ### calculate horizon line
        print('Calculating horizon line...')
        
        # get DEM height at point closest to to crag
        dlat_p = lats - lat_crag
        dlon_p = lons - lon_crag
        mask_point = np.logical_and(np.abs(dlat_p) == np.min(np.abs(dlat_p)), np.abs(dlon_p) == np.min(np.abs(dlon_p))) 
        h_p = float(dem_np[mask_point])
        
        # get height difference between every point of DEM and point of crag
        dh = dem_np - h_p
        
        # calculate azimut and distance between every point of DEM and crag
        az, d = latlon2dist(lat_crag, lon_crag, lats, lons)
        az -= 180
        az[az < 0] = az[az < 0] + 360
        
        # some trigonometry between the points to get elevation angle between every point of DEM and crag
        elev = np.arctan(dh/(d*1000))
            
        
        # check for highest elevation angle within sectors of azimut angles
        daz = 1.0
        az_all = np.arange(0, 360, daz)
        # az_all = float(az_all)
        elev_all = []
        for az_tmp in az_all:
            mask_tmp = np.logical_and(az >= az_tmp - daz/2, az < az_tmp + daz/2)
    
            elev_max_tmp = np.nanmax(np.where(mask_tmp, elev, np.nan))
    
            elev_all.append(elev_max_tmp)
    
        elev_all = np.array(elev_all)
        elev_all = np.rad2deg(elev_all)
            
        # do some median smooting of elevation angles
        elev_smooth = np.zeros(np.shape(elev_all))
        width_window = 10
        
        for ii in range(width_window, len(elev_smooth)-width_window):
            elev_smooth[ii] = np.median(elev_all[ii-width_window:ii+width_window])

        for ii in range(0, width_window):
            elev_smooth[ii] = np.median(elev_all[ii:ii+width_window])
    
        for ii in range(len(elev_smooth)-width_window, len(elev_smooth)):
            elev_smooth[ii] = np.median(elev_all[ii-width_window:ii])
                
            
        # if wall direction is included: set whole half dome to elev=90    
        # if include_wall_dir:
        if wall_dir < 90:
            mask_wall = np.logical_and(az_all > wall_dir + 90, az_all < 360 - (90 - wall_dir))
        elif wall_dir > 270:
            mask_wall = np.logical_and(az_all < wall_dir - 90, az_all > wall_dir - 270)
        else:
            mask_wall = np.logical_or(az_all < wall_dir - 90, az_all > wall_dir + 90)
            
        elev_smooth[mask_wall] = 90
        
        
        if make_horizon_tiris:
            
            command_wget = 'wget ' + url_to_tiris + ' -O tiris_tmp.txt'
            try:
                os.system(command_wget)
            except :
                raise RuntimeError('Tiris url did not work!')
                
            data_horizon = np.loadtxt('tiris_tmp.txt', skiprows=3)
            az_tiris = data_horizon[:,0]
            el_tiris_far = data_horizon[:,2]
            el_tiris = data_horizon[:,3]
            
            el_tiris_far = list(el_tiris_far)
            el_tiris = list(el_tiris)
            
            
        ########
        # saving stuff
        ########
        
        # save as json file        
        az_all = list(az_all)            
        elev_smooth = list(elev_smooth)
            
        crags_meta[cragname] = {'lat':lat_crag, 'lon':lon_crag, 'azimuth':az_all, 'horizon_elev':elev_smooth, 
                                'wall_dir':wall_dir, 'wall_angl':wall_angl, 'rocktype':rocktype, 'tree':tree}
        
        if make_horizon_tiris:
            crags_meta[cragname]['horizon_tiris'] = el_tiris
            crags_meta[cragname]['horizon_tiris_far'] = el_tiris_far
        
        
        # return crags_meta
                  
        print('Saving...')
        json_file = json.dumps(crags_meta)
    
        f = open('crags_meta.json', 'w')
        f.write(json_file)
        f.close()
            
    else:
        raise RuntimeError('Only y or n you dummy!!')



def latlon2dist(lat1, lon1, lat2, lon2):
    """calculates the distance and course between two coordinates in decimal-degree-format, 
    approximations only valid for rather short distances 
    lat/lon1: first coordinates pair (if movement: original position)
    lat/lon2: second coordinates pair (if movement: new position)"""
    
    re = 6730
    
    d_lat = lat2 - lat1
    d_y = 2 * re * np.pi * (d_lat / 360)
    
    d_lon = lon2 - lon1
    r_eff = np.cos(np.deg2rad(lat1)) * re
    d_x = 2 * r_eff * np.pi * (d_lon / 360)
    
    (course, dist) = uv2ddff(d_x, d_y)
    
    return course, dist
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def ddff2uv(dd, ff):
    """calculates wind components u and v out of wind direction dd and wind speed ff"""
    # convert dd in radian
    dd_rad = np.deg2rad(dd)
    
    # calculate angle between ff and u
    alfa = (3 * np.pi / 2) - dd_rad
    
    # calculate u and v
    (u, v) = pol2cart(ff, alfa)
    
    return u, v


def uv2ddff(u, v):
    """calculates wind direction and wind speed out of wind components u and v"""
    # transform coordinates 
    (ff, dd_rad) = cart2pol(u,v)
    
    # dd in degree
    dd = np.rad2deg(dd_rad)
    
    # dd in meteorological sense
    dd = 270 - dd
    
    # filter values < 0 and > 360
    dd = np.array(dd)
    
    dd[dd < 0] = dd[dd < 0] + 360
    dd[dd > 360] = dd[dd > 360] - 360
    
    return dd, ff 


