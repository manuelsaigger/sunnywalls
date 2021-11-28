#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:26:20 2020

@author: manuel

module to calculate sun exposure of climbing walls (mainly in Tirol)

requirements:
    numpy, pandas, pvlib, json, PIL
    SRTM data

"""

# import stuff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pvlib
import json
from PIL import Image

import sunny_helpers
import sunny_energy_balance


def main(cragname, date='today', use_data_srtm=True, use_tiris_far=True, make_plot=True, include_icon_data=False, include_energy_balance=False):
    
    
    # Open crags metadata
    with open('crags_meta.json') as data_file:
        crags_meta = json.load(data_file)
        
    # check if cragname exists, if not, ask if new one should be created
    if cragname in list(crags_meta.keys()):
        print('found crag horizon data....')
    else:
        print('Unknown crag, available are: {}'.format(list(crags_meta.keys())))
        create_new = input('Quit (q) or create a new one (c)?')
        if create_new == 'q':
            return
        elif create_new == 'c':
            lat_new = float(input('Latitude: '))
            lon_new = float(input('Longitude: '))
            
            # check if new coordinates are within srtm data
            check_lat = ~np.logical_and(lat_new >= 39, lat_new < 49)
            check_lon = ~np.logical_and(lon_new >= 3, lon_new < 13)
            
            if check_lat or check_lon:
                raise ValueError('Coordinate are outside default SRTM-range, will not be able to calculate the horizon line, in a perfect wourld, no a option to download new SRTM-data')
                
            # include wall direction to shade?
            wall_dir = int(input('Wall direction? '))
            wall_angl = int(input('Wall angle? '))
            rocktype = input('Rocktype? ')
            tree_str = input('Treeshadow at the wall? (y/n)')
            
            if tree_str == 'y':
                tree = True
            elif tree_str == 'n':
                tree = False
                
            if use_data_srtm:
                make_horizon_tiris = False
                url_to_tiris = None
            else:
                make_horizon_tiris = True
                url_to_tiris = input('URL of Tiris horizon data: ') 
            
            
            sunny_helpers.make_new_crag(cragname, lat_new, lon_new, wall_dir=wall_dir, wall_angl=wall_angl, tree=tree, rocktype=rocktype,
                                        make_horizon_tiris=make_horizon_tiris, url_to_tiris=url_to_tiris)
        
            with open('crags_meta.json') as data_file:
                crags_meta = json.load(data_file)
                
            
    # configure date
    if date == 'today':
        date_pd = pd.to_datetime(date)
        date_str = str(date_pd)[:10].replace('-', '')
    
    elif date == 'tomorrow':
        date_pd = pd.to_datetime('today') + np.timedelta64(24, 'h')
        date_str = str(date_pd)[:10].replace('-', '')
        
    else:
        date_str = date
        
    # configure horizon type
    if use_data_srtm:
        horizon_type = 'srtm'
    else:
        if use_tiris_far:
            horizon_type = 'tiris_far'
        else:
            horizon_type = 'tiris'
    
        
    t_day, az_sun, el_sun, az_terrain, el_terrain, t_sunstart, t_sunend, el_terrain_time = sunny_helpers.get_sun_height_duration(crags_meta, cragname, date_str, horizon_type=horizon_type)
    
    
    print('Direct sun possible between {} and {} UTC at {}'.format(t_sunstart, t_sunend, cragname))
    
    if include_icon_data:
        ds_icon = sunny_helpers.get_data_icon(cragname, date_str, crags_meta, include_energy_balance=True)
        
        if include_energy_balance:
                t_day_h, az_sun_h, el_sun_h, az_terrain, el_terrain, t_sunstart_h, t_sunend_h, el_terrain_time_h = sunny_helpers.get_sun_height_duration(
                    crags_meta, cragname, date_str, horizon_type=horizon_type, dt=60)
                
                T_rock = sunny_energy_balance.make_ebalance_day(ds_icon, az_sun_h, el_sun_h, el_terrain_time_h, el_terrain, cragname, crags_meta)
                
    else: 
        T_rock = None
            
            
        
    print(T_rock)
    
    
    if make_plot:
        
        sunny_helpers.make_plot_basic(cragname, date_str, az_terrain, el_terrain, az_sun, el_sun)
        
        if include_icon_data:
            sunny_helpers.make_plot_full(cragname, ds_icon, date_str, t_day, el_sun, el_terrain_time, t_sunstart, t_sunend,
                                         include_energy_balance=include_energy_balance, T_rock=T_rock)
        
        
    
        
    
    
    
    
    
    
    
    



def when_comes_the_sun(cragname, date, make_plot=False, path_to_meta='./crags_meta.json', dt=5, use_data_srtm=True):
    """old main function for sun exposure calculation
    Workflow: - load horizon elevation for crag (pre-calculated, if not, option to create new one)
              - calculate sun elevation for whole day at crag position
              - when is sun > horizon?
              
    Input:    - cragname....... str, name of the crag to be calculated (case-sensitive, normally upper case at beginning)
              - date........... str, format 'yyyy-mm-dd' 
              - make_plot...... bool, plot sun elevation and horizon height?
              - dt............. int, temporal resolution for calculation in min, default 5 min
              - use_data_srtm.. bool, default True, use coarse SRTM-DEM or fine Tiris for horizon information?
    """
    
    ##########     
    # horizon stuff
    ##########
    
    # load horizon data    
    with open('crags_meta.json') as data_file:
        crags_meta = json.load(data_file)
        
    # check if cragname exists, if not, ask if new one should be created
    if cragname in list(crags_meta.keys()):
        print('found crag horizon data....')
    else:
        print('Unknown crag, available are: {}'.format(list(crags_meta.keys())))
        create_new = input('Quit (q) or create a new one (c)?')
        if create_new == 'q':
            return
        elif create_new == 'c':
            lat_new = float(input('Latitude: '))
            lon_new = float(input('Longitude: '))
            
            # check if new coordinates are within srtm data
            check_lat = ~np.logical_and(lat_new >= 39, lat_new < 49)
            check_lon = ~np.logical_and(lon_new >= 3, lon_new < 13)
            
            if check_lat or check_lon:
                raise ValueError('Coordinate are outside default SRTM-range, will not be able to calculate the horizon line, in a perfect wourld, no a option to download new SRTM-data')
                
            # include wall direction to shade?
            wall_dir = int(input('Wall direction? '))
            wall_angl = int(input('Wall angle? '))
            rocktype = input('Rocktype? ')
            tree_str = input('Treeshadow at the wall? (y/n)')
            
            if tree_str == 'y':
                tree = True
            elif tree_str == 'n':
                tree = False
            
            
            
            make_new_crag(cragname, lat_new, lon_new, 
                          wall_dir=wall_dir, wall_angl=wall_angl, rocktype=rocktype, tree=tree)
        
            with open('crags_meta.json') as data_file:
                crags_meta = json.load(data_file)
        
    # select SRTM data or tiris
    if use_data_srtm:
        az_terrain = np.array(crags_meta[cragname]['azimuth'])
        el_terrain = np.array(crags_meta[cragname]['horizon_elev'])
    
    else:
        infile_tiris = 'horizon_tiris_{}.txt'.format(cragname)
        data_horizon = np.loadtxt(infile_tiris, skiprows=3)
        az_terrain = data_horizon[:,0]
        el_terrain_far = data_horizon[:,2]
        el_terrain = data_horizon[:,3]
        
        
    #######  
    # sun stuff
    #######
    
    # create time array for whole day with frequency dt
    freq_sun = '{}min'.format(dt)
    t_day = pd.date_range(start=date, freq=freq_sun, periods=24*60/dt)
    
    # calculate sun elevation angle for every time step at crag position
    sun = pvlib.solarposition.get_solarposition(t_day, crags_meta[cragname]['lat'], crags_meta[cragname]['lon'])
    
    az_sun = np.array(sun.azimuth)
    el_sun = np.array(sun.elevation)
    
    
    #######
    # sun duration stuff
    #######
    
    # interpolate horizon elevation at constant azimut angles to azimut angles of times for comparison with sun elevation
    elev_interp_ter = np.interp(az_sun, az_terrain, el_terrain)

    # get times when sun elevation > horizon elevation
    t = sun.index
    mask_sun = el_sun > elev_interp_ter
    if any(mask_sun):
        t_sunstart = t[mask_sun][0]
        t_sunend = t[mask_sun][-1]
    
    if any(mask_sun):
        print('Direct sun possible between {} and {} UTC at {}'.format(t_sunstart, t_sunend, cragname))
    else:
        print('No direct sun at {} on {}'.format(cragname, date))
    
    #######
    # plotting stuff
    #######
    
    if make_plot:
        plt.plot(az_terrain, el_terrain, 'k-', label='terrain')
        plt.plot(sun.azimuth, sun.apparent_elevation, 'g-', label='sun')
        plt.ylim([0,90])
        plt.xlim([0, 360])
        plt.xticks(np.arange(0,361,90))
        plt.xlabel('Azimuth')
        plt.ylabel('Elevation')
        plt.legend(loc='lower left')
        plt.title('Horizon line and sun path for {} on {}'.format(cragname, date))

    



def make_new_crag(cragname, lat_crag, lon_crag, path_to_srtm='./srtm_39_03.tif', path_to_meta='./crags_meta.json', wall_dir=0, wall_angl=90, rocktype='limestone', tree=False):
    
    
    with open(path_to_meta) as data_file:
        crags_meta = json.load(data_file)
    go_on = 'y'
    if cragname in list(crags_meta.keys()):
        go_on = input('This crag already exists, do you want to go on anyway? (y/n)  ')
        
    if go_on == 'n':
       return
    elif go_on == 'y':
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
        if include_wall_dir:
            if wall_dir < 90:
                mask_wall = np.logical_and(az_all > wall_dir + 90, az_all < 360 - (90 - wall_dir))
            elif wall_dir > 270:
                mask_wall = np.logical_and(az_all < wall_dir - 90, az_all > wall_dir - 270)
            else:
                mask_wall = np.logical_or(az_all < wall_dir - 90, az_all > wall_dir + 90)
            
            elev_smooth[mask_wall] = 90
        
        
        
        ########
        # saving stuff
        ########
        
        # save as json file        
        az_all = list(az_all)            
        elev_smooth = list(elev_smooth)
            
        crags_meta[cragname] = {'lat':lat_crag, 'lon':lon_crag, 'azimuth':az_all, 'horizon_elev':elev_smooth}
        
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

            
            
            
    
