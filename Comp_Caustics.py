#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 22:39:37 2025

@author: kieren
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from surf2stl import write

def image_read(file_path, shape):
    img = cv.imread(file_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img =  cv.resize(img, shape, interpolation=cv.INTER_NEAREST)
    return img


def create_xy_grids(i_n, j_n):
    x_grid = np.tile(np.arange(0, i_n+1), (j_n+1, 1))
    y_grid = np.tile(np.arange(0, j_n+1)[:, np.newaxis], i_n+1)
    return np.astype(x_grid, np.float64), np.astype(y_grid, np.float64)


def calc_grid_area(x_grid, y_grid):
    '''
    Calculates area of all pixels defined by x/y grids
    using shoelace formula
    '''
    x0 = x_grid[:-1, :-1]
    y0 = y_grid[:-1, :-1]
    x1 = x_grid[1:, :-1]
    y1 = y_grid[1:, :-1]
    x2 = x_grid[1:, 1:]
    y2 = y_grid[1:, 1:]
    x3 = x_grid[:-1, 1:]
    y3 = y_grid[:-1, 1:]
    
    # Shoelace formula, vectorized
    area_grid = 0.5 * np.abs(x0 * y1 + x1 * y2 + x2 * y3 + x3 * y0 - 
                             (y0 * x1 + y1 * x2 + y2 * x3 + y3 * x0))
    return area_grid

def calc_node_vals(grid):
    '''
    Calculates the edge node values from average surrounding pixel values
    '''
    pad = np.pad(grid, pad_width=1, mode='edge')
    vertex = (pad[:-1, :-1] + pad[:-1, 1:] + pad[1:, :-1] + pad[1:, 1:]) / 4
    return vertex


def relaxation(phi, f, omega):
    '''
    Updates each cell in a matrix (phi) using SOR formula
    Omega is a convergence parameter between 1-2
    Returns phi - updated matrix, and deltas - to be used for convergence monitoring
    '''
    
    (i_n, j_n) = phi.shape
    deltas = np.zeros_like(phi)
    
    for i in range(i_n):
        for j in range(j_n):
            center = phi[i, j]
            
            # enforcing zero derivative Neumann boundary
            up    = phi[i, j-1] if j > 0     else phi[i, j+1]
            down  = phi[i, j+1] if j < j_n-1 else phi[i, j-1]
            left  = phi[i-1, j] if i > 0     else phi[i+1, j]
            right = phi[i+1, j] if i < i_n-1 else phi[i-1, j]

            delta = omega / 4 * (up + down + left + right - 4 * center - f[i, j])
            phi[i, j] += delta
            deltas[i, j] = np.abs(delta)
    return phi, deltas



def successive_over_relaxation(f, omega, conv_tol, max_iters, log=True):
    '''
    Solves  Poisson equation, ∇^2 φ= f
    To find φ for a given f, using SOR method
    
    TODO: break if converges 
    TODO: figure out how to express conv_tol as percentage...
    TODO: minimum bounding on iterations (i.e. iterate till reaches 1e-5 or atleast 200 iterations)
    '''
    
    phi = np.zeros_like(f) # initalise solution array
    
    for i in range(max_iters):
        phi, deltas = relaxation(phi, f, omega)
        
        if log == True: print(f"iteration {i}: Mean change {np.mean(np.abs(deltas))}", end="\r")
        
        if np.mean(np.abs(deltas)) < conv_tol:
            break
        
    if log == True: print('\n')
    
    return phi


def calc_brightness_area_grid(x_grid, y_grid, img, SOR_params, log=False, plot=True):
    '''
    Iteratively optimises the coordinates of an x/y grid
    such that the area of each pixel matches the brightness of an image
    '''
    (iters, conv_tol, max_iters, omega) = SOR_params

    area = calc_grid_area(x_grid, y_grid)
    brightness = img/np.sum(img)*(img.shape[0]*img.shape[1]) 
    

    logs = [] if log else False
    
    ux_grid = x_grid.copy()
    uy_grid = y_grid.copy()
    for i in range(iters):
        loss = area - brightness 
        phi = successive_over_relaxation(loss, omega, conv_tol, max_iters)
        
        dx, dy = np.gradient(phi)
        dx, dy = calc_node_vals(dx), calc_node_vals(dy)
        dx[0, :], dx[-1, :] = 0, 0 ## enforce Neumann boundaries!!!
        dy[:, 0], dy[:, -1] = 0, 0
        
        ux_grid -= dy # WHO ON earth knows why this is flipped
        uy_grid -= dx
        area = calc_grid_area(ux_grid, uy_grid)
        
        if log:
            logs.append([loss, area, phi, dx, dy])
            
        if plot:
            fig, ((ax1, ax2), (ax3, ax4)) =  plt.subplots(2, 2, figsize=(45, 45), constrained_layout=True)
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_aspect('equal') 
            #qskip = 10
            #ax5.quiver(np.arange(0, i_n+1)[::qskip], -np.arange(0, j_n+1)[::qskip], dx[::qskip, ::qskip], dy[::qskip, ::qskip])
            ax1.pcolormesh(ux_grid, -uy_grid, area)
            ax1.set_title('Pcolormesh')
            ax2.imshow(loss)
            ax2.set_title('loss_grid')
            ax3.imshow(phi)
            ax3.set_title('phi out grid')
            ax4.quiver(np.arange(0, img.shape[0]+1), -np.arange(0, img.shape[1]+1), -dy, -dx)
            ax4.set_title('dx dy')

            plt.show()
    
    return ux_grid, uy_grid, logs

def calc_height_map(grids, lens_params, SOR_params, log=False, plot=True):
    '''
    Iteratively calculates a heightmap for a material of RI=n2
    Such that the surface normals maps coordiantes at (x1,y1) to
    (x2, y2) at a focal length d 
    
    deets = dimensions (mm), d (mm) and  RIs, n1 n2
    '''
    
    # unpack
    (x1_grid, y1_grid, x2_grid, y2_grid) = grids
    (dims, d, n1, n2) = lens_params 
    (iters, conv_tol, max_iters, omega) = SOR_params
    
    # normalize to dimensions!
    x1_grid = (x1_grid - np.min(x1_grid))/np.max(x1_grid) * dims[0]
    y1_grid = (y1_grid - np.min(y1_grid))/np.max(y1_grid) * dims[1]
    x2_grid = (x2_grid - np.min(x2_grid))/np.max(x2_grid) * dims[0]
    y2_grid = (y2_grid - np.min(y2_grid))/np.max(y2_grid) * dims[1]
    
    #print(x1_grid)
    #print(x2_grid)
    logs = [] if log else False
    
    D = d



    for i in range(iters):
        # Surface normals
        Nx = np.tan( np.arctan((x2_grid-x1_grid)/D)/(n1-n2))
        Ny = np.tan( np.arctan((y2_grid-y1_grid)/D)/(n1-n2))
        # Derivatives
        dNx = np.gradient(Nx, axis=0)
        dNy = np.gradient(Ny, axis=1)
        div = dNx+dNy

        h = successive_over_relaxation(div, omega, conv_tol, max_iters)
        # update focal distance to reflect height before iterating again
        D = d - h
        
        if log:
            logs.append(h)
            
        if plot:
            plt.imshow(h)
            plt.show()
            
    return h, logs



def lens_maker(img_path):
    '''
    Rewrite params to be more functional...!
    '''
    pix_shape = (512, 512)
    img = image_read(img_path, pix_shape)
    
    ## sor Params
    conv_tol = 1e-5
    max_iters = 1000
    omega = 1.935
    iters = 5

    SOR_params = (iters, conv_tol, max_iters, omega)
    
    ## sor Params
    conv_tol = 6e-5
    max_iters = 500
    omega = 1.935
    iters = 3

    SOR2_params = (iters, conv_tol, max_iters, omega)
    
    
    ## Lens params
    dims = (100, 100) # mm
    d = 1000 # mm
    n1 = 1
    n2 = 1.495 # acrylic
    lens_params = (dims, d, n1, n2)
    
    filename = 'test.stl'

    
    xi_grid, yi_grid = create_xy_grids(pix_shape[0], pix_shape[1])
    xf_grid, yf_grid, logs = calc_brightness_area_grid(xi_grid, yi_grid, img, SOR_params)
    
    grids = ( xi_grid, yi_grid, xf_grid, yf_grid)
    h, logs = calc_height_map(grids, lens_params, SOR2_params)
    
    write(filename, xi_grid, yi_grid, h*50)

    
