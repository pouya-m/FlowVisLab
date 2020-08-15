import tools, pyprocess, validation, filters, scaling
from smoothn import smoothn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from skimage import draw
import os


# this function runs the main PIV analysis
def ProcessPIV (args, bga, bgb, reflection, stg):
    # read images into numpy arrays
    file_a, file_b, counter = args
    frame_a = tools.imread( file_a )
    frame_b  = tools.imread( file_b )
    # removing background and reflections
    if bga is not None:
        frame_a = frame_a - bga
        frame_b =frame_b - bgb
        frame_a[reflection==255] = 0
        frame_b[reflection==255] = 0
    #applying a static mask (taking out the regions where we have walls)
    pnts = draw.polygon(stg['YM'], stg['XM'], frame_a.shape)
    frame_a[pnts] = 0
    frame_b[pnts] = 0
    plt.imshow(frame_a, cmap='gray')
    plt.show()
    
    # main piv processing
    u, v = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=stg['WS'], overlap=stg['OL'], dt=stg['DT'], search_area_size=stg['SA'], sig2noise_method=None)
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=stg['WS'], overlap=stg['OL'] )
    u, v, mask = validation.local_median_val(u, v, 2000, 2000, size=2)
    if stg['BVR'] == 'on':
        u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
        u, *_ = smoothn(u, s=0.5)
        v, *_ = smoothn(v, s=0.5)
    x, y, u, v = scaling.uniform(x, y, u, v, stg['SC'])
    # saving the results
    save_file = tools.create_path(file_a, 'Analysis')
    tools.save(x, y, u, v, mask, save_file+'.dat')
    

# this function draws the vector field
def DrawPIVPlot(files, bg_a, points):
    # reading saved data and creating vector labels
    x, y, u, v, mask = tools.read_data(files)
    label = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            label.append(f'Ux:{u[i, j]:6.4f} , Uy:{v[i, j]:6.4f} (mm/s)')
    # plotting the results
    #fig, ax = plt.subplots()
    #ax.imshow(np.flipud(bg_a), cmap='gray')
    #plt.draw()
    q = ax.quiver(x, y, u, v, color='b', units='xy', minlength=0.1, minshaft=1.2)
    ax.set_title('Velocity Field', size=16)
    patches = [Polygon(points, True)]
    p = PatchCollection(patches, alpha=1.0)
    ax.add_collection(p)
    ax.axis([0, 780, 0, 580])
    plt.xlabel('x (mm)', size=14, labelpad=2)
    plt.ylabel('y (mm)', size=14, labelpad=-10)
    return fig, q, label


# this function handles the whole processing
def ProcessHandler(stg):
    # setting up the path and grabbing the files
    analysis_path = tools.create_directory(stg['DP'])
    task = tools.Multiprocesser( data_dir=stg['DP'], pattern_a='frame*.jpg' )
    # finding background and reflections in the images
    if stg['BR'] == 'on':
        print('preprocessing:')
        bg_a, bg_b = task.find_background(50, 10 , 6)
        reflection = tools.mark_reflection(150, task.files_a, os.path.join(stg['DP'], 'reflection.tif'))
    else:
        bg_a, bg_b, reflection = None, None, None
    # start processing data
    print('main process:\nprocessing images...')
    task.n_files = 4
    main_process = partial(ProcessPIV, bga=bg_a, bgb=bg_b, reflection=reflection, stg=stg)
    task.run( func=main_process, n_cpus=6)
    print('- done processing')
    '''
    fig, ax = plt.subplots(2,2)
    img = tools.imread(task.files_b[0])
    bg = bg_b
    ax[0,0].imshow(img, cmap='gray')
    ax[0,1].imshow(bg, cmap='gray')
    ax[1,0].imshow(reflection, cmap='gray')
    img = img - bg
    img[reflection==255] = 0
    ax[1,1].imshow(img, cmap='gray')
    plt.show()
    
    img = tools.imread(task.files_b[0])
    plt.imshow(img, cmap='gray')
    plt.show()
    '''
    return bg_a

# test function
def TestRun():
    # setting up the process settings
    stg = {}
    stg['WS'] = 80
    stg['OL'] = 40
    stg['SA'] = 80
    stg['SC'] = 1
    stg['BR'] = 'on'
    stg['BVR'] = 'on'
    stg['DT'] = 0.001094
    # orifice_nozzle:
    #stg['XM'] = [570,570,680,780,780,0,0,105,230,230]
    #stg['YM'] = [580,435,0,0,580,580,0,0,435,580]
    #stg['PP'] = np.array([[575,0],[575,105],[580,145],[680,580],[780,580],[780,0],[230,0],[230,105],[225,145],[105,580],[0,580],[0,0]])/stg['SC']
    # orifice_flat:
    stg['XM'] = [660,660,520,540,660,660,780,780,86,87,232,212,87,95,0,0]
    stg['YM'] = [580,440,440,405,405,0,0,580,580,440,440,405,405,0,0,580]
    stg['PP'] = [[660,0],[660,140],[520,140],[540,175],[660,175],[660,580],[780,580],[780,0],[86,0],[87,140],[232,140],[212,175],[87,175],[95,580],[0,580],[0,0]]
    #stg['PP'] = [[0,0],[0,1],[1,0]]

    stg['DP'] = os.path.join('E:\\repos\\FlowVisLab\\Images', 'Orifice_flat')

    bg_a = ProcessHandler(stg)

    # plot results
    files = os.path.join(stg['DP'], f'Analysis/frame0000.dat')
    fig, q, label = DrawPIVPlot(files, bg_a,  stg['PP'])
    return fig


if __name__ == '__main__':
    fig = TestRun()
    plt.show()