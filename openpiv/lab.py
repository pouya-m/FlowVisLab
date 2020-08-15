import tools, pyprocess, validation, filters, scaling
from smoothn import smoothn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os


# this function runs the main PIV analysis
def ProcessPIV (args, bga, bgb, reflection, stg):
    # read images into numpy arrays
    file_a, file_b, counter = args
    frame_a = tools.imread( file_a )
    frame_b  = tools.imread( file_b )
    # removing background and reflections
    if bgb is not None:
        frame_a = frame_a - bga
        frame_b =frame_b - bgb
        frame_a[reflection==255] = 0
        frame_b[reflection==255] = 0
    #plt.imshow(frame_a, cmap='gray')
    #plt.show()
    
    # main piv processing
    u, v = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=stg['WS'], overlap=stg['OL'], dt=stg['DT'], search_area_size=stg['SA'], sig2noise_method=None)
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=stg['WS'], overlap=stg['OL'] )
    
    if stg['BVR'] == 'on':
        u, v, mask = validation.local_median_val(u, v, stg['UV'], stg['VV'], size=2)
        #u, v, mask = validation.global_val(u, v, u_thresholds=(-200,200), v_thresholds=(-200,1000))
        u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
        u, *_ = smoothn(u, s=0.5)
        v, *_ = smoothn(v, s=0.5)
    x, y, u, v = scaling.uniform(x, y, u, v, stg['SC'])
    # saving the results
    save_file = tools.create_path(file_a, 'Analysis')
    tools.save(x, y, u, v, x, save_file+'.dat')
    

# this function draws the vector field
def DrawPIVPlot(files, scale):
    # reading saved data and creating vector labels
    x, y, u, v, label = tools.read_data(files)
    label = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            label.append(f'Ux:{u[i, j]:6.4f} , Uy:{v[i, j]:6.4f} (mm/s)')
    # plotting the results
    fig, ax = plt.subplots()
    bg = tools.imread(os.path.join(os.path.dirname(files), 'background.jpg'))
    ax.imshow(bg, cmap='gray', extent=[0., 780/scale, 0., 580/scale])
    q = ax.quiver(x, y, u, v, color='b', units='xy', minlength=0.1, minshaft=1.2)
    ax.set_title('Velocity Field', size=16)
    ax.set_xlabel('x (mm)', size=14, labelpad=2)
    ax.set_ylabel('y (mm)', size=14, labelpad=-10)
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
        reflection = tools.mark_reflection(180, task.files_a)
        tools.imsave(os.path.join(stg['DP'], 'Analysis/reflection.jpg'), reflection)
    else:
        bg_a, bg_b, reflection = tools.imread(os.path.join(stg['DP'], 'avg.jpg')), None, None
    tools.imsave(os.path.join(stg['DP'], 'Analysis/background.jpg'), bg_a)
    # start processing data
    print('main process:\nprocessing images...')
    task.n_files = 6
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
    '''

# test function
def TestRun():
    # setting up the process settings
    stg = {}
    stg['WS'] = 48
    stg['OL'] = 16
    stg['SA'] = 64
    stg['SC'] = 1
    stg['BR'] = 'on'
    stg['BVR'] = 'on'
    stg['DT'] = 0.001094
    stg['DP'] = os.path.join('E:\\repos\\FlowVisLab\\Images', 'Orifice_flat')
    stg['UV'] = 2000
    stg['VV'] = 2000

    ProcessHandler(stg)

    # plot results
    files = os.path.join(stg['DP'], f'Analysis/frame0000.dat')
    fig, q, label = DrawPIVPlot(files, stg['SC'])
    return fig


if __name__ == '__main__':
    fig = TestRun()
    plt.show()