from openpiv import tools, pyprocess, validation, filters
from openpiv.smoothn import smoothn
import os,glob
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from imageio import imread, imsave
from skimage import draw
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def process (args, bga, bgb, reflection):
    file_a, file_b, counter = args

    # read images into numpy arrays
    frame_a = tools.imread( file_a )
    frame_b  = tools.imread( file_b )
  
    # removing background and reflections
    frame_a = frame_a - bga
    frame_b =frame_b - bgb
    frame_a[reflection==255] = 0
    frame_b[reflection==255] = 0

    #applying a static mask (taking out the regions where we have walls)
    yp = [580,435,0,0,580,580,0,0,435,580]
    xp = [570,570,680,780,780,0,0,105,230,230]
    pnts = draw.polygon(yp, xp, frame_a.shape)
    frame_a[pnts] = 0
    frame_b[pnts] = 0

    # checking the resulting frame
    #fig, ax = plt.subplots(2,2)
    #ax[0,0].imshow(frame_a_org, cmap='gray')
    #ax[0,1].imshow(frame_a, cmap='gray')
    #ax[1,0].imshow(frame_b_org, cmap='gray')
    #ax[1,1].imshow(frame_b, cmap='gray')
    #plt.tight_layout()
    #plt.show()
    
    # main piv processing
    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=48, overlap=16, dt=0.001094, search_area_size=64, sig2noise_method='peak2peak')
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=48, overlap=16 )
    u, v, mask = validation.local_median_val(u, v, 2000, 2000, size=2)
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    u, *_ = smoothn(u, s=1.0)
    v, *_ = smoothn(v, s=1.0)

    # saving the results
    save_file = tools.create_path(file_a, 'Analysis')
    tools.save(x, y, u, v, mask, save_file+'.dat')

def plotResult(file):
    x, y, u, v, mask = tools.read_data(file)
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, color='b', units='xy', minlength=0.1, minshaft=1.2)
    patches = [Polygon([[575,0],[575,105],[580,145],[680,580],[780,580],[780,0],[230,0],[230,105],[225,145],[105,580],[0,580],[0,0]], True)]
    p = PatchCollection(patches, alpha=1.0)
    ax.add_collection(p)
    ax.axis([0, 780, 0, 580])
    plt.show()


if __name__ == '__main__':
    '''
    # setting up the path and grabbing the files
    run_path = 'E:\\repos\\FlowVisLab\\Images'
    data_path = os.path.join(run_path, 'raw_001094')
    analysis_path = tools.create_directory(data_path)
    task = tools.Multiprocesser( data_dir=data_path, pattern_a='frame*.jpg' )
    
    # finding background and reflections in the images
    print('preprocessing:')
    bg_a, bg_b = task.find_background(50, 10 , 6)
    reflection = tools.mark_reflection(130, task.files_a, os.path.join(data_path, 'ref.tif'))

    # start processing data
    print('main process:\nprocessing images...')
    main_process = partial(process, bga=bg_a, bgb=bg_b, reflection=reflection)
    task.n_files = 10
    task.run( func=main_process, n_cpus=4)
    print('- done processing')
    
    # plotting the result
    plotResult('E:\\repos\\FlowVisLab\\Images\\raw_001094\\Analysis\\frame0000.dat')
    '''
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    files = sorted(glob.glob('E:\\repos\\FlowVisLab\\Images\\raw_001094\\Analysis\\frame*.dat'))
    x, y, *_ = tools.read_data(files[0])
    idx = find_nearest(x[0,:], 400)
    idy = find_nearest(y[:,0], 200)
    u1 = []
    v1 = []
    for i in range(len(files)):
        x, y, u, v, mask = tools.read_data(files[i])
        u1.append(u[idx,idy])
        v1.append(v[idx,idy])
    fig, ax = plt.subplots()
    ax.plot(u1)
    ax.plot(v1)
    plt.show()
    