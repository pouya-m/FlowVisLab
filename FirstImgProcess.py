from openpiv import tools, pyprocess, validation, filters
from smoothn import smoothn
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from imageio import imread, imsave
from skimage import filters as filt
from skimage import draw
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def process (args):
  file_a, file_b, counter = args
    
  # read images into numpy arrays
  frame_a = tools.imread( file_a )
  frame_b  = tools.imread( file_b )
  print(counter+1)

  # process image pair with piv algorithm.
  u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
      window_size=32, overlap=16, dt=0.0015, search_area_size=32, sig2noise_method='peak2peak')
  x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=32, overlap=16 )

  u, v, mask1 = validation.sig2noise_val( u, v, sig2noise, threshold = 1.0 )
  u, v, mask2 = validation.global_val( u, v, (-2000, 2000), (-2000, 4000) )
  u, v, mask3 = validation.local_median_val(u, v, 400, 400, size=2)
  #u, v, mask4 = validation.global_std(u, v, std_threshold=3)
  mask = mask1 | mask2 | mask3
  #u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)

  save_file = tools.create_path(file_a, 'Analysis')
  tools.save(x, y, u, v, mask, save_file+'.dat')

if __name__ == '__main__':
    run_path = 'E:\Repos\FlowVisLab\Images'
    raw_data_path = os.path.join(run_path, 'raw_000594')
    Analysis_path = tools.create_directory(raw_data_path)
    task = tools.Multiprocesser( data_dir=raw_data_path, pattern_a='frame*.jpg' )
    #task.run( func = process, n_cpus=1)
    frame_a = imread(task.files_a[30])
    frame_b = imread(task.files_b[30])
    
    #bg = imread(os.path.join(raw_data_path, 'avg.jpg'))
    bg_a, bg_b = task.find_background(50, 10 , 6)
    imsave(os.path.join(raw_data_path, 'bg_a.jpg'), bg_a)
    imsave(os.path.join(raw_data_path, 'bg_b.jpg'), bg_b)
    #ref = tools.mark_background(100, task.files_a, os.path.join(raw_data_path, 'ref.jpg'))
    frame_a = frame_a - bg_a
    frame_b =frame_b - bg_b
    #frame_a[ref==255] = 0
    #frame_b[ref==255] = 0
    #applying a static mask
    yp = [580,435,0,0,580,580,0,0,435,580]
    xp = [570,570,680,780,780,0,0,105,230,230]
    pnts = draw.polygon(yp, xp, frame_a.shape)
    frame_a[pnts] = 0
    frame_b[pnts] = 0
    
    
    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=48, overlap=16, dt=1, search_area_size=64, sig2noise_method='peak2peak')
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=48, overlap=16 )
    u, v, mask = validation.local_median_val(u, v, 5, 5, size=2)
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    u, *_ = smoothn(u, s=1.0)
    v, *_ = smoothn(v, s=1.0)

    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v, color='b', units='xy', minlength=0.1, minshaft=1.2)
    patches = [Polygon([[575,0],[575,105],[580,145],[680,580],[780,580],[780,0],[230,0],[230,105],[225,145],[105,580],[0,580],[0,0]], True)]
    p = PatchCollection(patches, alpha=1.0)
    ax.add_collection(p)
    ax.axis([0, 780, 0, 580])
    plt.show()
    
    
    
    