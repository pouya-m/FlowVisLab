from openpiv import tools, pyprocess, validation
import os

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


run_path = ''
raw_data_path = os.path.join(run_path, 'Images')
Analysis_path = tools.create_directory(run_path)
task = tools.Multiprocesser( data_dir=raw_data_path, pattern_a='*LA.TIF', pattern_b='*LB.TIF' )
task.run( func = process, n_cpus=1)