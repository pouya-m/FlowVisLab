from openpiv import tools, pyprocess, validation, filters, scaling
from openpiv.smoothn import smoothn
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os, glob, warnings
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.collections import PatchCollection
from matplotlib.patches import Wedge

warnings.filterwarnings("ignore")


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
    u, v, s2n = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=stg['WS'], overlap=stg['OL'], dt=stg['DT'], search_area_size=stg['SA'], sig2noise_method='peak2peak')
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, window_size=stg['WS'], overlap=stg['OL'] )
    
    if stg['BVR'] == 'on':
        u, v, mask = validation.local_median_val(u, v, stg['MF'][0], stg['MF'][1], size=2)
        u, v, mask = validation.global_val(u, v, u_thresholds=stg['GF'][0], v_thresholds=stg['GF'][1])
        u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
        u, *_ = smoothn(u, s=0.5)
        v, *_ = smoothn(v, s=0.5)
    x, y, u, v = scaling.uniform(x, y, u, v, stg['SC'])
    # saving the results
    save_file = tools.create_path(file_a, 'Analysis')
    tools.save(x, y, u, v, s2n, save_file+'.dat')
    

# this function draws the vector field
def DrawPIVPlot(files, scale, bg):
    # reading saved data and creating vector labels
    x, y, u, v, label = tools.read_data(files)
    label = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            label.append(f'Ux:{u[i, j]:4.2f} , Uy:{v[i, j]:4.2f} (mm/s)')
    # plotting the results
    fig, ax = plt.subplots()
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
        #tools.imsave(os.path.join(stg['DP'], 'Analysis/reflection.jpg'), reflection)
    else:
        bg_a, bg_b, reflection = tools.imread(os.path.join(stg['DP'], 'avg.jpg')), None, None
    #tools.imsave(os.path.join(stg['DP'], 'Analysis/background.jpg'), bg_a)
    # start processing data
    print('main process:\nprocessing images...')
    #task.n_files = 6
    main_process = partial(ProcessPIV, bga=bg_a, bgb=bg_b, reflection=reflection, stg=stg)
    task.run( func=main_process, n_cpus=6)
    print('- done processing')
    '''
    fig, ax = plt.subplots(2,2)
    img = tools.imread(task.files_a[0])
    bg = bg_a
    ax[0,0].imshow(img, cmap='gray')
    ax[0,1].imshow(bg, cmap='gray')
    ax[1,0].imshow(reflection, cmap='gray')
    img = img - bg
    img[reflection==255] = 0
    ax[1,1].imshow(img, cmap='gray')
    plt.show()
    '''
    return bg_a


# this function writes the gerris simulation file
def WriteGerrisFile(file, stg):
    with open(file, 'w') as fh:
        fh.write('2 1 GfsSimulation GfsBox GfsGEdge {} {')
        fh.write('\n  GfsTime {{ end = {} }}'.format(stg['ST']))
        fh.write('\n  GfsRefine {}'.format(stg['MR']))
        fh.write('\n  GfsRefineSolid {}'.format(stg['MR']))
        fh.write('\n  GfsSolid model.gts')
        fh.write('\n  GfsInit {{}} {{ U = {} }}'.format(stg['IV']))
        fh.write('\n  EventStop { istep = 1 } U 0.001 DU')
        fh.write('\n  GfsOutputTime {{ step = {} }} stdout'.format(stg['ODT']))
        fh.write('\n  GfsOutputSimulation {{ start = {} step = {} }} output_%.3f.txt {{'.format(stg['OST'], stg['ODT']))
        fh.write('\n    variables = U,V,P')
        fh.write('\n    format = text')
        fh.write('\n    solid = 1')
        fh.write('\n  }')
        fh.write('\n  PhysicalParams { L = 80 }')
        fh.write('\n  SourceViscosity 1.004\n}')
        fh.write('\nGfsBox {')
        fh.write('\n  left   = GfsBoundary {{ GfsBcDirichlet U {{ return (1)*{}; }} GfsBcDirichlet V 0 }}'.format(stg['IV']))
        fh.write('\n  top    = Boundary { BcDirichlet U 0 BcDirichlet V 0 }')
        fh.write('\n  bottom = Boundary { BcDirichlet U 0 BcDirichlet V 0 }\n}')
        fh.write('\nGfsBox {')
        fh.write('\n  right  = GfsBoundaryOutflow')
        fh.write('\n  top    = Boundary { BcDirichlet U 0 BcDirichlet V 0 }')
        fh.write('\n  bottom = Boundary { BcDirichlet U 0 BcDirichlet V 0 }\n}')
        fh.write('\n1 2 right')


def update_quiver(num, q, x, y):
    x, y, u, v, _ = tools.read_data(files[num])
    q.set_UVC(u,v)
    return q


def GeneratePIVanim(files, scale, bg):
    x, y, u, v, _ = tools.read_data(files[0])
    fig, ax = plt.subplots()
    ax.imshow(bg, cmap='gray', extent=[0., 780/scale, 0., 580/scale])
    q = ax.quiver(x, y, u, v, color='b', units='xy', minlength=0.1, minshaft=1.2)
    ax.set_title('Velocity Field', size=16)
    ax.set_xlabel('x (mm)', size=14, labelpad=2)
    ax.set_ylabel('y (mm)', size=14, labelpad=-10)

    anim = animation.FuncAnimation(fig, update_quiver,frames=45, fargs=(q, x, y),
                               interval=1000, blit=False)
    plt.show()


def SavePIVanim(address, scale, bg):
    files = sorted(glob.glob(os.path.join(address, 'Analysis/frame*.dat')))
    x, y, u, v, _ = tools.read_data(files[0])
    fig = Figure(figsize=(6, 4.5), dpi=120)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.imshow(bg, cmap='gray', extent=[0., 780/scale, 0., 580/scale])
    q = ax.quiver(x, y, u, v, color='b', units='xy', minlength=0.1, minshaft=1.2)
    ax.set_title('Velocity Field', size=16)
    ax.set_xlabel('x (mm)', size=14, labelpad=2)
    ax.set_ylabel('y (mm)', size=14, labelpad=2)
    #plt.tight_layout(pad=10)
    canvas.draw()
    imglist = []
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.renderer.buffer_rgba())
    imglist.append(img)
    for i in range(len(files)):
        x, y, u, v, _ = tools.read_data(files[i])
        q.set_UVC(u,v)
        canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.renderer.buffer_rgba())
        imglist.append(img)

    imglist[0].save(os.path.join(os.path.dirname(files[0]), '1result.gif'),
               save_all=True, append_images=imglist[1:], optimize=False, duration=200, loop=0)


def SaveCFDanim(address, p, vmin, vmax):
    files = sorted(glob.glob(os.path.join(address, 'output_*.txt')))
    a = np.loadtxt(files[0], skiprows=1)
    x, y, u, v, pressure = a[:,0], a[:,1], a[:,3], a[:,4], a[:,5]
    fig = Figure(figsize=(5, 4.7), dpi=120)
    canvas = FigureCanvasAgg(fig)
    ax = fig.subplots()
    # x and y axes are messed up bc we are plotting and rotating the results CCW at the same time
    cntr = ax.tricontourf(-y, x, pressure, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
    fig.colorbar(cntr, ax=ax, aspect=30, shrink=0.5)
    q = ax.quiver(-y, x, -v, u, units='xy', color='k', minlength=0.1, minshaft=1.2)
    ax.add_collection(p)
    ax.axis([-40, 40, -40, 80])
    ax.set_title('Velocity Field + pressure contour', size=16, pad=25)
    ax.set_xlabel('x (mm)', size=14, labelpad=10)
    ax.set_ylabel('y (mm)', size=14, labelpad=-2)
    canvas.draw()
    imglist = []
    img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.renderer.buffer_rgba())
    imglist.append(img)
    for i in range(len(files)):
        a = np.loadtxt(files[i], skiprows=1)
        u, v, pressure = a[:,3], a[:,4], a[:,5]
        #ax.cla()
        cntr = ax.tricontourf(-y, x, pressure, levels=20, cmap='hot', vmin=vmin, vmax=vmax)
        q = ax.quiver(-y, x, -v, u, color='k', units='xy', minlength=0.1, minshaft=1.2)
        ax.add_collection(p)
        canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.renderer.buffer_rgba())
        imglist.append(img)

    imglist[0].save(os.path.join(os.path.dirname(files[0]), 'CFD.gif'),
               save_all=True, append_images=imglist[1:], optimize=False, duration=200, loop=0)


# test function
def TestRun():
    # setting up the process settings
    stg = {}
    stg['WS'] = 64
    stg['OL'] = 32
    stg['SA'] = 80
    stg['SC'] = 1
    stg['BR'] = 'on'
    stg['BVR'] = 'off'
    stg['DT'] = 0.001094
    stg['DP'] = os.path.join('E:\\repos\\FlowVisLab\\Images', 'raw_001094_25')
    stg['MF'] = [8000,15000]
    stg['GF'] = [(-10000,10000),(-1000,20000)]

    #bg = ProcessHandler(stg)
    bg = tools.imread(os.path.join(stg['DP'], 'avg.jpg'))

    # plot results
    files = os.path.join(stg['DP'], f'Analysis/frame0008.dat')
    fig, q, label = DrawPIVPlot(files, stg['SC'], bg)
    return fig

if __name__ == '__main__':

    #bg = tools.imread('E:\\repos\\FlowVisLab\\Images\\Orifice_flat\\avg.jpg')
    #files = sorted(glob.glob('E:\\repos\\FlowVisLab\\Images\\Orifice_flat\\Analysis\\frame*.dat'))
    #SavePIVanim('E:\\repos\\FlowVisLab\\Images\\Orifice_flat', 25, bg)
    #h=-27.5
    #p = PatchCollection([Wedge((-40,h+13.5), 6.625, -90, 0)], alpha=1)
    #SaveCFDanim('C:\\Users\\Asus\\Desktop\\Remote Lab\\CFD', p, vmin=-120000, vmax=24000)
    fig = TestRun()
    plt.show()