import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

def read_data(filename):
    """function to read the saved data file and reconstruct the field data
    
    Parameters
    ----------
    filename :  string, path to the file

    Returns
    --------
    x, y, z, u, v, p : numpy arrays containing the position and field data
    """
    a = np.loadtxt(filename, skiprows=1)
    
    return a[:,0], a[:,1], a[:,2], a[:,3], a[:,4], a[:,5]

def DrawPlot(file):
    x, y, z, u, v, p = read_data(file)
    fig, ax = plt.subplots()
    ax.quiver(-y, x, -v, u, units='xy', color='b')
    patches = [Polygon([[-40,-35],[-40,40],[-6.7,40],[-2.5,24.3],[-2.5,10]], True), \
        Polygon([[40,-35],[40,40],[6.7,40],[2.5,24.3],[2.5,10]], True)]
    p = PatchCollection(patches, alpha=0.8)
    ax.add_collection(p)
    plt.axis([-40, 40, -40, 120])
    plt.show()

#DrawPlot('C:\\Users\\Asus\\Desktop\\Remote Lab\\Gerris files\\output_4.000.txt')

def WriteGerrisFile(file):
    with open(file, 'w') as fh:
        fh.write('2 1 GfsSimulation GfsBox GfsGEdge {} {')
        fh.write('\n  GfsTime { end = 5 }')
        fh.write('\n  GfsRefine 5')
        fh.write('\n  GfsRefineSolid 5')
        fh.write('\n  GfsSolid model.gts')
        fh.write('\n  GfsInit {} { U = 100.00000000000000000000 }')
        fh.write('\n  EventStop { istep = 1 } U 0.001 DU')
        fh.write('\n  GfsOutputTime { step = .50000000000000000000 } stdout')
        fh.write('\n  GfsOutputSimulation { start = 4 step = .50000000000000000000 } output_%.3f.txt {')
        fh.write('\n    variables = U,V,P')
        fh.write('\n    format = text')
        fh.write('\n    solid = 1')
        fh.write('\n  }')
        fh.write('\n  PhysicalParams { L = 80 }')
        fh.write('\n  SourceViscosity 1.004\n}')
        fh.write('\nGfsBox {')
        fh.write('\n  left   = GfsBoundary { GfsBcDirichlet U { return (1)*100.00000000000000000000; } GfsBcDirichlet V 0 }')
        fh.write('\n  top    = Boundary { BcDirichlet U 0 BcDirichlet V 0 }')
        fh.write('\n  bottom = Boundary { BcDirichlet U 0 BcDirichlet V 0 }\n}')
        fh.write('\nGfsBox {')
        fh.write('\n  right  = GfsBoundaryOutflow')
        fh.write('\n  top    = Boundary { BcDirichlet U 0 BcDirichlet V 0 }')
        fh.write('\n  bottom = Boundary { BcDirichlet U 0 BcDirichlet V 0 }\n}')
        fh.write('\n1 2 right')

WriteGerrisFile('C:\\Users\\Asus\\Desktop\\Remote Lab\\Gerris files\\simulation.dat')