import tools
import numpy as np

#x = np.array([[0,1,2,3,4],[0,1,2,3,4],[0,1,2,3,4]])
#y = np.array([[2,2,2,2,2],[1,1,1,1,1],[0,0,0,0,0]])
#tools.save(x, y, x, x, x, 'E:\\repos\\FlowVisLab\\test.dat')
xx , yy, *_ = tools.read_data('E:\\repos\\FlowVisLab\\openpiv\\frame0000.dat')
b = round(22.01)
print(xx)