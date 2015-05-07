import numpy as np
from numba.decorators import jit as jit
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes
import matplotlib.cm as cm
from scipy.misc import imread
from som import Som


def generateChart(obj):
    
    r = []
    g = []
    b = []

    for i in range(obj.num_featureX):
        tempr = []
        tempb = []
        tempg = []
        for j in range(obj.num_featureY):
            tempr.append(obj.wts_input_map[(i,j)][0])
            tempg.append(obj.wts_input_map[(i,j)][1])
            tempb.append(obj.wts_input_map[(i,j)][2])
    
        r.append(tempr)
        g.append(tempg)
        b.append(tempb)
    
    r = np.asarray(r)
    g = np.asarray(g)
    b = np.asarray(b)

    print("Generating the chart...")

    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
    kwargs = dict(origin="lower", interpolation="nearest")
    ax.imshow_rgb(r, g, b, **kwargs)

    ax.RGB.set_xlim(0., obj.num_featureX)
    ax.RGB.set_ylim(0.9, obj.num_featureY)

    plt.draw()
    plt.show()


som = Som(3, 50, 50, 561)
print("Self-organising Map: initialised...")

#generateChart(som)

print(len(som.wts_input_map))

def get_list(l):
    newl = []
    for i in l:
        newl.append(float(i))

    return newl


#plt.axis([0, 4000, 0, 4000])
#img = imread("Environment4/map4.jpg")

with open('Environment1/hidden.data', 'r') as fp:
    i = 0
    for line in fp:
        #if i%50 == 0: generateChart(som)
        list_input = get_list(line.split()[:3])
        list_pos = get_list(line.split()[3:])
        som.step(i, list_input)
        i = i+1
'''
som.set_total_iterations(500)
with open('Environment4/hidden3.data', 'r') as fp:
    for line in fp:
        list_input = get_list(line.split()[:3])
        list_pos = get_list(line.split()[3:])
        som.step(i, list_input)
        plt.scatter((list_pos[0] + 0.5), (list_pos[1] + 0.5), color=som.bmu[i])
        i = i+1
   
print("Self-organising Map: trained...")

#plt.imshow(img,zorder=0)
plt.show()
'''
generateChart(som)
'''
for i in range(som.num_featureX):
    for j in range(som.num_featureY):
        print(som.wts_input_map[(i,j)])
'''
'''
nparray = []
r = []
g = []
b = []

for i in range(som.num_featureX):
    tempr = []
    tempb = []
    tempg = []
    for j in range(som.num_featureY):
        tempr.append(som.wts_input_map[(i,j)][0])
        tempg.append(som.wts_input_map[(i,j)][1])
        tempb.append(som.wts_input_map[(i,j)][2])
        nparray.append(som.wts_input_map[(i,j)])        

    r.append(tempr)
    g.append(tempg)
    b.append(tempb)
    
r = np.asarray(r)
g = np.asarray(g)
b = np.asarray(b)
nparray = np.asarray(nparray)

#print(r)
#print(g)
#print(b)

print("Generating the chart...")


fig = plt.figure(1)
ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
kwargs = dict(origin="lower", interpolation="nearest")
ax.imshow_rgb(r, g, b, **kwargs)

#ax.RGB.set_xlim(0., 9.5)
ax.RGB.set_xlim(0., som.num_featureX)
#ax.RGB.set_ylim(0.9, 10.6)
ax.RGB.set_ylim(0.9, som.num_featureY)

plt.draw()
plt.show()
'''


'''

def get_demo_image():
    import numpy as np
    from matplotlib.cbook import get_sample_data
    f = get_sample_data("axes_grid/bivariate_normal.npy", asfileobj=False)
    z = np.load(f)
    # z is a numpy array of 15x15
    return z, (-3,4,-4,3)

F = plt.figure(1, (som.num_featureX, som.num_featureY))
grid = ImageGrid(F, 111, # similar to subplot(111)
                nrows_ncols = (1, 3),
                axes_pad = 0.1,
                add_all=True,
                label_mode = "L",
                )

#Z, extent = get_demo_image() # demo image

im1=r
im2=g
im3=b
vmin, vmax = nparray.min(), nparray.max()
for i, im in enumerate([im1, im2, im3]):
    ax = grid[i]
    ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

plt.draw()
plt.show()
'''
