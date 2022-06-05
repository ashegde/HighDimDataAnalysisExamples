#Arun Hegde
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt
from matplotlib import cm

# 2D case

x = np.linspace(-1,1,1000)
y_lhalf = (1 - np.sqrt(abs(x)))**2
y_l1 = 1-abs(x)
y_l2 = np.sqrt(1-x**2)
y_linf = np.tile([-1],len(x))

fig, ax = plt.subplots(figsize=(5,5))
plt.plot(x,y_lhalf,'b',x,-y_lhalf,'b')
plt.plot(x,y_l1, 'r', x,-y_l1,'r')
plt.plot(x,y_l2, 'y', x,-y_l2,'y')
plt.plot(x,y_linf, 'm', x, -y_linf, 'm', y_linf, x, 'm', -y_linf, x, 'm')
plt.title(r'2D unit norm balls: from $L_{0.5}$ to $L_{\inf}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig('pnorm_2d.pdf')
plt.close()
# 3D case: L_{0.5} ball

def f(x, y):
    z_temp = (1 - np.sqrt(abs(x))-np.sqrt(abs(y)))
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if z_temp[ii][jj] < 0 :
                z_temp[ii][jj] = np.nan
    return z_temp**2


x = np.linspace(-1,1,1000)
y = np.linspace(-1,1,1000)

xv, yv = np.meshgrid(x,y)

z_lhalf = f(xv,yv)

fig, ax = plt.subplots(figsize=(5,5))
ax = plt.axes(projection='3d')
ax.plot_wireframe(xv, yv, z_lhalf,color='b', rstride=50, cstride = 50)
ax.plot_wireframe(xv, yv, -z_lhalf,color='b',rstride=50, cstride = 50)
plt.title(r'3D unit norm ball for $L_{0.5}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('half_norm_3d.pdf')
plt.close()

# 3D case: L_1 ball

def f(x, y):
    z_temp = (1 - abs(x)-abs(y))
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if z_temp[ii][jj] < 0 :
                z_temp[ii][jj] = np.nan
    return z_temp


x = np.linspace(-1,1,1000)
y = np.linspace(-1,1,1000)
x = np.append(x, 0)
y = np.append(y, 0)
xv, yv = np.meshgrid(x,y)

z_l1 = f(xv,yv)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(xv, yv, -z_l1,color='r',rstride=50, cstride = 50)
ax.plot_surface(xv, yv, z_l1,color='r')
ax.view_init(30, 45)
plt.title(r'3D unit norm ball for $L_{1}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('one_norm_3d.pdf')
plt.close()

# 3D case: L-2 ball

def f(x, y):
    z_temp = (1 - x**2-y**2)
    for ii in range(x.shape[0]):
        for jj in range(x.shape[1]):
            if z_temp[ii][jj] < 0 :
                z_temp[ii][jj] = np.nan
    return np.sqrt(z_temp)


x = np.linspace(-1,1,1000)
y = np.linspace(-1,1,1000)
x = np.append(x, 0)
y = np.append(y, 0)
xv, yv = np.meshgrid(x,y)

z_l2 = f(xv,yv)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(xv, yv, z_l2,color='y', rstride=50, cstride = 50)
ax.plot_wireframe(xv, yv, -z_l2,color='y',rstride=50, cstride = 50)
plt.title(r'3D unit norm ball for $L_{2}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('two_norm_3d.pdf')
plt.close()

# 3D case: Linf ball

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
v = np.array([[1, 1, 1], [1, 1, -1], [1, -1, -1],  [-1, -1, -1], [-1, -1, 1],[-1, 1, 1], [-1, 1, -1], [1, -1, 1]])
faces = [ [v[0],v[7],v[4],v[5]], [v[0],v[1],v[2],v[7]],[v[7],v[4],v[3],v[2]]]
ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
ax.add_collection3d(Poly3DCollection(faces,
 facecolors='m', linewidths=1, edgecolors='k', alpha=.1))

plt.title(r'3D unit norm ball for $L_{2}$')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('inf_norm_3d.pdf')
plt.close()

# 1 norm and plane intersection


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#create the affine set
affineSet = [ [[-2,2, 2], [-2,-2, 1], [2,-2, 1], [1,1, 2]]  ]
collection1 = Poly3DCollection(affineSet, linewidths=1, edgecolors='k', alpha=.3)
collection1.set_facecolor('cyan')
ax.add_collection3d(collection1)

p = 1.6 #tune this to intersect the affine space
v = np.array([[0,0,p], [p,0,0],[0,p,0], [0,0,-p], [-p,0,0],[0,-p,0] ])
faces = [ [v[0],v[1], v[2]], [v[0],-v[1], v[2]], [v[0],v[1], -v[2]], [v[0],-v[1], -v[2]],[-v[0],v[1], v[2]], [-v[0],-v[1], v[2]], [-v[0],v[1], -v[2]], [-v[0],-v[1], -v[2]]  ]

ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
collection = Poly3DCollection(faces, linewidths=1, edgecolors='k', alpha=.5)
collection.set_facecolor('grey')
ax.add_collection3d(collection)

ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2,2)
ax.set_zlim3d(-1.5,2)
ax.view_init(5, 25)

plt.title(r'$L_{1}$ ball intersection with the plane')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('one_norm_and_plane_3d.pdf')
plt.close()
