"""visualize canonical_face_model in world coordinate system
"""
import sys
sys.path.append(r"C:\Users\poyuan.chih\Desktop\dlib_headpose")
from src.face_3d import get_face3d_468, get_face3d_68, get_face3d_14, get_cube_pts
import matplotlib.pyplot as plt

if __name__ == "__main__":
    object_pts = get_face3d_68()
    object_14 = get_face3d_14()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    origin = (0, 0, 0)
    xaxis = (1, 0, 0)
    yaxis = (0, 1, 0)
    zaxis = (0, 0, 1)
    ax.scatter3D(origin[0], origin[1], origin[2], c='blue')
    ## x-axis
    ax.quiver(origin[0], origin[1], origin[2], xaxis[0], xaxis[1], xaxis[2], length=1, normalize=True, color='b')
    ## y-axis
    ax.quiver(origin[0], origin[1], origin[2], yaxis[0], yaxis[1], yaxis[2], length=1, normalize=True, color='g')
    ## z-axis
    ax.quiver(origin[0], origin[1], origin[2], zaxis[0], zaxis[1], zaxis[2], length=1, normalize=True, color='r')
    ## 68 pts
    ax.scatter3D(object_pts[:, 0], object_pts[:, 1], object_pts[:, 2], c='black')
    ## selected 14 pts
    ax.scatter3D(object_14[:, 0], object_14[:, 1], object_14[:, 2], c='red', s=50)
    ## add text
    for idx, pt in enumerate(object_14):
        ax.text(pt[0], pt[1], pt[2], f"{idx}")
    
    ## add 3d box here
    # get (xmin, xmax), (ymin, ymax), (zmin, zmax)
    xmin, xmax, ymin, ymax, zmin, zmax = get_cube_pts()
    ax.plot3D([xmin, xmax, xmax, xmin, xmin],
              [ymin, ymin, ymax, ymax, ymin],
              [zmin, zmin, zmin, zmin, zmin],
              c='b'
              )
    ax.plot3D([xmin, xmax, xmax, xmin, xmin],
              [ymin, ymin, ymax, ymax, ymin],
              [zmax, zmax, zmax, zmax, zmax],
              c='b')
    ax.plot3D([xmin, xmin],
              [ymin, ymin],
              [zmin, zmax],
              c='b')
    ax.plot3D([xmax, xmax],
              [ymin, ymin],
              [zmin, zmax],
              c='b')
    ax.plot3D([xmin, xmin],
              [ymax, ymax],
              [zmin, zmax],
              c='b')
    ax.plot3D([xmax, xmax],
              [ymax, ymax],
              [zmin, zmax],
              c='b')
        #xmin, ymin, zmin,
        #    xmax, ymin, zmin,
        #    xmax, ymax, zmin,
        #    xmax, ymax, zmax,
        #    xmin, ymin, zmin)
    plt.show()
