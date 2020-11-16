import os
import numpy as np
from stl import mesh
import matplotlib as mplot
import mpl_toolkits
from mpl_toolkits import mplot3d
import matplotlib.pyplot as mplot
import open3d as o3d
from scipy.spatial import Delaunay

#Poisson Surface Reconstruction - 2-3 minute compile
mesh = (o3d.io.read_triangle_mesh("STLs/dragon.stl"))
cloud = o3d.geometry.PointCloud()
cloud.points = mesh.vertices
#print(cloud.has_normals())
cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=1000))
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    render, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cloud, depth=15)
print(render)
o3d.visualization.draw_geometries([render])
vertices_to_remove = densities < np.quantile(densities, 0.01)
render.remove_vertices_by_mask(vertices_to_remove)
print(render)
o3d.visualization.draw_geometries([render])

#Matplotlib - Extremely fast, low poly only
for file in os.listdir('PlatonicSolids'):
    #Using an existing stl file:
    testMesh = mesh.Mesh.from_file("PlatonicSolids/"+file)
    #testMesh = mesh.Mesh.from_file("STLs/"+file)
    testMesh.normals

    fig = mplot.figure()
    axes = mplot3d.Axes3D(fig)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(testMesh.vectors))
    scale = testMesh.points.flatten('F')
    axes.auto_scale_xyz(scale, scale, scale)
    mplot.show()

#TetraMesh - Fast but not precise or accurate
for file in os.listdir('STLs'):
    mesh = (o3d.io.read_triangle_mesh("STLs/"+file))
    cloud = o3d.geometry.PointCloud()
    cloud.points = mesh.vertices
    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(cloud)
    o3d.visualization.draw_geometries([tetra_mesh])

#Alpha Shape - Fast Compile but creates false images when the alpha is high enough to hit all the points
mesh = (o3d.io.read_triangle_mesh("STLs/dragon.stl"))
cloud = o3d.geometry.PointCloud()
cloud.points = mesh.vertices
tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(cloud)
for alpha in np.logspace(np.log10(5), np.log10(0.25), num=5):
    print(f"alpha={alpha:.3f}")
    Tmesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha, tetra_mesh, pt_map)
    Tmesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([Tmesh])

#SciPy Delaunay Triangulation - 30 min compile time. Creates a mesh inside a tetramesh
points = np.asarray(mesh.vertices)
tri = Delaunay(points) # points: np.array() of 3d points 
indices = tri.simplices
vertices = points[indices]
print(tri)
#o3d.visualization.draw_geometries([tri])

#Code used to plot the Delaunay
def plot_tri(ax, points, tri):
    edges = collect_edges(tri)
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for (i,j) in edges:
        x = np.append(x, [points[i, 0], points[j, 0], np.nan])
        y = np.append(y, [points[i, 1], points[j, 1], np.nan])
        z = np.append(z, [points[i, 2], points[j, 2], np.nan])
    ax.plot3D(x, y, z, color='g', lw='0.1')

    ax.scatter(points[:,0], points[:,1], points[:,2], color='b')

def collect_edges(tri):
    edges = set()
    def sorted_tuple(a,b):
        return (a,b) if a < b else (b,a)
    #Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
    for (i0, i1, i2, i3) in tri.simplices:
        edges.add(sorted_tuple(i0,i1))
        edges.add(sorted_tuple(i0,i2))
        edges.add(sorted_tuple(i0,i3))
        edges.add(sorted_tuple(i1,i2))
        edges.add(sorted_tuple(i1,i3))
        edges.add(sorted_tuple(i2,i3))
    return edges

fig = mplot.figure()
axes = mplot3d.Axes3D(fig)
plot_tri(axes, points, tri)
mplot.show()