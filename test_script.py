# All the Imports

import numpy as np
import os

import time
from functools import partial
from multiprocessing import Pool
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
import matplotlib as mpl
from skimage import measure

import pymesh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import sys

sys.path.append('..')
# from datasets import *
# from transforms import *
import glob
from numpy.linalg import norm
import SimpleITK as sitk
import itk
from itkwidgets import view

import vtk
from vtk import vtkConnectivityFilter


# [STAR] All the Helper functions

# Get Centroid of all the cells
def get_cell_centroid(vtk_mesh):
    centroid_array = np.zeros([vtk_mesh.GetNumberOfCells(), 3])
    num_of_cells = vtk_mesh.GetNumberOfCells()

    for i in range(num_of_cells):
        c = vtk_mesh.GetCell(i)
        p = c.GetPoints()
        p1 = np.array(p.GetData())
        centroid_array[i] = np.mean(p1, axis=0)

    return centroid_array

# Get Normal of all the cells
def get_cell_normals(vtk_mesh):
    # Get Normals for the cells of the mesh
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(vtk_mesh)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.SplittingOff()
    normals_filter.ConsistencyOn()
    normals_filter.AutoOrientNormalsOff()
    normals_filter.Update()
    
    output1 = normals_filter.GetOutput()
    d1 = np.array(output1.GetCellData().GetNormals())
    
    return d1

def my_resample_image(input_image):
    input_size = itk.size(input_image)
    input_spacing = itk.spacing(input_image)
    input_origin = itk.origin(input_image)
    Dimension = input_image.GetImageDimension()

    output_spacing = [0.25, 0.25, 0.25]#[0.364583, 0.364583, 0.364583]

    output_size = [int(input_size[0]*input_spacing[0]/output_spacing[0]), int(input_size[1]*input_spacing[1]/output_spacing[1]), int(input_size[2]*input_spacing[2]/output_spacing[2])]
    # output_spacing = [input_spacing[d] / scale for d in range(Dimension)]
    # output_origin = [input_origin[d] + 0.5 * (output_spacing[d] - input_spacing[d])
    #                  for d in range(Dimension)]

    print('Input Values ')
    print('input_size ', input_size)
    print('input_spacing ', input_spacing)
    print('input_origin ', input_origin)
    
    output_origin = input_origin
    scale_transform = itk.ScaleTransform.D3.New()
    
    interpolator = itk.LinearInterpolateImageFunction.New(input_image)
    
    print('Type of input_image is ', type(input_image))
    
    # Using default Interpolator
    resample_filter = itk.ResampleImageFilter.IF3IF3.New()
    resample_filter.SetInput(input_image)
    resample_filter.SetTransform(scale_transform)
    resample_filter.SetDefaultPixelValue(0)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetOutputDirection(input_image.GetDirection())
    resample_filter.SetSize(output_size)
    resample_filter.Update()

    resampled = resample_filter.GetOutput()

    output_size = itk.size(resampled)
    output_spacing = itk.spacing(resampled)
    output_origin = itk.origin(resampled)
    
    print('---------------------------------')
    print('Outputs Values ')
    print('output_size ', output_size)
    print('output_spacing ', output_spacing)
    print('output_origin ', output_origin)
    
    return resampled

def get_cuberille_mesh(input_image, mesh_type):
    #input_image  = my_resample_image(input_image)
    
    InterpolatorType = itk.BSplineInterpolateImageFunction.IF3DF
    interpolator = InterpolatorType.New()
    interpolator.SetSplineOrder(3)

    cuberille = itk.CuberilleImageToMeshFilter.IF3MF3.New()
    cuberille.SetInput(input_image.astype(itk.F))
    cuberille.SetIsoSurfaceValue(0.1)
    cuberille.GenerateTriangleFacesOn()
    cuberille.ProjectVerticesToIsoSurfaceOn()
    cuberille.SetProjectVertexSurfaceDistanceThreshold(0.001)
    cuberille.SetProjectVertexMaximumNumberOfSteps(500)
    cuberille.SetProjectVertexStepLength(0.25)
    cuberille.SetInterpolator(interpolator)
    
    cuberille.Update()
    cuberille_output = cuberille.GetOutput()
    
    # Write the Mesh to obtain the Largest Connected Component using VTK
    itk.meshwrite(cuberille_output, 'cuberille_output_'+mesh_type+'.vtk')
    print('Got Cuberille Mesh')
    
    if 1:
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName('cuberille_output_'+mesh_type+'.vtk')
        reader.Update()
        vtk_mesh = reader.GetOutput()
        print('Reading of Cuberille Output Done')
        
        connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
        connectivityFilter.SetInputData(vtk_mesh)
        connectivityFilter.SetExtractionModeToAllRegions()
        connectivityFilter.Update()
        
        valid_meshes = []
        append_polydata = vtk.vtkAppendPolyData()
        
        smoothing_filter = vtk.vtkSmoothPolyDataFilter()
        smoothing_filter.SetNumberOfIterations(50) 

        for i in range(connectivityFilter.GetNumberOfExtractedRegions()):
            connectivityFilter.AddSpecifiedRegion(i)
            connectivityFilter.SetExtractionModeToSpecifiedRegions()
            connectivityFilter.Update()
            
            out1 = connectivityFilter.GetOutput()
            
            #print(i, out1.GetNumberOfCells(), out1.GetNumberOfPoints())
            
            smoothing_filter.SetInputData(out1)
            smoothing_filter.Update()
            out1 = smoothing_filter.GetOutput()

            if out1.GetNumberOfCells() > 3000:
                polydata = vtk.vtkPolyData()
                polydata.ShallowCopy(out1)
                append_polydata.AddInputData(polydata)
            
            connectivityFilter.DeleteSpecifiedRegion(i)
        
        append_polydata.Update()
        
        output_combined = append_polydata.GetOutput()

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(output_combined)
        writer.SetFileName('output_combined_'+mesh_type+'.vtk')
        writer.Update()
        print('Got Filtered Mesh with only large connected components')

        print('output_combined cells ', output_combined.GetNumberOfCells())
        print('vtk_mesh cells ', vtk_mesh.GetNumberOfCells())

        # Read the Largest Connected Component as ITK Mesh
        output = itk.meshread('output_combined_'+mesh_type+'.vtk')
        print('Reading of output_combined Done')
    
    #output = cuberille_output
    verts = np.zeros([output.GetNumberOfPoints(), 3])
    faces = np.zeros([output.GetNumberOfCells(), 3])
    
    c = output.GetCells()
    
    for i in range(faces.shape[0]):
        e = c.GetElement(i)
        pd = e.GetPointIdsContainer()
        pd_numpy = np.array(pd)
        faces[i] = pd_numpy
    
    for i in range(verts.shape[0]):
        verts[i] = output.GetPoints().ElementAt(i)
    
    return verts, faces, output, output_combined

def get_neighbors(mesh, root, element, search_range=1, edge_only=False):
    """
    get neighbors of given element(vertices/faces) on mesh
    :param mesh: a PyMesh mesh object
    :param root: id of root face/vertex
    :param element: 'face' or 'f' for searching faces, and 'vertex' or 'v' for searching vertices
    :param search_range: level of searching rings on mesh
    :param edge_only: if True, only return neighbors at the edge of search range, otherwise return all neighbors within it.
    :return:
    """
    mesh.enable_connectivity()
    to_be_visited_list = []  # searching faces
    visited_set = set()  # visited neighbor faces
    # a dictionary to maintain meta information (used for path formation)
    search_dist = dict()
    search_dist[root] = 0
    to_be_visited_list.append(root)
    while not to_be_visited_list == []:
        current_node = to_be_visited_list[0]

        if search_dist[current_node] < search_range:
            # For each child of the current tree process
            temp_neighbors = None
            if element == 'face' or element == 'f':
                temp_neighbors = mesh.get_face_adjacent_faces(current_node)
            elif element == 'vertex' or element == 'v':
                temp_neighbors = mesh.get_vertex_adjacent_vertices(current_node)
            else:
                ValueError("Wrong Element Type: can only be  'v'/'vertex' for vertex, or 'f'/'face' for face")

            for neighbor in temp_neighbors:

                # The node has already been processed, so skip over it
                # if neighbor in visited_set:
                if neighbor in search_dist.keys():
                    continue

                # The child is not enqueued to be processed, so enqueue this level of children to be expanded
                if neighbor not in to_be_visited_list:
                    # create metadata for these nodes
                    if not neighbor in search_dist.keys():
                        search_dist[neighbor] = search_dist[current_node] + 1
                    else:
                        search_dist[neighbor] = min(search_dist[neighbor], search_dist[current_node] + 1)
                    # enqueue these nodes
                    to_be_visited_list.append(neighbor)

        # We finished processing the root of this subtree, so add it to the closed set
        to_be_visited_list.remove(current_node)
        if not current_node == root:
            if (not edge_only) or search_dist[current_node] == search_range:
                visited_set.add(current_node)
        pass

    return list(visited_set)

def smooth_face_label_single_core(mesh, face_labels, smooth_rings):
    """
    Single process version of smooth the labels of all faces
    :param mesh:
    :param face_labels:
    :param smooth_rings:
    :return:
    """

    smoothed_labels = np.zeros(face_labels.shape)
    for id in range(len(face_labels)):
        neighbor_faces = get_neighbors(mesh, id, 'face', search_range=smooth_rings)
        if np.sum(face_labels[neighbor_faces]) < 0:
            smoothed_labels[id] = -1
        elif np.sum(face_labels[neighbor_faces]) > 0:
            smoothed_labels[id] = 1
        else:
            smoothed_labels[id] = face_labels[id]
    return smoothed_labels

def get_main_mesh(meshes, threshold=1, merge=True):
    """
    Get the main mesh(es) with a threhold of number of vertices from a list of meshes
    The goal is to remove small connected-component caused by segmentation artifacts
    :param meshes:
    :param threshold: meshes with number of vertices below the threshold will be removed
    :return: largest mesh(es): a list if num_max>1
    """
    assert (threshold > 1 and isinstance(threshold, int)), "threshold must be an >1 integer"

    num_vertices = np.array([meshes[i].num_vertices for i in range(len(meshes))])

    ordered_mesh_size_indices = num_vertices.argsort()[::-1]

    main_mesh_indices = []

    for ind in ordered_mesh_size_indices:
        if num_vertices[ind] < threshold:
            break
        else:
            main_mesh_indices.append(ind)

    if not merge:
        return [meshes[i] for i in main_mesh_indices]
    else:
        return pymesh.merge_meshes([meshes[i] for i in main_mesh_indices])

# Create a Mesh by selecting relevant faces only
def get_vtk_sub_mesh(input_mesh, inner_face_list):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Add Cells
    counter = 0
    point_ids_hash = {}
    for i in inner_face_list:
        tri = vtk.vtkTriangle()
        current_cell_points = input_mesh.GetCell(i).GetPointIds()
        
        if current_cell_points.GetNumberOfIds() != 3:
            print('Ids are not three Some Issue ', current_cell_points.GetNumberOfIds())
        
        for i in range(3):
            point_id = current_cell_points.GetId(i)
            
            if point_id in point_ids_hash:
                new_point_id = point_ids_hash[point_id]
            else:
                new_point_id = counter
                point_ids_hash[point_id] = new_point_id
                counter = counter + 1
            
            tri.GetPointIds().SetId(i, new_point_id)
        
        cells.InsertNextCell(tri)

    points.SetNumberOfPoints(len(list(point_ids_hash.keys())))

    # Add points
    for i in point_ids_hash:
        p = input_mesh.GetPoint(i)
        new_id = point_ids_hash[i]
        points.SetPoint(new_id, p)

    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    polydata.Modified()

    return polydata

def smooth_mesh_segmentation(mesh, face_labels, smooth_rings, max_rings=None, n_workers=1):
    """
    Spatially smooth the binary labels of face labels on a surface mesh, the smoothing is done by nearest neighbors
    The neighbors is controlled by smooth-rings which is defined as the discrete "geodesic" distance to the faces.
    E.g. smooth_rings=1 will use the faces directly connected to the target face as neighbors,
        smooth_rings=2 will add the faces directly connected to the ring1 neighbors as neighbors.
    if smooth_rings = 0, smoothing will not be excuted
    The smoothing will be repeated with +1 smooth_ring until it reachs max_rings or the segmentation is sufficiently
    smoothed (after segmentation, each connected component in the original mesh is segemented into two connected components).
    When max_ring is None, it will be set as smooth_ring and therefore only one smoothing iteration will be excuted.
    :param mesh:
    :param face_labels: The binary labeling have to be -1 or 1
    :param smooth_rings: size of smoothing rings(neighbor range on mesh graph)
    :param max_rings: if max_rings is given, the smooth_rings keeps growing until the smoothed mesh has no additional
    disconnected meshes or reaching maximum iterations
    :returns inner_mesh(label -1, surface touching bones), outer_mesh(label 1),
    inner_face_list(face indices of inner mesh), outer_face_list(face indices of outer mesh)
    """
    if smooth_rings == 0:
        inner_face_list = np.where(face_labels == -1)[0]
        outer_face_list = np.where(face_labels == 1)[0]

        inner_mesh = get_vtk_sub_mesh(mesh, inner_face_list)
        outer_mesh = get_vtk_sub_mesh(mesh, outer_face_list)
        #inner_mesh = pymesh.submesh(mesh, inner_face_list, num_rings=0)
        #outer_mesh = pymesh.submesh(mesh, outer_face_list, num_rings=0)
        return inner_mesh, outer_mesh, inner_face_list, outer_face_list

    if max_rings is None:
        max_rings = smooth_rings

    assert smooth_rings <= max_rings, "ERROR: Smoothing rings must be no more than max rings"

    mesh.enable_connectivity()

    while True:
        # mn test; todo maybe, remove again; use fix_mesh instead?
        # mesh, _ = pymesh.remove_duplicated_vertices(mesh)

        # one iteration of smoothing
        if n_workers>1:
            with Pool(processes=n_workers, initializer=mesh_process_pool_init, initargs=(mesh,)) as pool:
                smoothed_label = pool.map(partial(smooth_face_label, face_labels=face_labels, smooth_rings=smooth_rings),
                                          range(len(face_labels)))
        else:
            smoothed_label = smooth_face_label_single_core(mesh, face_labels, smooth_rings)
        smoothed_label = np.array(smoothed_label)

        # get faces with each label
        inner_face_list = np.where(smoothed_label == -1)[0]
        outer_face_list = np.where(smoothed_label == 1)[0]

        # get two segmented meshes
        inner_mesh = pymesh.submesh(mesh, inner_face_list, num_rings=0)
        outer_mesh = pymesh.submesh(mesh, outer_face_list, num_rings=0)

        # keep growing neighbor ring size for smoothing untill no artifacts
        num_mesh_components = len(pymesh.separate_mesh(mesh))
        if (len(pymesh.separate_mesh(inner_mesh)) == num_mesh_components and
                len(pymesh.separate_mesh(outer_mesh)) == num_mesh_components):
            print("Well smoothed mesh segmentation")
            break
        elif smooth_rings >= max_rings:
            print("Reach maximum smoothing rings")
            break
        else:
            print("Smooth kernal {} is small. Now using size {}".format(smooth_rings, smooth_rings + 1))
            smooth_rings += 1

        face_labels = smoothed_label

    return inner_mesh, outer_mesh, inner_face_list, outer_face_list

# Check the labels output obtained by only using face_normal and connect_direction
def split_femoral_cartilage_surface(mesh, face_normal, face_centroid, smooth_rings=1, max_rings=None, n_workers=1):
    """
    Split a cartilage surface mesh into the inner and outer surface
    :param mesh:femoral cartilage surface mesh
    :param smooth_rings:
    :param max_rings
    :return: inner_mesh(label -1, surface touching bones), outer_mesh(label 1),
    inner_face_list(face indices of inner mesh), outer_face_list(face indices of outer mesh)
    """
    
    # face normals are towards the inner of cartilage surface
    # get the center of bounding box of femoral cartilage
    (xmin,xmax,ymin,ymax,zmin,zmax)  = mesh.GetBounds()
    bbox_min = np.array([xmin, ymin, zmin])
    bbox_max = np.array([xmax, ymax, zmax])

    center = (bbox_min + bbox_max) / 2
    
    inner_outer_label_list = np.zeros(mesh.GetNumberOfCells())  # up:1, down:-1
    for k in range(mesh.GetNumberOfCells()):
    #for k in range(face_normal1.shape[0]):
        # get the direction from the center to the current face centroid
        connect_direction = center - face_centroid[k, :]

        # if the direction is the same with the normal, then it is outer side (labeled as -1) of cartilage surface
        # we only cares the direction on x-y plane
        # print(k, face_normal[k].shape, np.sign(face_normal[k]), face_normal[k])
        
        if np.dot(connect_direction[:2], face_normal[k, :2]) < 0:
            inner_outer_label_list[k] = 1
        else:
            inner_outer_label_list[k] = -1
    
    # For debugging purpose only
    # To check if the normal vector has same direction in both 
    # Cuberille Mesh and Marching Cubes Mesh
    # for k in range(mesh.num_faces):
    #     # get the direction from the center to the current face centroid
    #     connect_direction = center - face_centroid[k, :]

    #     # if the direction is the same with the normal, then it is outer side (labeled as -1) of cartilage surface
    #     # we only cares the direction on x-y plane
    #     # print(k, face_normal[k].shape, np.sign(face_normal[k]), face_normal[k])
        
    #     if np.dot(connect_direction[:2], face_normal[k, :2]) < 0:
    #         inner_outer_label_list[k] = 1
    #     else:
    #         inner_outer_label_list[k] = -1
    
    print('Count of labels in the Mesh')
    print((inner_outer_label_list == 1).sum())
    print((inner_outer_label_list == -1).sum())

    return smooth_mesh_segmentation(mesh, inner_outer_label_list, smooth_rings=smooth_rings, 
                                    max_rings=max_rings,
                                    n_workers=n_workers)

def split_tibial_cartilage_surface(mesh, face_normal, face_centroid, smooth_rings=1, max_rings=None, n_workers=1):
    """
    split femoral cartilage into two inner(touching the tibial bone) and outer surfaces by 
    clustering face normals
    :param mesh: tibial cartilage surface mesh
    :param smooth_rings:
    :param max_rings:
    :return: inner_mesh(label -1, surface touching bones), outer_mesh(label 1),
    inner_face_list(face indices of inner mesh), outer_face_list(face indices of outer mesh)
    """
    #mesh.enable_connectivity()

    #mesh.add_attribute("face_centroid")
    #mesh.add_attribute("face_normal")
    #mesh.add_attribute("vertex_gaussian_curvature")
    
    mesh_centroids = face_centroid#mesh.get_attribute('face_centroid').reshape(-1, 3)
    mesh_centroids_normalized = (mesh_centroids - np.mean(mesh_centroids, axis=0)) / \
                                (np.max(mesh_centroids, axis=0) - np.min(mesh_centroids, axis=0))

    mesh_normals = face_normal
    #mesh_normals = mesh.get_attribute('face_normal').reshape(-1, 3)
    #mesh_face_index = mesh.get_attribute('face_index').reshape(-1, 1)

    # clustering normals
    features = np.concatenate((mesh_centroids_normalized * 1, mesh_normals * 1), axis=1)
    
    #est = KMeans(n_clusters=2)
    est = KMeans(n_clusters=2, algorithm="full")
    #est = SpectralClustering(n_clusters=2)
    labels = est.fit(features).labels_

    # transfer 0/1 labels to -1/1 labels
    inner_outer_label_list = labels * 2 - 1

    # set inner surface which contains mostly positive normals
    if mesh_normals[inner_outer_label_list == -1, 1].mean() < 0:
        inner_outer_label_list = -inner_outer_label_list

    print('Count of labels in the Mesh')
    print((inner_outer_label_list == 1).sum())
    print((inner_outer_label_list == -1).sum())

    return smooth_mesh_segmentation(mesh, inner_outer_label_list, smooth_rings=smooth_rings, 
                                    max_rings=max_rings,
                                    n_workers=n_workers)

def compute_mesh_thickness(mesh, cartilage, smooth_rings=1, max_rings=None, n_workers=1):
    """
    compute the thickness from each vertex on the cartilage to the opposite surface
    :param mesh(pymesh.mesh object):
    :param cartilage(string): 'FC' femoral cartilage or 'TC' tibial cartilage
    :return:thickness at each vertex of mesh
    """
    mesh.add_attribute("vertex_index")
    # mesh.add_attribute("vertex_normal")

    # split the cartilage into inner surface that interfacing the bone and the outer surface
    if cartilage == 'FC':
        inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_tibial_cartilage_surface(mesh,
                                                                                                   smooth_rings=smooth_rings,
                                                                                                   max_rings=max_rings,
                                                                                                   n_workers=n_workers)
    elif cartilage == 'TC':
        inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_tibial_cartilage_surface(mesh,
                                                                                                  smooth_rings=smooth_rings,
                                                                                                  max_rings=max_rings,
                                                                                                  n_workers=n_workers)
    else:
        ValueError("Cartilage can only be FC or TC")

    # computer vertex distances to opposite surface
    inner_thickness = np.sqrt(pymesh.distance_to_mesh(outer_mesh, inner_mesh.vertices)[0])
    outer_thickness = np.sqrt(pymesh.distance_to_mesh(inner_mesh, outer_mesh.vertices)[0])

    # combine into a single thickness list
    thickness = np.zeros(mesh.num_vertices)
    
    print("thickness ", inner_thickness.shape, outer_thickness.shape, thickness.shape)
    
    thickness[inner_mesh.get_attribute('vertex_index').astype(int)] = inner_thickness
    thickness[outer_mesh.get_attribute('vertex_index').astype(int)] = outer_thickness
    return thickness

def get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob, spacing, thickness=True,
                                                       save_path_FC=None, save_path_TC=None,
                                                       prob=True, transform=None):
    """
    Extract cartilage mesh from segmentation (3d numpy array) and compute the distance of mesh vetices to the opposite surface
    :param segmentation: HxWxD(xC), three class segmentation (probability) map of femoral cartilage, tibial cartilage
     and background
    :param spacing: spacing of segmentation map in H,W,D
    :param prob: if the input segmentation is probability map
    :param transform: the transformation that to map the voxel coordinates (default) of vertices to world coordinates
                    if None, keep the voxel coordinates (included the spacing though),
                    otherwise it should be a tuple of two numpy arrays (R, T)
                    R is a 3x3 rotation matrix and T is the translation vetor of length 3
                    The world coordinates are computed by P_w = P_v x R + T
    :return: meshes of femoral and tibial cartilage with the additional attribute "distance":
    """

    # Use marching cubes to obtain the surface mesh of shape
    print("Extract surfaces")
    FC_verts, FC_faces, FC_normals, FC_values = measure.marching_cubes_lewiner(FC_prob, 0.5,
                                                                               spacing=spacing,
                                                                               step_size=1,
                                                                               gradient_direction="ascent")

    TC_verts, TC_faces, TC_normals, TC_values = measure.marching_cubes_lewiner(TC_prob, 0.5,
                                                                               spacing=spacing,
                                                                               step_size=1,gradient_direction="ascent")

    FC_mesh = pymesh.form_mesh(FC_verts, FC_faces)
    TC_mesh = pymesh.form_mesh(TC_verts, TC_faces)

    FC_mesh, _ = pymesh.remove_duplicated_vertices(FC_mesh)
    TC_mesh, _ = pymesh.remove_duplicated_vertices(TC_mesh)

    FC_mesh_main = get_main_mesh(pymesh.separate_mesh(FC_mesh), threshold=1000, merge=True)
    TC_mesh_main = get_main_mesh(pymesh.separate_mesh(TC_mesh), threshold=1000, merge=True)

    if thickness:
        print("Compute FC mesh thickness")
        FC_thickness = compute_mesh_thickness(FC_mesh_main, cartilage='FC', smooth_rings=10, max_rings=None,
                                              n_workers=1)
        print('Done computing FC mesh thickness')

        print("Compute TC mesh thickness")
        TC_thickness = compute_mesh_thickness(TC_mesh_main, cartilage='TC', smooth_rings=10, max_rings=None,
                                              n_workers=1)
        print('Done computing TC mesh thickness')

    if transform:
        FC_mesh_main = pymesh.form_mesh(voxel_to_world_coord(FC_mesh_main.vertices, transform),
                                        FC_mesh_main.faces)
        TC_mesh_main = pymesh.form_mesh(voxel_to_world_coord(TC_mesh_main.vertices, transform),
                                        TC_mesh_main.faces)

    if thickness:
        FC_mesh_main.add_attribute("vertex_thickness")
        FC_mesh_main.set_attribute("vertex_thickness", FC_thickness)

        TC_mesh_main.add_attribute("vertex_thickness")
        TC_mesh_main.set_attribute("vertex_thickness", TC_thickness)

        if save_path_FC:
            pymesh.save_mesh(save_path_FC, FC_mesh_main, "vertex_thickness", ascii=True)
        if save_path_TC:
            pymesh.save_mesh(save_path_TC, TC_mesh_main, "vertex_thickness", ascii=True)
    else:
        if save_path_FC:
            pymesh.save_mesh(save_path_FC, FC_mesh_main, ascii=True)
        if save_path_TC:
            pymesh.save_mesh(save_path_TC, TC_mesh_main, ascii=True)

    return FC_mesh_main, TC_mesh_main

def my_get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob, spacing, thickness=True,
                                                       save_path_FC=None, save_path_TC=None,
                                                       prob=True, transform=None):
    """
    Extract cartilage mesh from segmentation (3d numpy array) and compute the distance of mesh vetices to the opposite surface
    :param segmentation: HxWxD(xC), three class segmentation (probability) map of femoral cartilage, tibial cartilage
     and background
    :param spacing: spacing of segmentation map in H,W,D
    :param prob: if the input segmentation is probability map
    :param transform: the transformation that to map the voxel coordinates (default) of vertices to world coordinates
                    if None, keep the voxel coordinates (included the spacing though),
                    otherwise it should be a tuple of two numpy arrays (R, T)
                    R is a 3x3 rotation matrix and T is the translation vetor of length 3
                    The world coordinates are computed by P_w = P_v x R + T
    :return: meshes of femoral and tibial cartilage with the additional attribute "distance":
    """

    # Use marching cubes to obtain the surface mesh of shape
    print("Extract surfaces")
    # TODO: Replace Code here to get the Mesh using Cuberille
    FC_verts, FC_faces, FC_normals, FC_values = measure.marching_cubes_lewiner(FC_prob, 0.5,
                                                                               spacing=spacing,
                                                                               step_size=1,
                                                                               gradient_direction="ascent")

    TC_verts, TC_faces, TC_normals, TC_values = measure.marching_cubes_lewiner(TC_prob, 0.5,
                                                                               spacing=spacing,
                                                                               step_size=1,gradient_direction="ascent")

    FC_mesh = pymesh.form_mesh(FC_verts, FC_faces)
    TC_mesh = pymesh.form_mesh(TC_verts, TC_faces)

    FC_mesh, _ = pymesh.remove_duplicated_vertices(FC_mesh)
    TC_mesh, _ = pymesh.remove_duplicated_vertices(TC_mesh)

    FC_mesh_main = get_main_mesh(pymesh.separate_mesh(FC_mesh), threshold=1000, merge=True)
    TC_mesh_main = get_main_mesh(pymesh.separate_mesh(TC_mesh), threshold=1000, merge=True)

    if thickness:
        print("Compute FC mesh thickness")
        FC_thickness = compute_mesh_thickness(FC_mesh_main, cartilage='FC', smooth_rings=10, max_rings=None,
                                              n_workers=1)
        print('Done computing FC mesh thickness')

        print("Compute TC mesh thickness")
        TC_thickness = compute_mesh_thickness(TC_mesh_main, cartilage='TC', smooth_rings=10, max_rings=None,
                                              n_workers=1)
        print('Done computing TC mesh thickness')

    if transform:
        FC_mesh_main = pymesh.form_mesh(voxel_to_world_coord(FC_mesh_main.vertices, transform),
                                        FC_mesh_main.faces)
        TC_mesh_main = pymesh.form_mesh(voxel_to_world_coord(TC_mesh_main.vertices, transform),
                                        TC_mesh_main.faces)

    if thickness:
        FC_mesh_main.add_attribute("vertex_thickness")
        FC_mesh_main.set_attribute("vertex_thickness", FC_thickness)

        TC_mesh_main.add_attribute("vertex_thickness")
        TC_mesh_main.set_attribute("vertex_thickness", TC_thickness)

        if save_path_FC:
            pymesh.save_mesh(save_path_FC, FC_mesh_main, "vertex_thickness", ascii=True)
        if save_path_TC:
            pymesh.save_mesh(save_path_TC, TC_mesh_main, "vertex_thickness", ascii=True)
    else:
        if save_path_FC:
            pymesh.save_mesh(save_path_FC, FC_mesh_main, ascii=True)
        if save_path_TC:
            pymesh.save_mesh(save_path_TC, TC_mesh_main, ascii=True)

    return FC_mesh_main, TC_mesh_main

def get_cartilage_surface_mesh_from_segmentation_file(segmentation_file, thickness=True,
                                                      save_path_FC=None, save_path_TC=None,
                                                      prob=True, coord='voxel'):
    """
    compute cartilage thickness from a segmentation file
    :param segmentation_file: the image file and a tupe of seperated files of the FC/TC segmentation mask/probmaps
    :param save_path_FC:
    :param save_path_TC:
    :param prob: if True, the input segmentation is probability maps, otherwise is segmentation mask
    :param coord: the coordinate system the output mesh lie in.
                  'voxel': the image space;
                  'nifti': the world space follows convention in nifti definition, used in ITK-SNAP and NiftyReg
                  'itk': the world space follows ITK convention, used in ITK and simpleITK
    :return:
    """
    if type(segmentation_file) == str:
        segmentation = sitk.ReadImage(segmentation_file)

        # the np array from itk are ordered in z,y,x
        segmentation_np = np.swapaxes(sitk.GetArrayFromImage(segmentation), 0, 2)

        if prob:
            FC_prob = segmentation_np[:, :, :, 0]
            TC_prob = segmentation_np[:, :, :, 1]
        else:
            FC_prob = (segmentation == 1).astype(float)
            TC_prob = (segmentation == 2).astype(float)

        # get rotation matrix and translation vector for mapping from voxel space to world space
        if coord == 'voxel':
            transform = None
        elif coord == 'nifti':
            transform = get_voxel_to_world_transform_nifti(segmentation[0])

        spacing = segmentation.GetSpacing()

    elif type(segmentation_file) == tuple:
        if type(segmentation_file[0]) == str and type(segmentation_file[1]) == str:
            segmentation = [sitk.ReadImage(file) for file in segmentation_file]
            FC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[0]), 0, 2).astype(float)
            TC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[1]), 0, 2).astype(float)

            # the np array from itk are ordered in z,y,x
            if coord == 'voxel':
                transform = None
            elif coord == 'nifti':
                transform = get_voxel_to_world_transform_nifti(segmentation[0])
            spacing = segmentation[0].GetSpacing()
        else:
            TypeError("The segmentation files must be a tuple of strings, but a tuple of ({}, {}) is given".format(
                type(segmentation_file[0]), type(segmentation_file[1])))

    else:
        TypeError("The segmentation file must be a str type or a tuple of strings, but {} is given".format(type(segmentation_file)))


    return get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob,
                                                              spacing=spacing,
                                                              thickness=thickness,
                                                              save_path_TC=save_path_TC,
                                                              save_path_FC=save_path_FC,
                                                              prob=prob,
                                                              transform=transform)

def my_get_cartilage_surface_mesh_from_segmentation_file(segmentation_file, thickness=True,
                                                      save_path_FC=None, save_path_TC=None,
                                                      prob=True, coord='voxel'):
    """
    compute cartilage thickness from a segmentation file
    :param segmentation_file: the image file and a tupe of seperated files of the FC/TC segmentation mask/probmaps
    :param save_path_FC:
    :param save_path_TC:
    :param prob: if True, the input segmentation is probability maps, otherwise is segmentation mask
    :param coord: the coordinate system the output mesh lie in.
                  'voxel': the image space;
                  'nifti': the world space follows convention in nifti definition, used in ITK-SNAP and NiftyReg
                  'itk': the world space follows ITK convention, used in ITK and simpleITK
    :return:
    """
    if type(segmentation_file) == str:
        segmentation = sitk.ReadImage(segmentation_file)

        # the np array from itk are ordered in z,y,x
        segmentation_np = np.swapaxes(sitk.GetArrayFromImage(segmentation), 0, 2)

        if prob:
            FC_prob = segmentation_np[:, :, :, 0]
            TC_prob = segmentation_np[:, :, :, 1]
        else:
            FC_prob = (segmentation == 1).astype(float)
            TC_prob = (segmentation == 2).astype(float)

        # get rotation matrix and translation vector for mapping from voxel space to world space
        if coord == 'voxel':
            transform = None
        elif coord == 'nifti':
            transform = get_voxel_to_world_transform_nifti(segmentation[0])

        spacing = segmentation.GetSpacing()

    elif type(segmentation_file) == tuple:
        if type(segmentation_file[0]) == str and type(segmentation_file[1]) == str:
            segmentation = [sitk.ReadImage(file) for file in segmentation_file]
            FC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[0]), 0, 2).astype(float)
            TC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[1]), 0, 2).astype(float)

            # the np array from itk are ordered in z,y,x
            if coord == 'voxel':
                transform = None
            elif coord == 'nifti':
                transform = get_voxel_to_world_transform_nifti(segmentation[0])
            spacing = segmentation[0].GetSpacing()
        else:
            TypeError("The segmentation files must be a tuple of strings, but a tuple of ({}, {}) is given".format(
                type(segmentation_file[0]), type(segmentation_file[1])))

    else:
        TypeError("The segmentation file must be a str type or a tuple of strings, but {} is given".format(type(segmentation_file)))


    return my_get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob,
                                                              spacing=spacing,
                                                              thickness=thickness,
                                                              save_path_TC=save_path_TC,
                                                              save_path_FC=save_path_FC,
                                                              prob=prob,
                                                              transform=transform)

def plot_mesh_segmentation(mesh1, mesh2):
    import visvis as vv

    app = vv.use()
    a1 = vv.subplot(111)
    
    FC_vis_up = vv.mesh(mesh1.vertices, mesh1.faces)
    FC_vis_up.faceColor = 'r'
    
    FC_vis_down = vv.mesh(mesh2.vertices, mesh2.faces)
    FC_vis_down.faceColor = 'b'
    
    app.Run()

def get_vtk_mesh(verts, faces):
    points = vtk.vtkPoints()
    cells = vtk.vtkCellArray()

    # Add Cells
    point_ids_hash = {}
    for i in range(0, faces.shape[0]):
        tri = vtk.vtkTriangle()
        current_cell_points = faces[i]
        
        for i in range(3):
            point_id = current_cell_points[i]
            tri.GetPointIds().SetId(i, point_id)
            point_ids_hash[point_id] = 1
        
        cells.InsertNextCell(tri)

    points.SetNumberOfPoints(len(list(point_ids_hash.keys())))

    # Add points
    for i in point_ids_hash:
        p = verts[i]
        points.SetPoint(i, p)

    # Create a poly data object
    vtk_mesh = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    vtk_mesh.SetPoints(points)
    vtk_mesh.SetPolys(cells)
    vtk_mesh.Modified()

    connectivityFilter = vtk.vtkPolyDataConnectivityFilter()
    connectivityFilter.SetInputData(vtk_mesh)
    connectivityFilter.SetExtractionModeToAllRegions()
    connectivityFilter.Update()
        
    valid_meshes = []
    append_polydata = vtk.vtkAppendPolyData()
    
    smoothing_filter = vtk.vtkSmoothPolyDataFilter()
    smoothing_filter.SetNumberOfIterations(50) 

    for i in range(connectivityFilter.GetNumberOfExtractedRegions()):
        connectivityFilter.AddSpecifiedRegion(i)
        connectivityFilter.SetExtractionModeToSpecifiedRegions()
        connectivityFilter.Update()
        
        out1 = connectivityFilter.GetOutput()
        
        smoothing_filter.SetInputData(out1)
        smoothing_filter.Update()
        out1 = smoothing_filter.GetOutput()

        if out1.GetNumberOfCells() > 3000:
            polydata = vtk.vtkPolyData()
            polydata.ShallowCopy(out1)
            append_polydata.AddInputData(polydata)
        
        connectivityFilter.DeleteSpecifiedRegion(i)
    
    append_polydata.Update()
    output_combined = append_polydata.GetOutput()

    return output_combined

# [STAR] Code For Testing the split_femoral_cartilage_surface method

#fc_prob_file = '/media/pranjal.sahu/cde12877-34df-449d-8202-07ba08ef2e6e/OLD/DATASETS/example_oai_data/example_data/OAI_results/9010060/MR_SAG_3D_DESS/LEFT_KNEE/12_MONTH/FC_probmap.nii.gz'
#tc_prob_file = '/media/pranjal.sahu/cde12877-34df-449d-8202-07ba08ef2e6e/OLD/DATASETS/example_oai_data/example_data/OAI_results/9010060/MR_SAG_3D_DESS/LEFT_KNEE/12_MONTH/TC_probmap.nii.gz'


fc_prob_file = '/mnt/newdrive/OLD/DATASETS/example_oai_data/example_data/OAI_results/9010060/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/FC_probmap.nii.gz'
tc_prob_file = '/mnt/newdrive/OLD/DATASETS/example_oai_data/example_data/OAI_results/9010060/MR_SAG_3D_DESS/LEFT_KNEE/72_MONTH/TC_probmap.nii.gz'

start_time  = time.time()
#fc_prob_file = '/mnt/newdrive/OLD/DATASETS/example_oai_data/example_data/OAI_results/9010060/MR_SAG_3D_DESS/RIGHT_KNEE/24_MONTH/FC_probmap.nii.gz'
#tc_prob_file = '/mnt/newdrive/OLD/DATASETS/example_oai_data/example_data/OAI_results/9010060/MR_SAG_3D_DESS/RIGHT_KNEE/24_MONTH/TC_probmap.nii.gz'


segmentation_file =  (fc_prob_file, tc_prob_file)

segmentation = [sitk.ReadImage(file) for file in segmentation_file]

FC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[0]), 0, 2).astype(float)
TC_prob = np.swapaxes(sitk.GetArrayFromImage(segmentation[1]), 0, 2).astype(float)

coord = 'voxel'

# the np array from itk are ordered in z,y,x
if coord == 'voxel':
    transform = None
elif coord == 'nifti':
    transform = get_voxel_to_world_transform_nifti(segmentation[0])
spacing = segmentation[0].GetSpacing()


thickness = True
prob = True

print("Extract surfaces")

FC_prob_img = itk.imread(segmentation_file[0], itk.F)
TC_prob_img = itk.imread(segmentation_file[1], itk.F)

FC_verts, FC_faces, FC_normals, FC_values = measure.marching_cubes_lewiner(FC_prob, 0.2,
                                                                               spacing=spacing,
                                                                               step_size=1, 
                                                                               gradient_direction="ascent")
TC_verts, TC_faces, TC_normals, TC_values = measure.marching_cubes_lewiner(TC_prob, 0.2,
                                                                            spacing=spacing,
                                                                            step_size=1, 
                                                                            gradient_direction="ascent")
                                                                            
#FC_verts, FC_faces, FC_itk_mesh, FC_vtk_mesh = get_cuberille_mesh(FC_prob_img, 'FC')
#TC_verts, TC_faces, TC_itk_mesh, TC_vtk_mesh = get_cuberille_mesh(TC_prob_img, 'TC')

#print('Types of objects are ')
#print(type(FC_verts), type(FC_faces), type(FC_itk_mesh), type(FC_vtk_mesh))

print('Got the Mesh. Now constructing PyMesh for further computation')

#FC_mesh = pymesh.form_mesh(FC_verts, FC_faces)
#TC_mesh = pymesh.form_mesh(TC_verts, TC_faces)

FC_mesh = get_vtk_mesh(FC_verts, FC_faces)
TC_mesh = get_vtk_mesh(TC_verts, TC_faces)

FC_mesh_main = FC_mesh
TC_mesh_main = TC_mesh

FC_mesh_cell_normals = get_cell_normals(FC_mesh)
TC_mesh_cell_normals = get_cell_normals(TC_mesh)

FC_mesh_cell_centroids = get_cell_centroid(FC_mesh)
TC_mesh_cell_centroids = get_cell_centroid(TC_mesh)

print(FC_mesh_cell_normals.shape)
print(TC_mesh_cell_normals.shape)
print(FC_mesh_cell_centroids.shape)
print(TC_mesh_cell_centroids.shape)


current_mesh = FC_mesh_main
current_face_normals = FC_mesh_cell_normals
current_face_centroids = FC_mesh_cell_centroids

if 1:
    smooth_rings = 0
    max_rings = None
    inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_femoral_cartilage_surface(current_mesh,
                                                                                            current_face_normals,
                                                                                            current_face_centroids,
                                                                                            smooth_rings=smooth_rings,
                                                                                            max_rings=max_rings,
                                                                                            n_workers=1)
    print('Types of inner_mesh, outer_mesh ')
    print(type(inner_mesh), type(outer_mesh), type(inner_face_list), type(outer_face_list), inner_face_list.shape)

    # For printing the points in the faces of the Mesh
    #TC_mesh_main_faces = current_mesh.faces
    #TC_mesh_main_vertices = current_mesh.vertices

    end_time  = time.time()
    print('Elapsed Time ', end_time - start_time)

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileVersion(42)

    writer.SetInputData(inner_mesh)
    writer.SetFileName('inner_mesh.vtk')
    writer.Update()
    
    writer.SetInputData(outer_mesh)
    writer.SetFileName('outer_mesh.vtk')
    writer.Update()

    # For getting the distance of meshes
    distance_filter = vtk.vtkDistancePolyDataFilter()
    distance_filter.SetInputData(0, inner_mesh)
    distance_filter.SetInputData(1, outer_mesh)
    distance_filter.SignedDistanceOff()
    distance_filter.SetComputeSecondDistance(True)
    distance_filter.Update()
    
    distance_inner = distance_filter.GetOutput()
    distance_outer = distance_filter.GetSecondDistanceOutput ()

    writer.SetInputData(distance_inner)
    writer.SetFileName('distance_inner.vtk')
    writer.Update()

    writer.SetInputData(distance_outer)
    writer.SetFileName('distance_outer.vtk')
    writer.Update()

    print(inner_mesh.GetNumberOfPoints(), distance_inner.GetNumberOfPoints())
    print(outer_mesh.GetNumberOfPoints(), distance_outer.GetNumberOfPoints())

    #print('Distance output type is ', type(distance_output))
    #print('Distance output is ', distance_output)
    
    #view(geometries=[inner_mesh, outer_mesh])

    #plot_mesh_segmentation(inner_mesh, outer_mesh)
    # import visvis as vv
    # mesh1 = selected_vtk_mesh
    # app = vv.use()
    # a1 = vv.subplot(111)
    # FC_vis_up = vv.mesh(mesh1.vertices, mesh1.faces)#, values=FC_thickness)
    # FC_vis_up.colormap = vv.CM_JET
    # app.Run()

else:
    smooth_rings = 1
    FC_thickness = compute_mesh_thickness(FC_mesh_main, 
                                      cartilage='FC', 
                                      smooth_rings=smooth_rings, 
                                      max_rings=None,
                                      n_workers=1)

    end_time  = time.time()
    print('Elapsed Time ', end_time - start_time)
    import visvis as vv
    mesh1 = FC_mesh_main
    app = vv.use()
    a1 = vv.subplot(111)
    FC_vis_up = vv.mesh(mesh1.vertices, mesh1.faces, values=FC_thickness)
    FC_vis_up.colormap = vv.CM_JET
    app.Run()

# smooth_rings = 5
# max_rings = None
# inner_mesh, outer_mesh, inner_face_list, outer_face_list = split_tibial_cartilage_surface(TC_mesh_main,
#                                                                                           smooth_rings=smooth_rings,
#                                                                                           max_rings=max_rings,
#                                                                                           n_workers=1)
