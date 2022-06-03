#Source : https://pylas.readthedocs.io/en/latest/examples.html
import sys
#sys.path.append('<path to .laz files>')
sys.path.append('/home/lucas/Gis_game/gis_proto/')

import laspy as lp
import numpy as np
import pandas as pd
from scipy.interpolate import NearestNDInterpolator

from sklearn.neighbors import KDTree


import open3d as o3d

## Module containing Wrapper method for CRS transformations (Private property of the company)
#import game.geoutils as geoutils
#EPSG code of dataset used.
#crs=31370


# Height Factor Hyperparameters
n_neigh=15
sigma_t=0.5

#DBSCAN hyperparameters
space_db=1.1
neigh_db=5

#not important
np.seterr(invalid='ignore')

#Visualization wrapper
def open3d_viz(points, without_statistical_outliers = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if not without_statistical_outliers:
        o3d.visualization.draw_geometries([pcd])
    else:
        _,cind=pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=1.0)
        o3d.visualization.draw_geometries([pcd.select_by_index(cind)])    

#Dataframe creator to use panda workflow (normally make use of Wrapper of Micrcosme's geoutils package to transform in the good CRS)
def df_maker(point_cloud):
    merc_points=np.vstack((point_cloud.x, point_cloud.y, point_cloud.z))
    #points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    #transformer =geoutils.get_transformer(geoutils.get_CRS(crs), geoutils.mercator)
    #merc_points = transformer.transform(*points.transpose())
    
    df = pd.DataFrame({'x':merc_points[0], 'y':merc_points[1], 'z':merc_points[2]})   
    return df

###Height Factor###

#Computation of difference in elevation dz_i, for each points, given its neighbors index=ind.
def dz_metric(points,ind,n_neighbors):
    dz_array=np.ndarray((len(points),n_neighbors),dtype=float)
    for j in range(dz_array.shape[1]):
        dz_array[:,j]=points[ind][:,:,2][:,j]-points[ind][:,:,2][:,0]  
    return(dz_array)    

#Neighbors search using KD-tree and partial Height Factor=dz_array/dist computation for each points.
def kN_heigth_distance_factor(points,n_neighbors):
    kdt = KDTree(points, leaf_size=2*n_neighbors, metric='euclidean')
    dist,ind=kdt.query(points, k=n_neighbors, return_distance=True)
    dz_array=dz_metric(points,ind,n_neighbors)
    heigth_distance_factor_array=np.divide(dz_array,dist)
    return(heigth_distance_factor_array)

#Computation of the total Height Factor dz for each points and computation of the mean and standard deviation (std) of the obtained distribution.
def summed_height_dist(factor_array):
    n,m=factor_array.shape
    summed_factor=np.ndarray(n,dtype=float)
    summed_factor[:]=np.nansum(factor_array,axis=1,dtype=float)
    mean,std=summed_factor.mean(dtype=float), summed_factor.std(dtype=float)
    return(summed_factor,mean,std)

#Full Height Factor classifier. Takes as input a dataframe with the point spatial coordinate 
# returns a dataframe containing only points for which abs(dz)<mean+std and where std<std_t
#This definition is build recursively.
def heigth_index_selector(pd_frame,n_neighbors):
    points=np.vstack((pd_frame['x'],pd_frame['y'], pd_frame['z'])).transpose()
    heigth_factor_array=kN_heigth_distance_factor(points,n_neighbors)
    summed_array,mean,std=summed_height_dist(heigth_factor_array)
    pd_frame['neighbors height factor']=summed_array
    if (std>sigma_t):
        return_frame=pd_frame.loc[abs(pd_frame['neighbors height factor'])<=(mean+std),:]
        return(heigth_index_selector(return_frame.loc[:, pd_frame.columns != 'neighbors height factor'],n_neighbors))
    else:
        return(pd_frame.loc[:, pd_frame.columns != 'neighbors height factor'],std)    

###DBSCAN grid's node classifier

#Moore Majority vote Algorithm for chosing biggest cluster in DBscan classifier.
def majority_vote(sequence):
    i=0
    for value in sequence:
        if i==0:
            m=value
            i=1
        elif m==value:
            i=i+1
        else:
            i=i-1
    return(m)

#DBSCAN classifier wrapper from open3d package (uncomment code to have a visualzitation of the created clusters and the kept cluster)
def db_scan(points,space,nbr_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(pcd.cluster_dbscan(eps=space, min_points=nbr_points, print_progress=True))
    #max_label = labels.max()
    #print(f"point cloud has {max_label + 1} clusters")
    #colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    #colors[labels < 0] = 0
    #pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    #o3d.visualization.draw_geometries([pcd])
    kept_label=majority_vote(labels)
    points=points[labels==kept_label]
    return(points)

###Second Pipeline classifier### : 

## MinKeep preprocessing and Height Factor classification. (Remove # for visualization)

def prototype2(las_path,nbr_points,n_neighbors):
    with lp.open(las_path) as f:
        for point_cloud in f.chunk_iterator(nbr_points):
            ##Dataframe creation.
            df=df_maker(point_cloud)
            #points=np.vstack((df['x'],df['y'], df['z'])).transpose()
            #open3d_viz(points)

            ##Definition of the sample edges for future interpolation.
            coords = [df['x'].min(),df['y'].min(),df['x'].max(),df['y'].max()] 

            ##Rounding of the (x,y) coordinates.  
            df.loc[:,('x','y')]=np.round(df.loc[:,('x','y')],0)
            #points=np.vstack((df['x'],df['y'], df['z'])).transpose()
            #open3d_viz(points)

            #Group all the points with same (x,y) coordinates.
            grp_df=df.groupby(['x','y'])  

            #Keep points with the lowest elevation for each group.
            grp_df=grp_df.agg({'z':'min'}).reset_index()
            #points=np.vstack((grp_df['x'],grp_df['y'], grp_df['z'])).transpose()
            #open3d_viz(points)
            ##Height Factor Classifier##
            height_df,std=heigth_index_selector(grp_df,n_neighbors)
            height_points=np.vstack((height_df['x'],height_df['y'], height_df['z'])).transpose()
            #open3d_viz(height_points)
            yield height_points , coords   

## Interpolation + DBSCAN classification + reinterpolation.## (visualization at each step activated)
def pipeline2(las_path,nbr_points,n_neighbors,vizual_bool=False):
    return_frame=pd.DataFrame({'x':[],'y':[],'z':[]})
    ##Call for the classified point of the building removing preprocessing + Skewness and Height Factor classification.
    for (points,coord) in prototype2(las_path,nbr_points,n_neighbors):

        ##Interpolate grid.
        grid_x,grid_y= np.mgrid[coord[0]:coord[2],coord[1]:coord[3]]
        interp = NearestNDInterpolator(points[:,:2], points[:,2])
        Z = interp(grid_x, grid_y)

        ##Clustering with DBSCAN and keep only biggest cluster.
        interpolated_points_1m = np.vstack([grid_x.ravel(), grid_y.ravel(),Z.ravel()]).transpose()
        #open3d_viz(interpolated_points_1m)
        interpolated_points_1m=db_scan(interpolated_points_1m, space_db, neigh_db)
        #open3d_viz(interpolated_points_1m)

        ##Reinterpolate grid.
        interp = NearestNDInterpolator(interpolated_points_1m[:,:2], interpolated_points_1m[:,2])
        Z = interp(grid_x, grid_y)

        ##Return obtained grid.
        interpolated_points_1m = np.vstack([grid_x.ravel(), grid_y.ravel(),Z.ravel()]).transpose()
        #open3d_viz(interpolated_points_1m)
        return_frame=pd.concat([return_frame,pd.DataFrame({'x':grid_x.ravel(),'y':grid_y.ravel(),'z':Z.ravel()})],axis=0)
        
        break

    return return_frame


###Pipeline for point cloud possessing a classification###

#create a dataframe with classification information inside  (also usualy transform coordinates into the video game's CRS)
def df_class(point_cloud):
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    #transformer =geoutils.get_transformer(geoutils.get_CRS(crs), geoutils.mercator)
    #merc_points = transformer.transform(*points.transpose())
    merc_points=points
    df = pd.DataFrame({'x':merc_points[0], 'y':merc_points[1], 'z':merc_points[2], 'class':np.array(point_cloud.classification)})
    df['class'].replace(1,'other', inplace=True)
    df['class'].replace(9,'ground', inplace=True)
    df['class'].replace(2,'ground', inplace=True)  
    return df

#Keep only ground and water points then interpolate the grid
def class_pipeline(las_path,nbr_points):
    with lp.open(las_path) as f:
        return_frame=pd.DataFrame({'x':[],'y':[],'z':[]})
        for point_cloud in f.chunk_iterator(nbr_points):
            df_test=df_class(point_cloud)
            df_ground=df_test.loc[df_test['class']=='ground']
            coord=df_ground['x'].min(),df_ground['y'].min(),df_ground['x'].max(),df_ground['y'].max()
            points=np.vstack((df_ground['x'],df_ground['y'], df_ground['z'])).transpose()
            grid_x,grid_y= np.mgrid[coord[0]:coord[2],coord[1]:coord[3]]
            interp = NearestNDInterpolator(points[:,:2], points[:,2])
            Z = interp(grid_x, grid_y)
            #interpolated_points_1m = np.vstack([grid_x.ravel(), grid_y.ravel(),Z.ravel()]).transpose()
            #open3d_viz(interpolated_points_1m)
            return_frame=pd.concat([return_frame,pd.DataFrame({'x':grid_x.ravel(),'y':grid_y.ravel(),'z':Z.ravel()})],ignore_index=True)
            break
    return(return_frame)


#Some example to play with
file4_path = 'buildings.laz'
file5_path = 'new_test.laz'
file6_path = 'last_test.laz'
file7_path = 'digue.laz'
file8_path = 'brugges.laz'
file9_path = 'crash_test.laz'
file10_path = 'test.laz'
file11_path = 'relief.laz'
file12_path = 'mignon.laz'
file13_path= 'wood_plain.laz'
file14_path= 'vrieselhof.laz'
file15_path='test_bis.laz'
file16_path='LiDAR_DHMV_2_P4_ATL12415_ES_20140110_15076_5_144500_207500.laz'
file17_path='presentation.laz'
file18_path='presentation_bis.laz'

#Block for final grid visualization.
grid_dataframe=pipeline2(file4_path, 630000, n_neigh)   
grid_points=np.vstack((grid_dataframe['x'],grid_dataframe['y'],grid_dataframe['z'])).transpose()
open3d_viz(grid_points)
