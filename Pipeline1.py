#Source : https://pylas.readthedocs.io/en/latest/examples.html
import sys
sys.path.append('/home/lucas/Gis_game/gis_proto/')

#Packages to be able to realize query on my local DB copy (service.APP is Microcosme property)
import dotenv
dotenv.load_dotenv()
from service.app import APP
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

#Wrapper for Bounding-box definition and query package (=Microcosme's property)
from game.geoutils import Bbox

#Package containing CRS transformation wrapper (=Microcosme property)
import game.geoutils as geoutils

#las/laz files reading package
import laspy as lp

#Data manipulation packages
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely as shp

# package utilized in the design of the skewness algorithm
from scipy.signal import argrelextrema
from scipy.ndimage import gaussian_filter1d

#interpolation and KDTree package used for design of the Height Factor 
from sklearn.neighbors import KDTree

# visualization package (contains DBSCAN) and interpolation package 
import open3d as o3d
from scipy.interpolate import NearestNDInterpolator



api_inst = APP(None)
crs=31370
quant_val=0.99
iter=50
r_t=20
sigma_t=1
space_db=1.1
neigh_db=5
n_neigh=15
 
#Visualization wrapper
def open3d_viz(points, without_statistical_outliers = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if not without_statistical_outliers:
        o3d.visualization.draw_geometries([pcd])
    else:
        _,cind=pcd.remove_statistical_outlier(nb_neighbors=10,std_ratio=1.0)
        o3d.visualization.draw_geometries([pcd.select_by_index(cind)])    

#GeoDataframe creation 
def df_maker(point_cloud):
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    transformer =geoutils.get_transformer(geoutils.get_CRS(crs), geoutils.mercator)
    merc_points = transformer.transform(*points.transpose())
    shp_points = [ shp.geometry.Point(x,y) for x,y in zip(merc_points[0],merc_points[1])] 
    df = gpd.GeoDataFrame({'x':merc_points[0], 'y':merc_points[1], 'z':merc_points[2],'intensity':point_cloud.intensity,'geometry':shp_points})
    return df



###Skewness algorithm ###

#Remove intensity outlier
def intensity_outliers_remover(df):
    centile=df['intensity'].quantile(quant_val)
    return(df.loc[df['intensity']<=centile])


#Compute Skewness value for each iteration
def skew_lister(pd_frame):
    sorted=pd_frame.copy()
    skew_list=np.array([pd_frame['intensity'].skew()])
    min=pd_frame['intensity'].min()
    max=pd_frame['intensity'].max()
    step=(max-min)/iter
    for i in range(iter-1):
        sorted.drop(sorted.loc[sorted['intensity']>=(max-(i+1)*step)].index,inplace=True) 
        skew_list=np.append(skew_list,sorted['intensity'].skew()) 
    return(skew_list,min,max)

#Find Maximums and minimums of the Skewness curve (not used)
def extremum_step(list):
    signal=gaussian_filter1d(list,2)
    maximums=argrelextrema(signal,np.greater)[0]
    minimums=argrelextrema(signal,np.less)[0]
    return(minimums,maximums)

#Find inflection points of the skewness curve
def inflection_points(list):
    signal=gaussian_filter1d(list,2)
    second=np.gradient(np.gradient(signal))
    infls = np.where(np.diff(np.sign(second)))[0]  
    if infls[0]==0:
        infls=np.delete(infls,0) 
    if infls[-1]==iter:
        infls=np.delete(infls,-1)
    return(infls)  

#Create an iterator where each element is a "class" with repsect to the splitting of the skewness curve we chose=cut_list (either maximums or inflections)
def classifier(df,cut_list,min,max):
    sorted=df.copy()
    step=(max-min)/iter
    cut_list=np.append(cut_list,iter)
    for i in cut_list:
        yield sorted.loc[sorted['intensity']>=(max-i*step)]
        sorted.drop(sorted.loc[sorted['intensity']>=(max-i*step)].index,inplace=True)

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
# returns a dataframe containing only points for which abs(dz)<mean+std
def heigth_index_selector(pd_frame,n_neighbors):
    points=np.dstack((pd_frame['x'],pd_frame['y'], pd_frame['z']))[0]
    heigth_factor_array=kN_heigth_distance_factor(points,n_neighbors)
    summed_array,mean,std=summed_height_dist(heigth_factor_array)
    pd_frame['neighbors height factor']=summed_array
    return_frame=pd_frame.loc[abs(pd_frame['neighbors height factor'])<=(mean+std)]
    return(return_frame,std) 

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

###First Pipeline classifier###

##Building removing preprocessing + Skewnes and Height Factor classification. (visualization at each step activated )
def prototype1(las_path,nbr_points,n_neighbors):
    with lp.open(las_path) as f:
        ##Read .laz by chunks of predefined length (here because of the recurrent encoding problem of the dataset (to be removed)).
        for point_cloud in f.chunk_iterator(nbr_points):
            ##Create dataframe and remove intensity outliers
            df=intensity_outliers_remover(df_maker(point_cloud))
            points=np.vstack((df['x'],df['y'], df['z'])).transpose()
            open3d_viz(points)
            ##Building removing##

            ##Define edge coordinate for using bbox querying of Microcosme company
            coords = [df['x'].min(),df['y'].min(),df['x'].max(),df['y'].max()]
            ##BBox query of buildings edges (not my work)
            bbox = Bbox(coords)
            buildings = api_inst.buildingMgr.get_buildings_position_df(bbox)
            ##In case there were buildings present remove points inside the defined Polygons.
            try:
                ##Create a Geodataframe with buildings as 'geometry' column.
                bld = gpd.GeoSeries.from_wkt(buildings['wkt']) 
                ##Buffering Polygon in order to suppress points around the buildings.
                bld=bld.apply(lambda x: x.buffer(2.0))
                ##Clipping 
                building_df=df.clip(bld)
                ##Define points inside the buildings Polygons.
                filter=df.within(building_df)
                ##Only keep points which are outside of it.
                building_removed_df=df.loc[~filter]
            except:
                ##If there are no buildings do nothing
                building_removed_df=df
                print("error from building remover")
            build_points=np.vstack((building_removed_df['x'],building_removed_df['y'], building_removed_df['z'])).transpose()
            open3d_viz(build_points)

            ##Skewness classifier##

            ##Define skew curve.
            skew_list,i_min,i_max=skew_lister(building_removed_df)
            ##Find inflection points.
            infls=inflection_points(skew_list)
            ##Create a list with all the class created.
            df_list=[*classifier(building_removed_df,infls,i_min,i_max)]
            ##Define the first class as the "ground" class.
            classified_df=df_list[0]
            i=1
            ##If there are to few points extend "ground" class to the next one.
            while len(classified_df)<nbr_points//r_t:
                classified_df=pd.concat([classified_df,df_list[i]])
                i=i+1
            classified_points=np.vstack((classified_df['x'],classified_df['y'], classified_df['z'])).transpose()
            open3d_viz(classified_points)

            ##Height Factor classifier##

            height_df,std=heigth_index_selector(classified_df,n_neighbors)
            ##Iterating Height Factor till threshold on standard deviation of the Height Factor distribution is reached.
            while std>sigma_t:
                height_df,std=heigth_index_selector(height_df,n_neighbors)   
            points=np.vstack((height_df['x'],height_df['y'], height_df['z'])).transpose()
            open3d_viz(points)
            yield points,coords

###Pipeline 1###

## Interpolation + DBSCAN classification + reinterpolation.## (visualization at each step activated)
def pipeline1(las_path,nbr_points,n_neighbors,vizual_bool=False):
    return_frame=pd.DataFrame({'x':[],'y':[],'z':[]})
    ##Call for the classified point of the building removing preprocessing + Skewness and Height Factor classification.
    for points,coord in prototype1(las_path,nbr_points,n_neighbors):
        ##Interpolate grid.
        grid_x,grid_y= np.mgrid[coord[0]:coord[2],coord[1]:coord[3]]
        interp = NearestNDInterpolator(points[:,:2], points[:,2])
        Z = interp(grid_x, grid_y)
        ##Clustering with DBSCAN and keep only biggest cluster.
        interpolated_points_1m = np.vstack([grid_x.ravel(), grid_y.ravel(),Z.ravel()]).transpose()
        open3d_viz(interpolated_points_1m)
        interpolated_points_1m=db_scan(interpolated_points_1m, 1.5, 5)
        open3d_viz(interpolated_points_1m)
        ##Reinterpolate grid.
        interp = NearestNDInterpolator(interpolated_points_1m[:,:2], interpolated_points_1m[:,2])
        Z = interp(grid_x, grid_y)
        ##Return obtained grid.
        interpolated_points_1m = np.vstack([grid_x.ravel(), grid_y.ravel(),Z.ravel()]).transpose()
        open3d_viz(interpolated_points_1m)
        return_frame=pd.concat([return_frame,pd.DataFrame({'x':grid_x.ravel(),'y':grid_y.ravel(),'z':Z.ravel()})],axis=0)
        break
    return return_frame



grid_dataframe=pipeline1('Philippe.laz', 500000, n_neigh)    
grid_points=np.vstack((grid_dataframe['x'],grid_dataframe['y'],grid_dataframe['z'])).transpose()
open3d_viz(grid_points)

