import os
from sys import platform
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from skspatial.objects import Plane


def ObjLegend():
    return {1: 'Walls', 2: 'Ceilings', 3: 'Floors' ,4: 'Beams' ,5:'Columns',6:'Doors',7:'Windows',8:'Pipes'}

def getShapelegend():
    return {"Plane":1 , "Cuboid":2 ,"Cylinders":3 , "Sphere":4 ,"Cone":5 , "Unidentified":0}

def Headers():
    return {"Shapeinfo": 0, "indicies": 1, "BBoxids": 2, "BBoxpoints": 3,
              "Hide": 4, "Category": 5, "Shape": 6, "Area": 7}

def revealShape(Category):
    if Category in range(1,8) and Category not in [4,5]:
        return getShapelegend()["Plane"]
    elif Category in [4,5]:
        return getShapelegend()["Cuboid"]
    else:
        return getShapelegend()["Cylinders"]


def Col(str):
    return Headers()[str]
def Obj(str):
    idx =[key for key , val in ObjLegend().items() if val == str]
    if len(idx) > 0 :
        return idx[0]
    return None

def run_connected_executable(input_path,output_path,max_point,min_dist,c,visualize):
    if platform == 'linux' or platform == 'linux2':
        os.system(f'./Executables/ConnectedGraph '+str(input_path)+' '+str(output_path)+' '+str(max_point)+' '+str(min_dist)+' '+str(c)+' '+str(visualize))
        return 0
    elif platform == 'win32' or platform == 'win64':
        os.system(f'Executables\\ConnectedGraph ' + str(input_path) + ' ' + str(output_path) + ' ' + str(max_point) + ' ' + str(min_dist) + ' ' + str(c) + ' ' + str(visualize))
        return 0
    elif platform == 'darwin':
        print("Platform is darwin / OS X")
        return 1
    else:
        print("Invaid platform found!")
        return 1

def PathCorrection(inputpath):
    if platform == 'win32' or platform == 'win64':
        return inputpath.replace("\\", "/")
    return inputpath


def exists(filepath):
    return True if os.path.exists(filepath) else False

def CheckFolderStatus(Folderpath):
    return Folderpath if os.path.exists(Folderpath) else os.makedirs(Folderpath , exist_ok=True)

def getCurrenttime(format = "%Y%m%d_%H-%M"):
    return str(datetime.now().strftime(format))

def ReadPlyFile(inputpath1 , folder= 'fusion'):

    if folder in ['fusion']:
        filename = 'fusion_'
    elif folder in ['segmentation']:
        filename = 'cleaned'
    else:
        filename = 'Img_'
    #inputpath1 = inputpath  # + os.sep + 'panoptic_segmentation/'
    plypth = [x for x in os.listdir(inputpath1 + os.sep + folder) if x.endswith('.ply') and filename in x]
    plypth = [folder + os.sep + plypth[0]] if len(plypth) > 0 else [x for x in os.listdir(inputpath1)
                                                                      if x.endswith('.ply') and 'Img_' in x]
    inputPlyfile = os.path.join(inputpath1,
                                plypth[0])
    return inputPlyfile

def ReadVerticesConnectedFiles(file_connected_path,file_vertex_path):
    read_connected_file = pd.read_csv(file_connected_path, sep='delimiter')

    vertex_all = pd.read_csv(file_vertex_path,sep=",").values

    vertex_all = np.column_stack((vertex_all[:,1:],vertex_all[:,0], np.ones(shape=len(vertex_all))))
    VID = read_connected_file['VIDs'].tolist()
    list_vertexs = [ list(map(int,(i.split(",")[1:]))) for i in VID[0:]]

    return vertex_all.round(3),list_vertexs

def AddNormalfromCloud(Vertex ,pointcloud ):
    Norms = np.array(pointcloud.normals)
    Orgpts = np.array(pointcloud.points)
    if len(Norms) != len(Vertex):
        raise IndexError
    Vertex = np.hstack((Vertex , Norms))
    return Vertex

def Planetxtread(inputfile):
    return np.loadtxt(inputfile)
