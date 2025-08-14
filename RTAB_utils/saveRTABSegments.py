import os
import pickle

from RTAB_utils.ios_rtab import RTAB2Cache



def rtabSegments(input_data_path, rgb_dir, depth_dir, pose_file, outputdatapath=None):

    stopf = len(os.listdir(rgb_dir))
    startf = 0
    stepf = 1
    subfilename = str(startf) + "_" + str(stopf) + "_" + str(stepf)
    os.makedirs(os.path.join(input_data_path, "PointcloudMergeResults"), exist_ok=True)
    os.makedirs(os.path.join(input_data_path, "PointcloudMergeResults","Segments_"+subfilename), exist_ok=True)
    
    tofpath = os.path.join(input_data_path, "PointcloudMergeResults", f"tofcameradata_{subfilename}.pkl")
    tof_seg_path = os.path.join(input_data_path, "PointcloudMergeResults", f"tofsegment_{subfilename}.pkl")
    rtspath = os.path.join(input_data_path, "PointcloudMergeResults", f"rtscameradata_{subfilename}.pkl")

    try:
        conversion = RTAB2Cache(input_data_path, rgb_dir, depth_dir, pose_file, startf, stopf, stepf, padding = True)
        tofCameraData, rtsCameraData, tofCameraModPoints, tofCameraOrgPoints = conversion.getTofCameraData(image_depth=False)
    except:
        raise("Error getting conversion data, problem with dataset")
    
    tofCameraDataSeg = []
    for i in range(len(tofCameraModPoints)):
        file_path = {}
        tof_seg_file = os.path.join(input_data_path, "PointcloudMergeResults","Segments_"+subfilename, f"tofcameradata_segments_{subfilename}_"+str(i)+".pkl")
        with open(tof_seg_file ,'wb') as f:
            pickle.dump(tofCameraData[i], f)
        frame_data = {}
        frame_data["frameNumber"] = tofCameraData[i]["frameNumber"]
        frame_data["fileName"] = os.path.join("PointcloudMergeResults","Segments_"+subfilename, f"tofcameradata_segments_{subfilename}_"+str(i)+".pkl")
        tofCameraDataSeg.append(frame_data)
        print("frames completed = ",i)
        
    with open(tof_seg_path ,'wb') as f:
        pickle.dump(tofCameraDataSeg, f) 
        

    with open(tofpath ,'wb') as f:
        pickle.dump(tofCameraData, f)
    with open(rtspath ,'wb') as f:
        pickle.dump(rtsCameraData, f)

if __name__ == "__main__":
    input_data_path = "../test_data/rtab"
    try:
        rtabSegments(input_data_path)
    except:
        raise