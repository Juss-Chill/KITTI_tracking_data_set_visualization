from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset
from pyproj import Transformer
import os
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from ccma import CCMA

import pandas as pd

def kitti_viewer(file_id):
    root=r"/home/asl/Muni/datasets/KITTI/Tracking"
    label_path = r"/home/asl/Muni/datasets/KITTI/Tracking/labels/training/label_02/" + str(file_id) + ".txt"
    gps_imu_path = r"/home/asl/Muni/datasets/KITTI/Tracking/GPS_IMU/training/oxts/" + str(file_id) + ".txt" # relocate this data to Training folder
    calib_data_path = r"/home/asl/Muni/datasets/KITTI/Tracking/calib/" + str(file_id) + ".txt"

    # Process the path to get the sequence number make it a file name
    seq_id = label_path.split(sep='/')[-1].split(sep='.')[0]
    res_path=r"/home/asl/Muni/datasets/KITTI/Tracking/dataset/" + f"{seq_id}.csv"
    dataset = KittiTrackingDataset(root,seq_id=seq_id[-1],label_path=label_path) # change the sq_id here


    traffic_participant_positions_Map_all_frames = {} # key: ID, value: Positions
    traffic_participant_dimensions_dict = {}               # key: ID, value: Dimensions of detected objects
    ccma = CCMA(w_ma=50, w_cc=100)                    # smooting the participants trajectories

    vi = Viewer(box_type="Kitti")
    gps_coords, vehicle_rotation = vi.get_gps_coords(file_path=gps_imu_path)
    # print(gps_coords.shape)
    imu_T_velo = vi.get_calib_data(file_path=calib_data_path, key="Tr_imu_velo").reshape(3,4)
    # handle the post processing of the data here
    imu_T_velo = np.vstack((imu_T_velo, np.asarray([0., 0., 0., 1.])))
    gps_coords = np.hstack((gps_coords, np.ones((gps_coords.shape[0], 1)))) #(N, 4)

    # print(vehicle_rotation.shape)
    # print(gps_coords.shape, len(dataset))

    # print(imu_T_velo)
    # print("===================")
    # print(np.linalg.inv(imu_T_velo))
    
    for i in range(len(dataset)):
        print("Frame : ", i)
        traffic_participant_positions = [] # move it above to store the data across all the frames
        traffic_participant_orientation = []
        traffic_participant_dimensions = []
        # P2, V2C, points, image, labels, label_names = dataset[i]
        try:
            P2, V2C, points, labels, label_names = dataset[i]
        except Exception as e:
            print(f"Frame skipped with error: {e}")
            continue #skip the frame
        

        if labels is not None:
            mask = (label_names=="Car") | ((label_names=="Cyclist")) | ((label_names=="Pedestrian")) | ((label_names=="Van"))
            labels = labels[mask]
            label_names = label_names[mask]
            vi.add_3D_boxes(imu_T_velo, traffic_participant_positions, traffic_participant_dimensions, labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09), show_ids=True)
            # vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1, car_model_path="/home/asl/Muni/datasets/KITTI/visualization_code/3D-Detection-Tracking-Viewer/viewer/car.obj")

        if((len(traffic_participant_positions) == 0) | (len(traffic_participant_dimensions) == 0)):
            #No detections in this frame, continue to next frames
            continue

        assert (len(traffic_participant_positions) == 0) == (len(traffic_participant_dimensions) == 0), "Mismatch in number of Bboxes and number of (x,y,z)'s"

        traffic_participant_id_class_positions = traffic_participant_positions.copy() # (id, vehicle positions) present in this frame
        
        traffic_participant_positions.clear()
        traffic_participant_orientation.clear()
        for id_pos in traffic_participant_id_class_positions:           # 0th idx has ID, 1st idx has class name
            traffic_participant_positions.append(np.asarray(id_pos[2])) # 2nd idx has (x,y,z) in mts, 3rd idx has heading_in_rads
            traffic_participant_orientation.append(np.asarray([0., 0., id_pos[3]])) # assuming Roll = Pitch = 0

        # postions of the traffic participants in the Velodyne frame
        traffic_participant_positions = np.asarray(traffic_participant_positions)       # (N, 3) ; N positions with (x,y,z)
        traffic_participant_orientation = np.asarray(traffic_participant_orientation)   #(N, 3); N orientatins with (roll:0, pitch:0, yaw)

        # postions of the traffic participants in the Earth frame (Global frame)
        xy = gps_coords[i]
        rpy = vehicle_rotation[i]
        rot = R.from_euler('ZYX', rpy)
        R_matrix = rot.as_matrix()

        IMU_T_Earth = np.eye(4)
        IMU_T_Earth[:3, :3] = R_matrix
        IMU_T_Earth[:, 3] = xy.T

        VELO_T_EARTH = IMU_T_Earth @ np.linalg.inv(imu_T_velo) # Matrix that transforms data form Velo --> Earth frame
        assert traffic_participant_positions.shape == traffic_participant_orientation.shape, "number of elements mismatch"
        
        traffic_participants_pose_Map = []
        for i in range(len(traffic_participant_positions)):
            # construct the object matrices in velo frame
            traffic_ele_pose = np.eye(4)
            traffic_ele_pose[:3, 3] = traffic_participant_positions[i].T
            yaw = traffic_participant_orientation[i][2] # stored in (R, P, Y) format
            traffic_ele_pose[:3, :3] = R.from_euler('zyx', [yaw, 0, 0]).as_matrix() # This is the object's frame i.e, [R | T] of the object in the velo frame assuming Roll = pitch = 0
            pose_in_Earth_matrix = VELO_T_EARTH @ traffic_ele_pose # [R | T] in the earth frame

            # Extract the position
            pos_global = pose_in_Earth_matrix[:3, 3]
            # Extract orientation
            roll, pitch, heading = R.from_matrix(pose_in_Earth_matrix[:3, :3]).as_euler('zyx')
            traffic_participants_pose_Map.append([pos_global[0], pos_global[1], pos_global[2], heading]) # store the (x,y,z,heading) in Earth frame


        # in the Map dictionary, take the key = first index of traffic_participant_id_positions
        #                                 value = element of traffic_participant_positions_IMU at the same position as key
        for id_class_pos, map_pos, bbox_dim in zip(traffic_participant_id_class_positions, traffic_participants_pose_Map, traffic_participant_dimensions):
            # id_pos is a tuple of (id, position_heading) ; (6, [52.36093521118164, 6.31562614440918, -1.238931655883789, heading_in_rad])
            # map_pos is a 3D position+heading_in_rad ; [4.58891002e+05 5.42865701e+06 1.12153440e+02 heading_in_rad]
            id, class_name, lln = id_class_pos[0], id_class_pos[1], id_class_pos[2]

            # check the existence of ID
            if(id not in traffic_participant_positions_Map_all_frames.keys()):
                traffic_participant_positions_Map_all_frames[id] = {}
                traffic_participant_dimensions_dict[id] = []
                if(class_name not in traffic_participant_positions_Map_all_frames[id].keys()):
                    traffic_participant_positions_Map_all_frames[id][class_name] = []
            
            traffic_participant_positions_Map_all_frames[id][class_name].append(np.asarray(map_pos)) # store only (class_name, 3D coords+heading_in_rad)
            traffic_participant_dimensions_dict[id].append(np.asarray([bbox_dim[0], bbox_dim[1], bbox_dim[2]]))


        vi.add_points(points[:,:3])

        # vi.add_imtraffic_participant_id_positionsage(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)

        # vi.show_2D()

        # vi.show_3D()
        # plt.scatter(gps_coords[:, 0], gps_coords[:, 1])
        # plt.scatter(traffic_participant_positions_Map[:,0],traffic_participant_positions_Map[:,1], color='r', marker = '*')
        # plt.show()
    
  

    #  plot GPS coords along with vehicle info
    plt.scatter(gps_coords[:, 0], gps_coords[:, 1], label="Ego position")
    plt.scatter(gps_coords[0,0], gps_coords[0,1],s=50, marker='D', c='red', label="start Ego position") # start location
    plt.scatter(gps_coords[-1,0], gps_coords[-1,1],s=50, marker='D', c='g', label="End Ego position") # end location

    # Preform Trajectory smooting using CCMA to reduce the noise
    for id, class_pos_lst in traffic_participant_positions_Map_all_frames.items(): # key - ID, Value = (class, position_array)
        # print(id, " : ", type(class_pos_lst)) # type of dict
        for class_info, pos_lst in traffic_participant_positions_Map_all_frames[id].items(): 
            # print(np.asarray(pos_lst).shape) # (N,4) each ID's path information with (x,y,z,heading_in_rad)
            pos_arr = np.asarray(pos_lst)
            
            try:
                pos_arr[:, :3] = ccma.filter(pos_arr[:, :3]) # pass (x,y,z) for smooting
            except:
                print("CCMA failed")

            traffic_participant_positions_Map_all_frames[id][class_info] = pos_arr
            # plt.plot(pos_arr[:, 0], pos_arr[:, 1], lw=2, color = np.random.rand(3,), marker = 'X') #,label=str(class_info)

    # Parase the data into the Pandas DF to write it to CSV
    data_dict = []
    # choose diffent colors for different IDs
    for id, class_pos_arr in traffic_participant_positions_Map_all_frames.items(): # key - ID, Value = (class, position_array)
        # print(id, " : ", pos_lst, type(pos_lst))
        for class_info, pos_arr in traffic_participant_positions_Map_all_frames[id].items(): 
            x = pos_arr[:, 0]
            y = pos_arr[:, 1]
            angles = pos_arr[:, 3]
            # Compute arrow directions (unit vectors)
            dx = np.cos(angles)
            dy = np.sin(angles)
            plt.plot(pos_arr[:, 0], pos_arr[:, 1], lw=2, color = np.random.rand(3,), marker = 'X') #,label=str(class_in
            plt.scatter(pos_arr[0,0], pos_arr[0,1],s=100, marker='D', c='red') # start location
            plt.scatter(pos_arr[-1,0], pos_arr[-1,1],s=100, marker='D', c='g') # end location
            # print(pos_arr.shape)
            # Plot heading arrows
            plt.quiver(
                x, y, dx, dy, 
                angles='xy', scale_units='xy', scale=1.5, 
                color='black', width=0.005
            )

            dims_arr = np.asarray(traffic_participant_dimensions_dict[id])
            data_dict.append({ "Track_ID":id, "vehicle_class":str(class_info), "path(x,y,z,heading)":pos_arr, "bbox_dims(lwh)":dims_arr})

    # convert to data frame and store it in CSV
    pd.DataFrame(data_dict).to_csv(res_path, index=False)
    # print(df.head())
    plt.legend()
    plt.show()

if __name__ == '__main__':
    files  =os.listdir("/home/asl/Muni/datasets/KITTI/Tracking/velodyne/")
    files = sorted(files)
    for file_name in files:
        print(f"========================= file : {file_name} ===========================================================")
        kitti_viewer(file_name)
