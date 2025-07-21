from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset
from pyproj import Transformer
import os
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from ccma import CCMA

def kitti_viewer():
    root=r"/home/asl/Muni/datasets/KITTI/Tracking"
    label_path = r"/home/asl/Muni/datasets/KITTI/Tracking/labels/training/label_02/0001.txt"
    gps_imu_path = r"/home/asl/Muni/datasets/KITTI/Tracking/GPS_IMU/training/oxts/0001.txt" # relocate this data to Training folder
    calib_data_path = r"/home/asl/Muni/datasets/KITTI/Tracking/calib/0001.txt"
    dataset = KittiTrackingDataset(root,seq_id=1,label_path=label_path) # change the sq_id here

    traffic_participant_positions_Map_all_frames = {} # key: ID, value: Positions

    # smooting the participants trajectories
    ccma = CCMA(w_ma=50, w_cc=100)

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
        # P2, V2C, points, image, labels, label_names = dataset[i]
        try:
            P2, V2C, points, labels, label_names = dataset[i]
        except Exception as e:
            print(f"Frame skipped with error: {e}")
            continue #skip the frame
        

        if labels is not None:
            mask = (label_names=="Car") #| ((label_names=="Cyclist")) | ((label_names=="Pedestrian")) | ((label_names=="Van"))
            labels = labels[mask]
            label_names = label_names[mask]
            vi.add_3D_boxes(imu_T_velo, traffic_participant_positions, labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09), show_ids=True)
            # vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1, car_model_path="/home/asl/Muni/datasets/KITTI/visualization_code/3D-Detection-Tracking-Viewer/viewer/car.obj")

        if(len(traffic_participant_positions) == 0):
            #No detections in this frame, continue to next frames
            continue

        traffic_participant_id_class_positions = traffic_participant_positions.copy() # (id, vehicle positions) present in this frame
        
        traffic_participant_positions.clear()
        for id_pos in traffic_participant_id_class_positions: # Ignore the first idx as it contains ID, 2nd idx as it contains class name
            traffic_participant_positions.append(np.asarray(id_pos[2]))

        # postions of the traffic participants in the Velodyne frame
        traffic_participant_positions = np.asarray(traffic_participant_positions) # (N, 4) ; N positions with (x,y,z,1)

        # postions of the traffic participants in the IMU+GPS frame
        traffic_participant_positions_IMU = np.linalg.inv(imu_T_velo) @ traffic_participant_positions.T # shape = (4, N) ; each column gives the 3D coordinate

        # postions of the traffic participants in the Earth frame (Global frame)
        xy = gps_coords[i]
        rpy = vehicle_rotation[i]
        rot = R.from_euler('ZYX', rpy)
        R_matrix = rot.as_matrix()

        IMU_T_Earth = np.eye(4)
        IMU_T_Earth[:3, :3] = R_matrix
        IMU_T_Earth[:, 3] = xy.T

        traffic_participant_positions_Map = (IMU_T_Earth @ traffic_participant_positions_IMU).T # (4X4) @ (4,N) = (4,N).T = (N,4)


        # in the Map dictionary, take the key = first index of traffic_participant_id_positions
        #                                 value = element of traffic_participant_positions_IMU at the same position as key
        for id_class_pos, map_pos in zip(traffic_participant_id_class_positions, traffic_participant_positions_Map):
            # id_pos is a tuple of (id, position) ; (6, [52.36093521118164, 6.31562614440918, -1.238931655883789, 1.0])
            # map_pos is a 3D position ; [4.58891002e+05 5.42865701e+06 1.12153440e+02 1.00000000e+00]
            id, class_name, lln = id_class_pos[0], id_class_pos[1], id_class_pos[2]

            # check the existence of ID
            if(id not in traffic_participant_positions_Map_all_frames.keys()):
                traffic_participant_positions_Map_all_frames[id] = {}
                if(class_name not in traffic_participant_positions_Map_all_frames[id].keys()):
                    traffic_participant_positions_Map_all_frames[id][class_name] = []
            
            traffic_participant_positions_Map_all_frames[id][class_name].append(np.asarray(map_pos[:3])) # store only (class_name, 3D coords)

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
        # print(id, " : ", pos_lst, type(pos_lst))
        for class_info, pos_lst in traffic_participant_positions_Map_all_frames[id].items(): 
            pos_arr = ccma.filter(np.asarray(pos_lst))
            traffic_participant_positions_Map_all_frames[id][class_info] = pos_arr
            # plt.plot(pos_arr[:, 0], pos_arr[:, 1], lw=2, color = np.random.rand(3,), marker = 'X') #,label=str(class_info)

    # choose diffent colors for different IDs
    for id, class_pos_lst in traffic_participant_positions_Map_all_frames.items(): # key - ID, Value = (class, position_array)
        # print(id, " : ", pos_lst, type(pos_lst))
        for class_info, pos_lst in traffic_participant_positions_Map_all_frames[id].items(): 
            plt.plot(pos_lst[:, 0], pos_lst[:, 1], lw=2, color = np.random.rand(3,), marker = 'X') #,label=str(class_in

    plt.legend()
    plt.show()

if __name__ == '__main__':
    kitti_viewer()
