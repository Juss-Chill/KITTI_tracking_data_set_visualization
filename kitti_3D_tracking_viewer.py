from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset
from pyproj import Transformer
import os
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

def kitti_viewer():
    root=r"/home/asl/Muni/datasets/KITTI/Tracking"
    label_path = r"/home/asl/Muni/datasets/KITTI/Tracking/labels/training/label_02/0000.txt"
    gps_imu_path = r"/home/asl/Muni/datasets/KITTI/Tracking/GPS_IMU/training/oxts/0000.txt" # relocate this data to Training folder
    calib_data_path = r"/home/asl/Muni/datasets/KITTI/Tracking/calib/0000.txt"
    dataset = KittiTrackingDataset(root,seq_id=0,label_path=label_path) # change the sq_id here

    traffic_participant_positions_Map_all_frames = {} # key: ID, value: Positions

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
            print(f"An error occurred: {e}")
        

        if labels is not None:
            mask = (label_names=="Car") | ((label_names=="Cyclist")) | ((label_names=="Pedestrian")) | ((label_names=="Van"))
            labels = labels[mask]
            label_names = label_names[mask]
            vi.add_3D_boxes(imu_T_velo, traffic_participant_positions, labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09), show_ids=True)
            # vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1, car_model_path="/home/asl/Muni/datasets/KITTI/visualization_code/3D-Detection-Tracking-Viewer/viewer/car.obj")

        if(len(traffic_participant_positions) == 0):
            #No detections in this frame, continue to next frames
            continue

        traffic_participant_id_positions = traffic_participant_positions.copy() # (id, vehicle positions) present in this frame
        
        traffic_participant_positions.clear()
        for id_pos in traffic_participant_id_positions: # Ignore the first idx as it contains ID
            traffic_participant_positions.append(id_pos[1])

         # postions of the traffic participants in the Velodyne frame
        traffic_participant_positions = np.asarray(traffic_participant_positions) # (N, 4) ; N positions with (x,y,z,1)
        # print("Velo : ", np.asarray(traffic_participant_positions).shape)

        # print(np.linalg.inv(imu_T_velo).shape ," , ", traffic_participant_positions.T.shape)
        # postions of the traffic participants in the IMU+GPS frame
        traffic_participant_positions_IMU = np.linalg.inv(imu_T_velo) @ traffic_participant_positions.T # shape = (4, N) ; each column gives the 3D coordinate
        # print("IMU : ", np.asarray(traffic_participant_positions_IMU).shape)

        # postions of the traffic participants in the Earth frame (Global frame)
        xy = gps_coords[i]
        rpy = vehicle_rotation[i]
        # print("XY : ", xy)
        # print("RPY: ", rpy)
        rot = R.from_euler('ZYX', rpy)
        R_matrix = rot.as_matrix()
        # print(R_matrix)
        IMU_T_Earth = np.eye(4)
        IMU_T_Earth[:3, :3] = R_matrix
        IMU_T_Earth[:, 3] = xy.T
        # print(IMU_T_Earth)
        traffic_participant_positions_Map = (IMU_T_Earth @ traffic_participant_positions_IMU).T # (4X4) @ (4,N) = (4,N).T = (N,4)
        # traffic_participant_positions_Map_all_frames.append(traffic_participant_positions_Map)
        # print("After transformation : ", traffic_participant_positions_Map.shape)
        # print(len(traffic_participant_id_positions), traffic_participant_positions_Map.shape)

        # in the Map dictionary, take the key = first index of traffic_participant_id_positions
        #                                 value = element of traffic_participant_positions_IMU at the same position as key
        for id_pos, map_pos in zip(traffic_participant_id_positions, traffic_participant_positions_Map):
            # id_pos is a tuple of (id, position) ; (6, [52.36093521118164, 6.31562614440918, -1.238931655883789, 1.0])
            # map_pos is a 3D position ; [4.58891002e+05 5.42865701e+06 1.12153440e+02 1.00000000e+00]
            

            # check the existence of ID
            if(id_pos[0] not in traffic_participant_positions_Map_all_frames.keys()):
                traffic_participant_positions_Map_all_frames[id_pos[0]] = []
            
            traffic_participant_positions_Map_all_frames[id_pos[0]].append(map_pos[:3]) # store only 3D coords

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
    plt.scatter(gps_coords[:, 0], gps_coords[:, 1])

    # choose diffent colors for different IDs
    # cmap = plt.c

    for id, pos_lst in traffic_participant_positions_Map_all_frames.items():
        # print(id, " : ", pos_lst, type(pos_lst))
        pos_arr = np.asarray(pos_lst)
        plt.plot(pos_arr[:, 0], pos_arr[:, 1], lw=2, color = np.random.rand(3,))

    plt.show()

if __name__ == '__main__':
    kitti_viewer()
