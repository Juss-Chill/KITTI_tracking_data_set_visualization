from viewer.viewer import Viewer
import numpy as np
from dataset.kitti_dataset import KittiTrackingDataset

def kitti_viewer():
    root=r"/home/asl/Muni/datasets/KITTI/Tracking"
    label_path = r"/home/asl/Muni/datasets/KITTI/Tracking/labels/training/label_02/0001.txt"
    gps_imu_path = r"/home/asl/Muni/datasets/KITTI/data_tracking_oxts_GPS_IMU/training/oxts/0000.txt"
    dataset = KittiTrackingDataset(root,seq_id=1,label_path=label_path)

    vi = Viewer(box_type="Kitti")

    for i in range(len(dataset)):
        # P2, V2C, points, image, labels, label_names = dataset[i]
        P2, V2C, points, labels, label_names = dataset[i]


        if labels is not None:
            mask = (label_names=="Car") | ((label_names=="Cyclist")) | ((label_names=="Pedestrian")) | ((label_names=="Van"))
            labels = labels[mask]
            label_names = label_names[mask]
            vi.add_3D_boxes(labels, ids=labels[:, -1].astype(int), box_info=label_names,caption_size=(0.09,0.09), show_ids=True)
            # vi.add_3D_cars(labels, ids=labels[:, -1].astype(int), mesh_alpha=1, car_model_path="/home/asl/Muni/datasets/KITTI/visualization_code/3D-Detection-Tracking-Viewer/viewer/car.obj")
        vi.add_points(points[:,:3])

        # vi.add_image(image)
        vi.set_extrinsic_mat(V2C)
        vi.set_intrinsic_mat(P2)

        # vi.show_2D()

        vi.show_3D()


if __name__ == '__main__':
    kitti_viewer()
