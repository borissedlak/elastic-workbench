import argparse
import time

import cv2
import numpy as np
import pykitti
from lxml import etree

import utils


# TODO: Get dataset if not there yet
# def ensure_demo_data(self):
#     server_url = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data'
#     download_url = '{}/{}/{}.zip'.format(server_url, self.configs.foldername[:-5], self.configs.foldername)
#     download_and_unzip(self.configs.dataset_dir, download_url)
#
#     self.model = create_model(self.configs)
#     assert os.path.isfile(self.configs.pretrained_path), "No file at {}".format(self.configs.pretrained_path)
#     self.model.load_state_dict(torch.load(self.configs.pretrained_path, map_location='cpu'))
#     # print('Loaded weights from {}\n'.format(self.configs.pretrained_path))
#
#     if DEVICE_NAME == "Laptop":
#         self.configs.device = torch.device('cpu')
#     else:
#         self.configs.device = torch.device('cpu' if self.configs.no_cuda else 'cuda:{}'.format(self.configs.gpu_idx))
#     self.model = self.model.to(device=self.configs.device)
#     self.model.eval()
#
#     self.demo_dataset = Demo_KittiDataset(self.configs)

# --- Utility: Parse tracklets.xml ---
def parse_tracklets(tracklet_path):
    tree = etree.parse(tracklet_path)
    root = tree.getroot().find("tracklets")

    tracklets = []
    for item in root.findall("item"):
        object_type = item.find("objectType").text
        h = float(item.find("h").text)
        w = float(item.find("w").text)
        l = float(item.find("l").text)
        first_frame = int(item.find("first_frame").text)

        poses = []
        for pose in item.find("poses").findall("item"):
            tx = float(pose.find("tx").text)
            ty = float(pose.find("ty").text)
            tz = float(pose.find("tz").text)
            rz = float(pose.find("rz").text)
            poses.append((tx, ty, tz, rz))

        tracklets.append({
            "type": object_type,
            "size": (h, w, l),
            "first_frame": first_frame,
            "poses": poses
        })

    return tracklets


# --- Utility: Draw 3D Box (projected to 2D BEV) ---
def draw_bev_box(bev_img, pose, size, color=(0, 0, 255), max_dist=50, res=0.1):
    x, y, z, ry = pose
    h, w, l = size

    # 3D box corners in object coordinate
    corners = np.array([
        [l / 2, w / 2],
        [l / 2, -w / 2],
        [-l / 2, -w / 2],
        [-l / 2, w / 2]
    ])

    # Rotate
    rot = np.array([[np.cos(ry), -np.sin(ry)],
                    [np.sin(ry), np.cos(ry)]])
    rotated = corners @ rot.T

    # Translate
    translated = rotated + np.array([x, y])

    # Project to BEV image space
    img_coords = np.int32(translated[:, :2] / res + max_dist / res)

    # Draw polygon
    cv2.polylines(bev_img, [img_coords], isClosed=True, color=color, thickness=2)


def point_cloud_to_bev(points, max_distance=50, resolution=0.1):
    mask = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2) < max_distance
    points = points[mask]

    x = points[:, 0] / resolution
    y = points[:, 1] / resolution
    x_img = np.int32(x + (max_distance / resolution))
    y_img = np.int32(y + (max_distance / resolution))

    size = int(2 * max_distance / resolution)
    bev = np.zeros((size, size, 3), dtype=np.uint8)
    bev[y_img, x_img] = [255, 255, 255]  # white
    return bev


@utils.print_execution_time
def fuse_pointclouds(pc_list, voxel_size=0.1):
    # pc_list: list of numpy arrays (Nx4 or Nx3)
    fused_points = np.vstack(pc_list)
    # pc_o3d = o3d.geometry.PointCloud()
    # pc_o3d.points = o3d.utility.Vector3dVector(fused_points[:, :3])
    # pc_down = pc_o3d.voxel_down_sample(voxel_size=voxel_size)
    return fused_points


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='data')
    parser.add_argument('--date', type=str, default='2011_09_26')
    parser.add_argument('--drive', type=str, default='0001')
    parser.add_argument('--max_distance', type=float, default=50.0)  # [10, 100]
    parser.add_argument('--resolution', type=float, default=0.1, help="meters/pixel --> lower = more complexity")
    parser.add_argument('--tracklet_path', type=str, default='tracklet_labels.xml')
    parser.add_argument('--fusion_size', type=int, default=1)  # [1,10]
    args = parser.parse_args()

    dataset = pykitti.raw(args.base_path, args.date, args.drive)
    tracklets = parse_tracklets(args.base_path + "/" + args.date + "/" + args.tracklet_path)

    fusion_buffer = []

    for i, velo in enumerate(dataset.velo):
        start_time = time.perf_counter()

        fusion_buffer.append(velo)
        if len(fusion_buffer) > args.fusion_size:
            fusion_buffer.pop(0)

        fused_points = fuse_pointclouds(fusion_buffer)

        bev = point_cloud_to_bev(fused_points, args.max_distance, args.resolution)

        # Overlay 3D boxes
        for obj in tracklets:
            if i < obj["first_frame"] or i - obj["first_frame"] >= len(obj["poses"]):
                continue
            pose = obj["poses"][i - obj["first_frame"]]
            draw_bev_box(bev, pose, obj["size"], color=(0, 0, 255), max_dist=args.max_distance, res=args.resolution)

        cv2.imshow("LIDAR BEV with Fused Frames", bev)
        if cv2.waitKey(10) == 27:
            break

        print(f"Frame {i}, Points fused: {len(fused_points)}, Processing time: {time.perf_counter() - start_time:.3f}s")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
