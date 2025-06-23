import cv2
import numpy as np
from lxml import etree


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


def fuse_pointclouds(pc_list):
    fused_points = np.vstack(pc_list)
    return fused_points