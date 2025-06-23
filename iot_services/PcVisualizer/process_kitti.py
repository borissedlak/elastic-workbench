import argparse
import numpy as np
import open3d as o3d
import pykitti
import os

def load_point_cloud(scan):
    # scan is raw Velodyne binary file bytes
    pc = o3d.geometry.PointCloud()
    points = np.frombuffer(scan, dtype=np.float32).reshape(-1, 4)[:, :3]  # drop intensity
    pc.points = o3d.utility.Vector3dVector(points)
    return pc

def voxel_downsample(pc, voxel_size):
    if voxel_size > 0:
        return pc.voxel_down_sample(voxel_size)
    return pc

def crop_point_cloud(pc, crop_range):
    # crop_range: (xmin, xmax, ymin, ymax, zmin, zmax)
    points = np.asarray(pc.points)
    mask = (
        (points[:, 0] >= crop_range[0]) & (points[:, 0] <= crop_range[1]) &
        (points[:, 1] >= crop_range[2]) & (points[:, 1] <= crop_range[3]) &
        (points[:, 2] >= crop_range[4]) & (points[:, 2] <= crop_range[5])
    )
    pc_crop = pc.select_by_index(np.where(mask)[0])
    return pc_crop

def fuse_frames(pc_list):
    if not pc_list:
        return None
    fused = pc_list[0]
    for pc in pc_list[1:]:
        fused += pc
    # Optional: remove duplicates or downsample fused cloud here
    return fused

def main(dataset_path, sequence, voxel_size, crop_range, fuse_count, save_output):
    # Load KITTI dataset
    dataset = pykitti.raw(dataset_path, sequence)

    pc_buffer = []

    # Iterate over frames
    for i, scan in enumerate(dataset.velo):
        pc = load_point_cloud(scan)

        # Crop
        pc = crop_point_cloud(pc, crop_range)

        # Add to buffer for fusion
        pc_buffer.append(pc)

        # If buffer full, fuse frames
        if len(pc_buffer) > fuse_count:
            pc_buffer.pop(0)

        fused_pc = fuse_frames(pc_buffer)

        # Downsample fused cloud
        fused_pc = voxel_downsample(fused_pc, voxel_size)

        print(f"Frame {i} - Points after fusion and downsampling: {len(fused_pc.points)}")

        # Visualize (optional)
        o3d.visualization.draw_geometries([fused_pc])

        # Save output if requested
        if save_output:
            out_path = os.path.join("output", f"frame_{i:06d}.ply")
            os.makedirs("output", exist_ok=True)
            o3d.io.write_point_cloud(out_path, fused_pc)
            print(f"Saved fused point cloud to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KITTI Point Cloud Processor with Adjustable Parameters")
    parser.add_argument("--dataset_path", type=str, default="data")
    parser.add_argument("--sequence", type=str, default="0001")
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size for downsampling (0 to disable)")
    parser.add_argument("--crop_range", type=float, nargs=6, default=[-20, 20, -10, 10, -3, 5],
                        help="Crop box limits: xmin xmax ymin ymax zmin zmax")
    parser.add_argument("--fuse_count", type=int, default=1, help="Number of frames to fuse (1 = no fusion)")
    parser.add_argument("--save_output", action="store_true", help="Save processed clouds as PLY files")

    args = parser.parse_args()

    main(args.dataset_path, args.sequence, args.voxel_size, args.crop_range, args.fuse_count, args.save_output)
