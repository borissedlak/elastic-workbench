import open3d as o3d
import numpy as np
import time
import os

# Load Eagle example point cloud
eagle = o3d.data.EaglePointCloud()
pcd = o3d.io.read_point_cloud(eagle.path)

print(f"Original number of points: {np.asarray(pcd.points).shape[0]}")

# Downsample to speed up
voxel_size = 0.002
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

print(f"Downsampled number of points: {np.asarray(pcd.points).shape[0]}")

# Set colors (optional, but improves visibility)
pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red points

# Create a scene
scene = o3d.visualization.rendering.Open3DScene(o3d.visualization.rendering.OffscreenRenderer(800, 600).scene)

# Or create a new renderer first
renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)

# Setup material
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultUnlit"  # no lighting (much faster)
# material.point_size = 5.0          # FAT points

# Add point cloud to scene
scene = renderer.scene
scene.add_geometry("eagle", pcd, material)

# Set background color
scene.set_background([0, 0, 0, 1])  # RGBA black

# Setup camera
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent()
diameter = np.linalg.norm(extent)
camera = scene.camera

camera.look_at(center, center + [0, 0, 1], [0, -1, 0])  # look from z+, up is -y
camera.set_projection(60.0, 800 / 600, 0.1, 1000.0)     # fov, aspect, near, far
camera.set_zoom(0.6)

# Wait a tiny bit if needed (not necessary here)

# Render to image
image = renderer.render_to_image()

# Save to disk
output_path = "eagle_offscreen.png"
o3d.io.write_image(output_path, image)

print(f"Saved image to {output_path}")

# Clean up
renderer.release()
