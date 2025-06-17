import logging

import open3d as o3d
import numpy as np
import time

import utils

logging.basicConfig(level=logging.INFO)

# Load Eagle example point cloud
eagle = o3d.data.EaglePointCloud()
pcd_1 = o3d.io.read_point_cloud(eagle.path)

# Downsample to speed up
pcd_2 = pcd_1.voxel_down_sample(voxel_size=0.015)

pcd_3 = pcd_1.voxel_down_sample(voxel_size=0.03)

time.sleep(0.1)

@utils.print_execution_time
def export_img(data):
    # Set colors (optional, but improves visibility)
    # pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red points

    # Create OffscreenRenderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(800, 600)

    # Get the scene from the renderer
    scene = renderer.scene

    # Setup material for the point cloud
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"  # no lighting for faster rendering
    # material.point_size = 5.0         # Make points bigger

    # Add the point cloud to the scene
    scene.add_geometry("eagle", data, material)

    # Set the background color (black in this case)
    scene.set_background([0, 0, 0, 1])  # RGBA black

    # Setup camera view (for visualization)
    bounds = data.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    # extent = bounds.get_extent()
    # diameter = np.linalg.norm(extent)
    camera = scene.camera

    # Camera parameters
    camera.look_at(center, center + [0, 0, 1], [0, -1, 0])  # Look at the center

    # Perspective projection using the correct method
    camera.set_projection(
        field_of_view=1000.0,  # Field of view in degrees
        aspect_ratio=800 / 600,  # Aspect ratio
        near_plane=0.1,  # Near plane
        far_plane=1000.0,  # Far plane
        field_of_view_type=o3d.visualization.rendering.Camera.FovType.Horizontal  # Specify the FOV type
    )

    # Render the scene to an image
    image = renderer.render_to_image()

    # Save the image to disk
    output_path = f"eagle_offscreen_{time.time()}.png"
    o3d.io.write_image(output_path, image)

    print(f"Saved image to {output_path}")

export_img(pcd_1)
export_img(pcd_2)
export_img(pcd_3)