import open3d as o3d
import os
import sys

def createPointCloud(depthPath, imagePath):

    # Load in color and depth image to create the point cloud
    color_raw = o3d.io.read_image(imagePath)
    depth_raw = o3d.io.read_image(depthPath)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    
    # Camera intrinsic parameters built into Open3D for Prime Sense
    camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        
    # Create the point cloud from images and camera intrisic parameters
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.4)
    vis.run()
    vis.destroy_window()
  
