import cv2
import numpy as np
from func import *
import glob
import matplotlib.pyplot as plt

def main():
    # File paths
    image_paths = glob.glob('Data/*.jpg')  # Adjust the path as needed
    image_paths.sort()
    calibration_path = 'Data/K.txt'

    if len(image_paths) < 2:
        print("Not enough images found in the Data directory. At least two images are required.")
        return

    # Load images and calibration data
    try:
        images = load_images(image_paths)
    except FileNotFoundError as e:
        print(e)
        return

    try:
        K = read_calibration(calibration_path)
    except Exception as e:
        print(f"Error reading calibration file: {e}")
        return

    print(f"Intrinsic Matrix K:\n{K}\n")

    # Initialize list to hold all 3D points and colors
    all_points_3d = []
    all_colors = []

    # Initialize global rotation and translation
    R_total = np.eye(3)
    t_total = np.zeros((3, 1))

    # Store the previous pose
    previous_R = R_total.copy()
    previous_t = t_total.copy()

    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]

        print(f"Processing image pair {i + 1} and {i + 2}...")

        # Feature matching
        try:
            src_pts, dst_pts, img_matches = feature_matching_and_display(img1, img2)
            print("Feature matching completed.\n")
        except ValueError as e:
            print(f"Feature matching error: {e}")
            continue

        # Recover relative pose
        try:
            R_rel, t_rel, mask_pose = recover_camera_pose(src_pts, dst_pts, K)
            print("Camera pose recovered.\n")
        except ValueError as e:
            print(f"Camera pose recovery error: {e}")
            continue

        # Normalize the translation vector to mitigate scale issues
        t_rel = t_rel / np.linalg.norm(t_rel)

        # Update global pose
        R_total = R_rel @ previous_R
        t_total = previous_t + previous_R @ t_rel

        # Triangulate points using global poses
        try:
            points_3d = triangulate_points(src_pts, dst_pts, previous_R, previous_t, R_total, t_total, K)
            print("Triangulation completed.\n")
        except Exception as e:
            print(f"Triangulation error: {e}")
            continue

        # Optionally perform bundle adjustment (commented out for simplicity)
        # Uncomment the following lines to enable bundle adjustment
        """
        try:
            points_3d_refined, R_total, t_total = bundle_adjustment(points_3d, src_pts, dst_pts, K, R_total, t_total)
            if points_3d_refined is not None:
                points_3d = points_3d_refined
                print("Bundle adjustment completed.\n")
        except Exception as e:
            print(f"Bundle adjustment error: {e}")
        """

        # Extract colors for the matched points in img1
        colors = extract_colors_from_image(img1, src_pts.reshape(-1, 2))

        # Append the triangulated points and their colors
        all_points_3d.append(points_3d.T)
        all_colors.append(colors)

        # Update previous pose
        previous_R = R_total.copy()
        previous_t = t_total.copy()

    if not all_points_3d:
        print("No 3D points were reconstructed. Exiting.")
        return

    # Concatenate all 3D points and colors
    all_points_3d = np.concatenate(all_points_3d, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    print(f"Total 3D points reconstructed: {all_points_3d.shape[0]}")
    print(f"Point cloud shape: {all_points_3d.shape}")
    print(f"Color data shape: {all_colors.shape}\n")

    # Visualize the point cloud
    visualize_point_cloud(all_points_3d)

    # Save the point cloud with colors
    save_point_cloud_as_ply(all_points_3d, all_colors, filename="output_with_colors.ply")

    print("Structure from Motion pipeline completed successfully.")

if __name__ == '__main__':
    main()
