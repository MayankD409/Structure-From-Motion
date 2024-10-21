import numpy as np
import cv2
import open3d as o3d
import re
import matplotlib.pyplot as plt

######################################################################################
                            # FEATURE MATCHING #
######################################################################################

def initialize_camera_matrix(f, cx, cy):
    """ Initialize the camera intrinsic matrix. """
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    return K

def feature_matching_and_display(img1, img2):
    """ Feature matching using SIFT and FLANN and displaying matched features. """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector with more features
    sift = cv2.SIFT_create(nfeatures=5000)  # Increased number of features

    # Find keypoints and descriptors
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)

    if descriptors_1 is None or descriptors_2 is None:
        raise ValueError("No descriptors found in one or both images.")

    # Initialize FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)  # Increased number of checks for better matching

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Relaxed Lowe's ratio test for more matches
            good_matches.append(m)

    print(f"Total matches found: {len(matches)}")
    print(f"Good matches after ratio test: {len(good_matches)}")

    if len(good_matches) < 10:
        raise ValueError("Not enough good matches found. Try capturing more images or different viewpoints.")

    # Draw matches (optional visualization for debugging)
    img_matches = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract location of good matches
    src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches])

    return src_pts, dst_pts, img_matches

######################################################################################
                            # CAMERA POSE RECOVERY #
######################################################################################

def recover_camera_pose(src_pts, dst_pts, K):
    """ Recover the relative camera pose using the Essential Matrix. """
    # Compute the Essential Matrix directly using RANSAC
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    if E is None:
        raise ValueError("Essential matrix computation failed.")

    # Decompose Essential Matrix to obtain the relative rotation and translation
    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K)

    return R, t, mask_pose

######################################################################################
                            # TRIANGULATION #
######################################################################################

def triangulate_points(src_pts, dst_pts, R1, t1, R2, t2, K):
    """ Triangulate 3D points from corresponding image points in two views. """
    # Projection matrices
    P1 = K @ np.hstack((R1, t1))
    P2 = K @ np.hstack((R2, t2))

    # Reshape points to (2, N)
    src_pts = src_pts.T  # Shape becomes (2, N)
    dst_pts = dst_pts.T

    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)
    points_3d = points_4d_hom[:3, :] / points_4d_hom[3, :]  # Convert to Euclidean coordinates

    return points_3d

######################################################################################
                            # BUNDLE ADJUSTMENT #
######################################################################################

def bundle_adjustment(points_3d, src_pts, dst_pts, K, R, t):
    """ Simple bundle adjustment to refine camera pose and 3D points. """
    # Ensure points are in the correct format and adequate for PnP solver
    object_points = points_3d.T.astype(np.float64)
    image_points = dst_pts.astype(np.float64)
    camera_matrix = K.astype(np.float64)
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Initialize rvec and tvec from R and t if available or use zeros
    if R is not None and t is not None:
        rvec, _ = cv2.Rodrigues(R)
        tvec = t.astype(np.float64)
    else:
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))

    # Check if there are enough points for solvePnP
    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        print("Not enough points to run solvePnP")
        return None, None, None

    # Refine pose using solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        print("solvePnP failed to find a solution.")
        return None, None, None

    # Convert rotation vector back to a rotation matrix
    R_refined, _ = cv2.Rodrigues(rvec)
    t_refined = tvec

    # Update 3D points using refined pose
    P2 = np.hstack((R_refined, t_refined))
    P2 = K @ P2
    P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
    P1 = K @ P1
    points_4d_hom = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
    points_3d_refined = points_4d_hom[:3, :] / points_4d_hom[3, :]

    return points_3d_refined, R_refined, t_refined

######################################################################################
                # 3D RECONSTRUCTION AND POST-PROCESSING #
######################################################################################

def visualize_point_cloud(points_3d):
    """ Visualize the point cloud with normals. """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Optionally orient normals to be consistent with the viewpoint
    pcd.orient_normals_towards_camera_location(np.array([0, 0, 0]))

    o3d.visualization.draw_geometries([pcd], window_name="3D Reconstruction", width=800, height=600, left=50, top=50)

def save_point_cloud_as_ply(points_3d, colors=None, filename="output.ply"):
    """ Save the point cloud as a PLY file for use in tools like Meshlab. """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(filename, pcd)
    print(f"Point cloud saved to {filename}")

######################################################################################
                            # HELPER FUNCTIONS #
######################################################################################

def read_calibration(file_path):
    """ Read camera intrinsic matrix from the calibration file. """
    with open(file_path, 'r') as file:
        content = file.read().replace('\n', '').replace(';', '')
        K_values = re.findall(r"[-+]?\d*\.\d+|\d+", content)
        if len(K_values) < 9:
            raise ValueError("Calibration file does not contain enough values for a 3x3 intrinsic matrix.")
        K = np.array(list(map(float, K_values[:9]))).reshape(3, 3)
    return K

def load_images(image_paths):
    """ Load all images from specified paths as color images. """
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
    for idx, img in enumerate(images):
        if img is None:
            raise FileNotFoundError(f"Image at path {image_paths[idx]} could not be loaded.")
    return images

def extract_colors_from_image(image, points):
    """Extract color values (BGR) from the image based on 2D points."""
    h, w = image.shape[:2]
    colors = []

    for point in points:
        x, y = int(point[0]), int(point[1])
        # Ensure the point is within image bounds
        if 0 <= x < w and 0 <= y < h:
            color = image[y, x]  # OpenCV uses (y, x) indexing
            colors.append(color / 255.0)  # Normalize to [0, 1] for Open3D
        else:
            colors.append([0, 0, 0])  # Black if point is out of bounds

    return np.array(colors)
