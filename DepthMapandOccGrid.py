import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_focal_length(sensor_width, fov_degrees):
    fov_radians = np.deg2rad(fov_degrees)
    focal_length = sensor_width / (2 * np.tan(fov_radians / 2))
    return focal_length

def stereo_rectification(left_image_path, right_image_path,
                         left_camera_matrix, right_camera_matrix,
                         left_dist_coeffs, right_dist_coeffs,
                         R, T):
    # Load the stereo images
    left_img = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Get the size of the images
    image_size = left_img.shape[:2][::-1]  # width and height

    # Stereo rectification
    rectify_scale = 1  # 0 for cropped, 1 for full
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_camera_matrix, left_dist_coeffs,
        right_camera_matrix, right_dist_coeffs,
        image_size, R, T, alpha=rectify_scale
    )

    # Compute the rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_16SC2
    )

    # Apply the rectification maps
    left_rectified = cv2.remap(left_img, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, right_map1, right_map2, cv2.INTER_LINEAR)

    return left_rectified, right_rectified, Q

def compute_depth_map(left_rectified, right_rectified, Q, far_disparity_value=0):
    # Improve stereo matching with StereoSGBM
    window_size = 5
    min_disp = 0
    num_disp = 16 * 10  # Must be divisible by 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Compute disparity map
    disparity_map = stereo.compute(left_rectified, right_rectified).astype(np.float32) / 16.0

    # Apply a median blur to the disparity map
    disparity_map = cv2.medianBlur(disparity_map, 5)

    # Identify gray floor areas and set their disparity to far_disparity_value
    gray_mask = cv2.inRange(left_rectified, 154, 156)  # Adjust range based on floor color
    disparity_map[gray_mask == 255] = far_disparity_value

    # Convert disparity map to depth map using Q matrix
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q)
    depth_map = points_3D[:, :, 2]

    # Normalize the depth map for visualization
    depth_map_visual = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_visual = np.uint8(depth_map_visual)

    return disparity_map, depth_map, depth_map_visual

# Example usage:
image_paths = {
    'C1': '/home/siddhant/Camera Images/_overhead_camera_overhead_camera1_image_raw.jpg',
    'C2': '/home/siddhant/Camera Images/_overhead_camera_overhead_camera2_image_raw.jpg',
    'C3': '/home/siddhant/Camera Images/_overhead_camera_overhead_camera4_image_raw.jpg',
    'C4': '/home/siddhant/Camera Images/_overhead_camera_overhead_camera3_image_raw.jpg'
}

# Example intrinsic parameters (camera matrices and distortion coefficients)
sensor_width = 640
sensor_height = 480
fov_degrees = 60

focal_length = calculate_focal_length(sensor_width, fov_degrees)
print(f"Focal Length: {focal_length} pixels")

# Set up Camera Projection Matrices
c_x, c_y = sensor_width / 2, sensor_height / 2
camera_matrix = np.array([[focal_length, 0, c_x], [0, focal_length, c_y], [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros((5, 1), dtype=np.float64)  # Assuming no distortion

# Example extrinsic parameters (rotation and translation between the cameras)
camera_positions = {
    'C1': np.array([-5, -2, 8], dtype=np.float64),
    'C2': np.array([-5, 3, 8], dtype=np.float64),
    'C3': np.array([1, -2, 8], dtype=np.float64),
    'C4': np.array([1, 3, 8], dtype=np.float64)
}

camera_rotations = {
    'C1': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    'C2': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    'C3': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64),
    'C4': np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
}

# Define camera pairs
camera_pairs = [('C3', 'C2'), ('C3', 'C4'), ('C1', 'C3'), ('C2', 'C4')]

# Rectify and compute depth maps for each pair
depth_maps = []

# For each camera pair
for cam1, cam2 in camera_pairs:
    R1, R2 = camera_rotations[cam1], camera_rotations[cam2]
    t1, t2 = camera_positions[cam1], camera_positions[cam2]

    R_rel = R1 @ R2.T  # Note the order is swapped for correct relative rotation
    if cam1 == 'C3' and cam2 == 'C2':
        T_rel = np.array([0, 5, 0], dtype=np.float64)  # Manually set translation for C3-C2
    else:
        T_rel = np.abs(t2 - t1)  # Convert negative translation to positive

    left_rectified, right_rectified, Q = stereo_rectification(
        image_paths[cam1], image_paths[cam2],
        camera_matrix, camera_matrix,
        dist_coeffs, dist_coeffs,
        R_rel, T_rel
    )

    disparity_map, depth_map, depth_map_visual = compute_depth_map(left_rectified, right_rectified, Q)
    depth_maps.append(depth_map)

# Display the results
plt.figure(figsize=(20, 10))

# Display individual depth maps
for i, (cam1, cam2) in enumerate(camera_pairs):
    plt.subplot(len(camera_pairs), 3, i * 3 + 1)
    plt.imshow(cv2.imread(image_paths[cam1], cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'{cam1} Rectified Image')
    plt.axis('off')

    plt.subplot(len(camera_pairs), 3, i * 3 + 2)
    plt.imshow(cv2.imread(image_paths[cam2], cv2.IMREAD_GRAYSCALE), cmap='gray')
    plt.title(f'{cam2} Rectified Image')
    plt.axis('off')

    plt.subplot(len(camera_pairs), 3, i * 3 + 3)
    plt.imshow(depth_maps[i], cmap='gray')
    plt.colorbar()
    plt.title(f'Depth Map {cam1}-{cam2}')
    plt.axis('off')

plt.show()
