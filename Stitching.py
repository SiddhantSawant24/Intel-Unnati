import cv2
import numpy as np
import matplotlib.pyplot as plt

def align_images(ref_img, target_img):
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

    # Detect and compute ORB features
    orb = cv2.ORB_create()
    keypoints_ref, descriptors_ref = orb.detectAndCompute(gray_ref, None)
    keypoints_target, descriptors_target = orb.detectAndCompute(gray_target, None)

    # Match features using BFMatcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_ref, descriptors_target)

    # Sort matches by distance (best matches first)
    matches = sorted(matches, key=lambda x: x.distance)

    # Initialize lists for storing manually selected matches
    src_pts = []
    dst_pts = []

    # Interactive match selection
    for match in matches:
        img_matches = cv2.drawMatches(
            ref_img, keypoints_ref, target_img, keypoints_target, [match], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imshow('Match', img_matches)
        print("Press 's' to select, 'r' to reject, 'q' to quit")
        key = cv2.waitKey(0) & 0xFF  # Wait for a key press and mask for 8-bit value

        if key == ord('s'):  # 's' key to select the match
            print("Match selected")
            src_pts.append(keypoints_ref[match.queryIdx].pt)
            dst_pts.append(keypoints_target[match.trainIdx].pt)
        elif key == ord('r'):  # 'r' key to reject the match
            print("Match rejected")
            continue  # Skip to the next match
        elif key == ord('q'):  # 'q' key to quit early
            print("Quitting selection early")
            break
        print(f'The number of matches selected is {len(src_pts)}')

    cv2.destroyAllWindows()  # Close all OpenCV windows after selection

    # Convert selected points to numpy arrays
    src_pts = np.array(src_pts, dtype=np.float32).reshape(-1, 2)
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape(-1, 2)

    # Check if enough points have been selected
    if len(src_pts) < 3:
        raise ValueError(f"Not enough matches are selected - {len(src_pts)}/3 required")

    # Estimate the translation matrix
    matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    tx, ty = matrix[0, 2], matrix[1, 2]  # Extract translation components

    # Warp target image to align with reference image
    height, width, channels = ref_img.shape
    aligned_target_img = cv2.warpAffine(target_img, matrix, (width, height))

    return aligned_target_img, matrix

# Load images
img1 = cv2.imread('/home/siddhant/Camera Images/_overhead_camera_overhead_camera4_image_raw.jpg')
img2 = cv2.imread('/home/siddhant/Camera Images/_overhead_camera_overhead_camera3_image_raw.jpg')
img3 = cv2.imread('/home/siddhant/Camera Images/_overhead_camera_overhead_camera2_image_raw.jpg')
img4 = cv2.imread('/home/siddhant/Camera Images/_overhead_camera_overhead_camera1_image_raw.jpg')

# Align images to the first image
aligned_img2, _ = align_images(img1, img2)
aligned_img3, _ = align_images(img1, img3)
aligned_img4, _ = align_images(img1, img4)

# Display the aligned images
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title('Image 1 (Reference)')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 2)
plt.title('Aligned Image 2')
plt.imshow(cv2.cvtColor(aligned_img2, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 3)
plt.title('Aligned Image 3')
plt.imshow(cv2.cvtColor(aligned_img3, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 4)
plt.title('Aligned Image 4')
plt.imshow(cv2.cvtColor(aligned_img4, cv2.COLOR_BGR2RGB))

plt.show()
