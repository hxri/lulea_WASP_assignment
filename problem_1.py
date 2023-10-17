import os
import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

if __name__=="__main__": 

    # Set the path to your dataset folder
    data_folder = "./data/"
    image_folder = os.path.join(data_folder, "images/")
    poses_file = os.path.join(data_folder, "pose.csv")
    calibration_file = os.path.join(data_folder, "sensor.yaml")

    # Load camera calibration parameters
    with open(calibration_file) as f:
        calibration_data = yaml.safe_load(f)
        intrinsics = calibration_data['intrinsics']
        fu, fv, cu, cv = intrinsics
        K = np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1]])
        focal = (fu + fv) / 2
        pp = (cu, cv)

    image_paths = []

    # Loop through image files and store paths
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".png"):  # Adjust the file extension as needed
            image_path = os.path.join(image_folder, filename)
            image_paths.append(image_path)

    # Print the number of image paths stored
    print(f"Loaded {len(image_paths)} image paths.")

    #################################################################################
    # Load Ground truth
    #################################################################################

    # Load the poses.csv file
    poses_df = pd.read_csv(poses_file)

    # Extract camera poses
    poses = []
    for index, row in poses_df.iterrows():
        p_RS_R = np.array([row['p_RS_R_x'], row['p_RS_R_y'], row['p_RS_R_z']])
        q_RS_wxyz = np.array([row['q_RS_w'], row['q_RS_x'], row['q_RS_y'], row['q_RS_z']])
        
        # Convert quaternion to rotation matrix
        norm_q = np.linalg.norm(q_RS_wxyz)
        q_RS_wxyz /= norm_q
        R_RS = np.array([
            [1 - 2 * (q_RS_wxyz[2]**2 + q_RS_wxyz[3]**2), 2 * (q_RS_wxyz[1]*q_RS_wxyz[2] - q_RS_wxyz[0]*q_RS_wxyz[3]), 2 * (q_RS_wxyz[0]*q_RS_wxyz[2] + q_RS_wxyz[1]*q_RS_wxyz[3])],
            [2 * (q_RS_wxyz[1]*q_RS_wxyz[2] + q_RS_wxyz[0]*q_RS_wxyz[3]), 1 - 2 * (q_RS_wxyz[1]**2 + q_RS_wxyz[3]**2), 2 * (q_RS_wxyz[2]*q_RS_wxyz[3] - q_RS_wxyz[0]*q_RS_wxyz[1])],
            [2 * (q_RS_wxyz[1]*q_RS_wxyz[3] - q_RS_wxyz[0]*q_RS_wxyz[2]), 2 * (q_RS_wxyz[2]*q_RS_wxyz[3] + q_RS_wxyz[0]*q_RS_wxyz[1]), 1 - 2 * (q_RS_wxyz[1]**2 + q_RS_wxyz[2]**2)]
        ])
        
        # Create the camera pose matrix [R_RS | t_RS]
        T_RS = np.hstack((R_RS, p_RS_R.reshape(3, 1)))
        T_RS = np.vstack((T_RS, np.array([0, 0, 0, 1])))
        
        poses.append(T_RS)

    print('Got {} Poses' .format(len(poses)))


    #################################################################################
    # Precess image sequence to camera estimate trajectory
    #################################################################################

    # Initialize ORB detector
    # orb = cv2.ORB_create(nfeatures=3000)
    camera_poses = []
    errors = []  # To store the errors

    cv2.namedWindow("Feature Correspondences", cv2.WINDOW_AUTOSIZE)

    # Loop through images and perform feature detection
    for i in tqdm(range(len(image_paths))):  # Start from the second image
        image_prev = cv2.imread(image_paths[i - 1], cv2.IMREAD_GRAYSCALE)
        image_curr = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create(nfeatures=1000)
        kp_curr, des_curr = orb.detectAndCompute(image_curr, None)

        # Perform feature tracking from the previous frame to the current frame
        if i > 1:  # Skip tracking for the first frame
            # Use the feature points (kp_prev) from the previous frame
            kp_prev, des_prev = orb.detectAndCompute(image_prev, None)
            

            # Create a BFMatcher (Brute Force Matcher) to match descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_prev, des_curr)

            # Sort the matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matching keypoints from both frames
            matched_kp_prev = [kp_prev[match.queryIdx] for match in matches]
            matched_kp_curr = [kp_curr[match.trainIdx] for match in matches]

            points_prev = np.array([kp.pt for kp in matched_kp_prev])
            points_curr = np.array([kp.pt for kp in matched_kp_curr])

            # Perform correspondence estimation and compute the essential matrix
            E, mask = cv2.findEssentialMat(points_prev, points_curr, K, method=cv2.RANSAC, prob=0.999, threshold=0.8)
            _, R, t, _ = cv2.recoverPose(E, points_prev, points_curr, K)

            inlier_points_prev = points_prev[mask.ravel() == 1]
            inlier_points_curr = points_curr[mask.ravel() == 1]

            # Triangulate the 3D points using the estimated camera pose
            P_prev = np.dot(K, np.hstack((np.identity(3), np.zeros((3, 1)))))
            P_curr = np.dot(K, np.hstack((R, t)))

            # Ensure that the inlier points have the same number of points
            assert inlier_points_prev.shape[0] == inlier_points_curr.shape[0], "Number of inlier points should match."

            # Triangulate 3D points using the two camera poses and 2D points
            points_3d = cv2.triangulatePoints(P_prev, P_curr, inlier_points_prev.T, inlier_points_curr.T)

            # print(points_3d[0])
            # Convert the homogeneous coordinates to Euclidean coordinates
            points_3d[:3, :] /= points_3d[3, :]


            # Initialize variables for scale factor calculation
            total_distance = 0.0
            num_distances = 0

            # Calculate the distances between consecutive 3D points
            for j in range(1, points_3d.shape[1]):
                # Calculate the Euclidean distance between the current and previous 3D points
                distance = np.linalg.norm(points_3d[:3, j] - points_3d[:3, j - 1])

                # Add the distance to the total and increment the counter
                total_distance += distance
                num_distances += 1

            # Calculate the average scale factor
            average_scale_factor = total_distance / num_distances
            # print(average_scale_factor)


            points_3d_float64 = points_3d[:3].T.astype(np.float64)  # Convert to float64
            inlier_points_curr_float32 = inlier_points_curr.astype(np.float32)  # Convert to float32
            # print(points_3d_float64.shape)

            # Estimate camera pose using PnP (Perspective-n-Point)
            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d_float64, inlier_points_curr_float32, K, distCoeffs=None)

            # Convert the rotation vector (rvec) to a rotation matrix (R)
            R, _ = cv2.Rodrigues(rvec)

            scaled_t = t / average_scale_factor

            # Combine R and tvec to get the transformation matrix T
            T = np.hstack((R, scaled_t))
            T = np.vstack((T, np.array([0, 0, 0, 1])))

            # Accumulate the camera pose
            if not camera_poses:
                camera_poses.append(T)
            else:
                prev_pose = camera_poses[-1]
                cur_pose = np.dot(prev_pose, T)
                camera_poses.append(cur_pose)

                # Calculate the error between the ground truth and estimated pose
                ground_truth_pose = poses[i] 
                estimated_pose = cur_pose
                error = np.linalg.norm(ground_truth_pose[:3, 3] - estimated_pose[:3, 3])
                errors.append(error)

            # Show the two images side by side
            img_combined = np.hstack((image_prev, image_curr))

            # Draw feature correspondences on the combined image
            img_combined_with_matches = cv2.drawMatches(image_prev, kp_prev, image_curr, kp_curr, matches, outImg=img_combined)

            # Display the image with correspondences in the same window
            cv2.imshow("Feature Correspondences", img_combined_with_matches)
            cv2.resizeWindow("Feature Correspondences", 1504, 480)
            cv2.waitKey(1)


        else:
            initial_pose = poses[i] 
            print("Initialization: Setting initial pose as ground truth for the first frame.")

    cv2.destroyAllWindows() 


    sum_squared_error = 0.0
    sum_squared_error += np.array(errors) ** 2
    ate = np.sqrt(sum_squared_error / len(poses))

    # Plot the errors
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111)
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Error (Absolute Trajectory Error)")
    ax.set_title("Error Between Ground Truth and Estimated Trajectory")
    ax.plot(range(len(errors)), errors)
    plt.show()

    # Save the estimated trajectory
    np.save('estimated_trajectory.npy', camera_poses)

    #################################################################################
    # Plot 3d Points
    #################################################################################

    # Extract the 3D points from 'points_3d'
    x_3d = points_3d[0, :]
    y_3d = points_3d[1, :]
    z_3d = points_3d[2, :]

    # Create a 3D scatter plot for the 3D points
    fig = px.scatter_3d(x=x_3d, y=y_3d, z=z_3d, opacity=0.7, title='3D Map')
    fig.update_layout(autosize=False, width=600, height=600, showlegend=True, margin={"l":0,"r":0,"t":0,"b":0})
    fig.update_traces(marker=dict(size=3, line=dict(width=2, color='DarkSlateGrey'), opacity=0.7))

    # Show the interactive 3D map
    fig.show()

    #################################################################################
    # Plot trajectory
    #################################################################################

    # Extract camera positions from 'camera_poses'
    camera_positions = [pose[:3, 3] for pose in camera_poses]

    # Separate X, Y, and Z coordinates for estimated trajectory
    x = [pose[0] for pose in camera_positions]
    y = [pose[1] for pose in camera_positions]
    z = [pose[2] for pose in camera_positions]

    ground_truth_positions = [pose[:3, 3] for pose in poses]

    # Separate X, Y, and Z coordinates for ground truth trajectory
    gt_x = [pose[0] for pose in ground_truth_positions]
    gt_y = [pose[1] for pose in ground_truth_positions]
    gt_z = [pose[2] for pose in ground_truth_positions]

    # Create an interactive 3D scatter plot
    fig = go.Figure()
    fig.update_layout(autosize=False, width=1200, height=600, showlegend=True, margin={"l":0,"r":0,"t":0,"b":0})

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', name='Estimated Trajectory'))
    fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode='lines+markers', name='Ground Truth Trajectory'))

    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.update_layout(title='Camera Trajectory')

    fig.show()