import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

if __name__=="__main__": 

    # Load original point cloud data
    original_point_cloud = o3d.io.read_point_cloud("./sample_cloud.ply")
    o3d.visualization.draw_geometries([original_point_cloud])

    # Convert the point cloud to a NumPy array
    points = np.asarray(original_point_cloud.points)

    # Find the minimum and maximum Z coordinates (heights) in the point cloud
    min_height = points[:, 2].min()
    max_height = points[:, 2].max()

    # Set a threshold height above the bottom (1 meter or more)
    threshold_height = min_height + 5.9

    # Create a mask for points above the threshold height
    above_threshold_mask = points[:, 2] >= threshold_height

    # Filter the points above the threshold height
    seed_point_cloud = original_point_cloud.select_by_index(np.where(above_threshold_mask)[0])

    # Visualize and save the seed point cloud
    o3d.visualization.draw_geometries([seed_point_cloud])
    o3d.io.write_point_cloud("seed_point_cloud.ply", seed_point_cloud)

    # Calculate point normals
    seed_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    # Parameters for distinguishing powerline-related points
    max_horizontal_angle = 4.0  # Maximum allowed horizontal angle (in degrees)
    min_vertical_length = 5.0  # Minimum vertical length of a powerline segment

    # Create a new point cloud to store powerline-related points
    powerline_cloud = o3d.geometry.PointCloud()

    # Iterate through each point and its normal
    for i, point in enumerate(seed_point_cloud.points):
        normal = seed_point_cloud.normals[i]
        
        # Calculate the vertical angle (angle between the normal and the vertical axis)
        vertical_angle = np.arccos(np.abs(np.dot(normal, [0, 0, 1]))) * 180.0 / np.pi
        
        # Check if the point's normal indicates a vertical element (powerline)
        if vertical_angle <= max_horizontal_angle:
            powerline_cloud.points.append(point)

    # Convert the powerline point cloud to a NumPy array for clustering
    powerline_points = np.asarray(powerline_cloud.points)

    # Standardize the data (mean=0, std=1) to prepare it for DBSCAN
    scaler = StandardScaler()
    scaled_powerline_points = scaler.fit_transform(powerline_points)

    # Apply DBSCAN clustering
    eps = 0.2  # Neighborhood radius
    min_samples = 50  # Minimum number of points in a cluster
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled_powerline_points)

    # Create a new point cloud for each cluster
    clustered_powerlines = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue 
        cluster_points = scaled_powerline_points[labels == label]
        cluster = o3d.geometry.PointCloud()
        cluster.points = o3d.utility.Vector3dVector(scaler.inverse_transform(cluster_points))
        clustered_powerlines.append(cluster)

    o3d.visualization.draw_geometries(clustered_powerlines, window_name="Powerline Clusters")


    fitted_lines = []

    for cluster in clustered_powerlines:
        # Convert the cluster to a NumPy array for PCA
        cluster_points = np.asarray(cluster.points)
        
        # Perform PCA to find the principal component
        covariance_matrix = np.cov(cluster_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Sort eigenvectors by eigenvalues to find the principal component
        principal_component = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Calculate the center of the cluster
        center = np.mean(cluster_points, axis=0)
        
        # Create a line segment with endpoints based on the principal component
        line_length = 25.0 
        endpoint1 = center - line_length / 2 * principal_component
        endpoint2 = center + line_length / 2 * principal_component
        
        # Create a line segment using Open3D
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([endpoint1, endpoint2])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1, 0, 0]]) 
        fitted_lines.append(line)

    # Visualize or save the fitted line segments
    o3d.visualization.draw_geometries(fitted_lines + [original_point_cloud], window_name="Fitted Powerline Segments")