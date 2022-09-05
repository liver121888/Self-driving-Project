#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2
from cv_bridge import CvBridge
import cv2
import tf

FRAME_ID = 'map'
DETECTION_COLOR_DICT = {'Car':(255, 255, 0), 'Pedestrian':(0, 226, 255), 'Cyclist':(141, 40, 255)}

LIFETIME = 0.1
LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
LINES+= [[4, 1], [5, 0]] # front face

Q_ZERO = tf.transformations.quaternion_from_euler(0, 0, 0)

def publish_camera(cam_pub, bridge, image, boxes, types):
    for box, data_type in zip(boxes, types):
        top_left = int(box[0]), int(box[1])
        bottom_right = int(box[2]), int(box[3])
        cv2.rectangle(image, top_left, bottom_right, DETECTION_COLOR_DICT[data_type], 2)
    cam_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))


def publish_point_cloud(pcl_pub, point_cloud):
    header = Header()
    header.stamp = rospy.Time()
    header.frame_id = FRAME_ID
    pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:, :3]))

def publish_ego_car(ego_car_pub):
    """
    Publish left and right 45 degree FOV lines and ego car model mesh
    """

    marker_array = MarkerArray()

    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    # to infinity
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP

    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2
    marker.pose.orientation.x = Q_ZERO[0]
    marker.pose.orientation.y = Q_ZERO[1]
    marker.pose.orientation.z = Q_ZERO[2]
    marker.pose.orientation.w = Q_ZERO[3]

    marker.points = []
    marker.points.append(Point(10, -10, 0))
    marker.points.append(Point(0, 0, 0))
    marker.points.append(Point(10, 10, 0))

    marker_array.markers.append(marker)

    mesh_marker = Marker()
    mesh_marker.header.frame_id = FRAME_ID
    mesh_marker.header.stamp = rospy.Time.now()

    mesh_marker.id = -1
    mesh_marker.lifetime = rospy.Duration()
    mesh_marker.type = Marker.MESH_RESOURCE
    mesh_marker.mesh_resource = "package://kitti_tutorial/meshes/bmw-x5.dae"
    mesh_marker.pose.position.x = 0.0
    mesh_marker.pose.position.y = 0.0
    mesh_marker.pose.position.z = -1.73

    q = tf.transformations.quaternion_from_euler(np.pi/2, 0, np.pi)
    mesh_marker.pose.orientation.x = q[0]
    mesh_marker.pose.orientation.y = q[1]
    mesh_marker.pose.orientation.z = q[2]
    mesh_marker.pose.orientation.w = q[3]

    mesh_marker.color.r = 1.0
    mesh_marker.color.g = 1.0
    mesh_marker.color.b = 1.0
    mesh_marker.color.a = 1.0

    mesh_marker.scale.x = 1.0
    mesh_marker.scale.y = 1.0
    mesh_marker.scale.z = 1.0

    marker_array.markers.append(mesh_marker)

    ego_car_pub.publish(marker_array)

def publish_imu(imu_pub, imu_data, log=False):
    """
    Publish IMU data
    """
    imu = Imu()
    imu.header.frame_id = FRAME_ID
    imu.header.stamp = rospy.Time.now()
     # prevent the data from being overwritten
    q = tf.transformations.quaternion_from_euler(float(imu_data.roll), float(imu_data.pitch), \
                                                     float(imu_data.yaw))
    imu.orientation.x = q[0]
    imu.orientation.y = q[1]
    imu.orientation.z = q[2]
    imu.orientation.w = q[3]
    imu.linear_acceleration.x = imu_data.af
    imu.linear_acceleration.y = imu_data.al
    imu.linear_acceleration.z = imu_data.au
    imu.angular_velocity.x = imu_data.wf
    imu.angular_velocity.y = imu_data.wl
    imu.angular_velocity.z = imu_data.wu

    imu_pub.publish(imu)
    if log:
        rospy.loginfo("imu msg published")

def publish_gps(gps_pub, gps_data, log=False):
    """
    Publish GPS data
    """
    gps = NavSatFix()
    gps.header.frame_id = FRAME_ID
    gps.header.stamp = rospy.Time.now()
    gps.latitude = gps_data.lat
    gps.longitude = gps_data.lon
    gps.altitude = gps_data.alt

    gps_pub.publish(gps)
    if log:
        rospy.loginfo("gps msg published")

def publish_3dbox(box3d_pub, corners_3d_velos, track_ids, object_types=None, \
    publish_id=True, publish_distance=False, log=False):
    """
    Publish 3d boxes in velodyne coordinate, with color specified by object_types
    If object_types is None, set all color to cyan
    corners_3d_velos : list of (8, 4) 3d corners
    """
    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        track_id = track_ids[i]
        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()

        # marker.id = track_id
        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(LIFETIME)
        marker.type = Marker.LINE_LIST

        if object_types is None:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 1.0
        else:
            b, g, r = DETECTION_COLOR_DICT[object_types[i]]
            marker.color.r = r/255.0
            marker.color.g = g/255.0
            marker.color.b = b/255.0

        marker.color.a = 1.0
        marker.scale.x = 0.2
        marker.pose.orientation.x = Q_ZERO[0]
        marker.pose.orientation.y = Q_ZERO[1]
        marker.pose.orientation.z = Q_ZERO[2]
        marker.pose.orientation.w = Q_ZERO[3]

        marker.points = []
        for l in LINES:
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)

        if publish_id:
            text_marker = Marker()
            text_marker.header.frame_id = FRAME_ID
            text_marker.header.stamp = rospy.Time.now()

        # separate from previous only with i
        text_marker.id = i + 1000
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(LIFETIME)
        text_marker.type = Marker.TEXT_VIEW_FACING

        # assign the position of the marker
        p4 = corners_3d_velo[4] # upper front left corner

        text_marker.pose.position.x = p4[0]
        text_marker.pose.position.y = p4[1]
        text_marker.pose.position.z = p4[2] + 0.5

        text_marker.pose.orientation.x = Q_ZERO[0]
        text_marker.pose.orientation.y = Q_ZERO[1]
        text_marker.pose.orientation.z = Q_ZERO[2]
        text_marker.pose.orientation.w = Q_ZERO[3]

        # text_marker.text = str(track_id)
        text_marker.text = str(track_ids[i])


        text_marker.scale.x = 1
        text_marker.scale.y = 1
        text_marker.scale.z = 1

        if object_types is None:
            text_marker.color.r = 0.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
        else:
            b, g, r = DETECTION_COLOR_DICT[object_types[i]]
            text_marker.color.r = r/255.0
            text_marker.color.g = g/255.0
            text_marker.color.b = b/255.0
        text_marker.color.a = 1.0
        marker_array.markers.append(text_marker)

    box3d_pub.publish(marker_array)