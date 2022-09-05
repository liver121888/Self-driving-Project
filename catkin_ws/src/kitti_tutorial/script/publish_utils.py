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
    q = tf.transformations.quaternion_from_euler(0, 0, 0)
    marker.pose.orientation.x = q[0]
    marker.pose.orientation.y = q[1]
    marker.pose.orientation.z = q[2]
    marker.pose.orientation.w = q[3]

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