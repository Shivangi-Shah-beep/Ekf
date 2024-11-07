#!/usr/bin/env python

#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, Pose, Quaternion, Point, Vector3
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class EKFOdometryNode:
    def __init__(self):
        rospy.init_node('ekf_odometry_node', anonymous=True)

        # Parameters
        self.R = rospy.get_param('~robot_radius', 0.2)  # Set to your robot's radius
        self.r_w = rospy.get_param('~wheel_radius', 0.05)  # Set to your wheel radius

        # State vector [theta, x, y, v, omega]
        self.X = np.zeros((5, 1))
        # Covariance matrix P (5x5)
        self.P = np.eye(5) * 0.1  # Initial covariance

        # Process noise covariance Q (5x5)
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])

        # Measurement noise covariance R (5x5)
        self.R_cov = np.diag([0.05, 0.05, 0.05, 0.05, 0.02])  # Adjust based on sensor characteristics

        # Specific parameters from Assignment 3
        self.a_v = 0.977
        self.G_v = 0.9962
        self.a_omega = 0.96
        self.G_omega = 1.356

        # Subscribers
        imu_sub = Subscriber('/imu/data', Imu)
        joint_sub = Subscriber('/joint_states', JointState)
        cmd_vel_sub = Subscriber('/cmd_vel', Twist)

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [imu_sub, joint_sub, cmd_vel_sub],
            queue_size=10,
            slop=0.02,  # Adjust slop based on sensor frequencies
            allow_headerless=True
        )
        self.sync.registerCallback(self.sensor_callback)

        # Publisher
        self.odom_pub = rospy.Publisher('/ekf_odom', Odometry, queue_size=10)

        # Previous timestamp
        self.prev_time = None

        # Handle absence of cmd_vel messages
        self.last_cmd_vel_time = rospy.Time.now()
        self.cmd_vel_timeout = rospy.Duration(1.0)  # 1 second timeout

        # Command velocities
        self.u_v = 0.0
        self.u_omega = 0.0

        rospy.loginfo("EKF Odometry Node Initialized with specific parameters")

    def sensor_callback(self, imu_msg, joint_msg, cmd_vel_msg):
        current_time = imu_msg.header.stamp  # Use IMU timestamp

        # Handle time delta
        if self.prev_time is None:
            self.prev_time = current_time
            return
        delta_t = (current_time - self.prev_time).to_sec()
        self.prev_time = current_time

        # Update command velocities
        self.update_cmd_vel(cmd_vel_msg, current_time)

        # Check for cmd_vel timeout
        self.check_cmd_vel_timeout(current_time)

        # Get measurements
        z = self.get_measurements(imu_msg, joint_msg)

        # EKF Prediction Step
        self.ekf_predict(delta_t)

        # EKF Update Step
        self.ekf_update(z)

        # Publish odometry
        self.publish_odometry(current_time)

    def update_cmd_vel(self, cmd_vel_msg, current_time):
        self.u_v = cmd_vel_msg.linear.x
        self.u_omega = cmd_vel_msg.angular.z
        self.last_cmd_vel_time = current_time

    def check_cmd_vel_timeout(self, current_time):
        if (current_time - self.last_cmd_vel_time) > self.cmd_vel_timeout:
            # No recent cmd_vel, assume zero velocities
            self.u_v = 0.0
            self.u_omega = 0.0

    def get_measurements(self, imu_msg, joint_msg):
        # Gyroscope measurement (omega_g)
        omega_g = imu_msg.angular_velocity.z

        # Joint states
        wheel_names = joint_msg.name
        wheel_velocities = joint_msg.velocity

        # Initialize wheel angular velocities
        omega_fr = omega_rr = omega_fl = omega_rl = 0.0

        # Map wheel names to velocities
        for name, velocity in zip(wheel_names, wheel_velocities):
            if name == 'front_right_wheel':
                omega_fr = velocity
            elif name == 'rear_right_wheel':
                omega_rr = velocity
            elif name == 'front_left_wheel':
                omega_fl = velocity
            elif name == 'rear_left_wheel':
                omega_rl = velocity

        # Measurement vector z
        z = np.array([[omega_fr],
                      [omega_rr],
                      [omega_fl],
                      [omega_rl],
                      [omega_g]])

        return z

    def ekf_predict(self, delta_t):
        # Extract state variables
        theta, x, y, v, omega = self.X.flatten()

        # State prediction
        theta_pred = theta + omega * delta_t
        x_pred = x + v * np.cos(theta) * delta_t
        y_pred = y + v * np.sin(theta) * delta_t
        v_pred = self.a_v * v + self.G_v * (1 - self.a_v) * self.u_v
        omega_pred = self.a_omega * omega + self.G_omega * (1 - self.a_omega) * self.u_omega

        self.X_pred = np.array([[theta_pred],
                                [x_pred],
                                [y_pred],
                                [v_pred],
                                [omega_pred]])

        # Jacobian A
        A = np.array([
            [1, 0, 0, 0, delta_t],
            [-v * np.sin(theta) * delta_t, 1, 0, np.cos(theta) * delta_t, 0],
            [v * np.cos(theta) * delta_t, 0, 1, np.sin(theta) * delta_t, 0],
            [0, 0, 0, self.a_v, 0],
            [0, 0, 0, 0, self.a_omega]
        ])

        # Process noise covariance Q
        Q = self.Q

        # Covariance prediction
        self.P_pred = A @ self.P @ A.T + Q

    def ekf_update(self, z):
        # Measurement matrix C
        C = np.array([
            [0, 0, 0, 1 / self.r_w, self.R / self.r_w],
            [0, 0, 0, 1 / self.r_w, self.R / self.r_w],
            [0, 0, 0, 1 / self.r_w, -self.R / self.r_w],
            [0, 0, 0, 1 / self.r_w, -self.R / self.r_w],
            [0, 0, 0, 0, 1]
        ])

        # Innovation
        y = z - C @ self.X_pred

        # Innovation covariance
        S = C @ self.P_pred @ C.T + self.R_cov

        # Kalman gain
        K = self.P_pred @ C.T @ np.linalg.inv(S)

        # State update
        self.X = self.X_pred + K @ y

        # Covariance update
        self.P = (np.eye(5) - K @ C) @ self.P_pred

    def publish_odometry(self, current_time):
        odom = Odometry()
        odom.header.stamp = current_time  # Reflects the time of the measurements
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        # Position
        odom.pose.pose.position = Point(self.X[1, 0], self.X[2, 0], 0.0)

        # Orientation (convert theta to quaternion)
        quaternion = quaternion_from_euler(0, 0, self.X[0, 0])
        odom.pose.pose.orientation = Quaternion(*quaternion)

        # Velocities
        odom.twist.twist.linear = Vector3(self.X[3, 0], 0.0, 0.0)
        odom.twist.twist.angular = Vector3(0.0, 0.0, self.X[4, 0])

        # Covariance
        # Map 5x5 state covariance to 6x6 pose covariance and 6x6 twist covariance
        pose_cov = np.zeros((6, 6))
        pose_cov[0, 0] = self.P[1, 1]  # x covariance
        pose_cov[1, 1] = self.P[2, 2]  # y covariance
        pose_cov[5, 5] = self.P[0, 0]  # theta covariance

        odom.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6))
        twist_cov[0, 0] = self.P[3, 3]  # v covariance
        twist_cov[5, 5] = self.P[4, 4]  # omega covariance

        odom.twist.covariance = twist_cov.flatten().tolist()

        # Publish odometry
        self.odom_pub.publish(odom)

def main():
    ekf_node = EKFOdometryNode()
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
