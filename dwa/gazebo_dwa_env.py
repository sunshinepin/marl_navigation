import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
import numpy as np
from dwa_control import dwa_control
from config import Config

class GazeboDWAEnv:
    def __init__(self, robot_index, goal):
        rospy.init_node(f"dwa_gazebo_{robot_index}", anonymous=True)

        self.config = Config()
        self.goal = goal
        self.robot_state = [0, 0, 0, 0, 0]
        self.obstacles = np.array([[5.0, 5.0], [3.0, 6.0], [3.0, 8.0], [3.0, 10.0]])

        self.vel_pub = rospy.Publisher(f"/car{robot_index+1}/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber(f"/car{robot_index+1}/odom_gazebo", Odometry, self.odom_callback)
        rospy.Subscriber(f"/car{robot_index+1}/velodyne/velodyne_points", LaserScan, self.laser_callback)

    def odom_callback(self, odom):
        self.robot_state[0] = odom.pose.pose.position.x
        self.robot_state[1] = odom.pose.pose.position.y
        quaternion = Quaternion(odom.pose.pose.orientation.w,
                                odom.pose.pose.orientation.x,
                                odom.pose.pose.orientation.y,
                                odom.pose.pose.orientation.z)
        self.robot_state[2] = quaternion.to_euler(degrees=False)[2]
        self.robot_state[3] = odom.twist.twist.linear.x
        self.robot_state[4] = odom.twist.twist.angular.z

    def laser_callback(self, scan):
        ranges = np.array(scan.ranges)
        angles = np.linspace(scan.angle_min, scan.angle_max, len(ranges))
        self.obstacles = np.array([[r * math.cos(a), r * math.sin(a)] for r, a in zip(ranges, angles) if r < scan.range_max])

    def control_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            u, trajectory = dwa_control(self.robot_state, self.config, self.goal, self.obstacles)
            self.publish_control(u)
            rate.sleep()

    def publish_control(self, u):
        vel_cmd = Twist()
        vel_cmd.linear.x = u[0]
        vel_cmd.angular.z = u[1]
        self.vel_pub.publish(vel_cmd)

if __name__ == '__main__':
    goals = [(10, 10), (10, -10)]  # 目标点
    robots = [GazeboDWAEnv(i, goal) for i, goal in enumerate(goals)]
    for robot in robots:
        robot.control_loop()
