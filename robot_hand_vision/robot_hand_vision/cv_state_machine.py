import rclpy
from rclpy.node import Node
from threading import Thread
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry as Odom
from sensor_msgs.msg import Image
import cv2
import math
from cv_bridge import CvBridge


class HandStateController(Node):
    """
    A ROS2 Node that controls various Neato behavior based on hand signs.
    """
    def __init__(self, image_topic):
        super().__init__('hand_state_controller')
        # create bridge between OpenCV and ROS ----------------------
        self.cv_image = None
        self.bridge = CvBridge()

        # create subscriptions and publishers -----------------------
        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(0.1, callback=self.run_loop)

        #create a thread to run the run loop on ---------------------
        thread = Thread(target=self.loop_wrapper)
        thread.start()

        #global teleop variables ------------------------------------
        self.teleop_direction = {0: 0.3, 1: -0.3, 2: 0}
        self.run_teleop = False

        #global spin variables --------------------------------------
        self.spin_angle = 360
        self.crnt_angle = None
        #global person follow variables -----------------------------

    def robot_angle(self, msg):
        """
        Converts the current angular pose of the robot to degrees.
        """
        w = msg.pose.pose.orientation.w
        self.crnt_angle = math.rad2deg(math.acos(w)*2)

    def get_gesture(self, image):
        return
    

    def run_loop(self):
        cap = cv2.VideoCapture(0)
        ret, self.cv_image = cap.read()

        cv2.imshow("hand state machine", self.cv_image)



def main():
    """
    Initializes and publishes the HandStateController node.
    """
    rclpy.init()

    handstate_publish = HandStateController()

    rclpy.spin(handstate_publish)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    handstate_publish.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()