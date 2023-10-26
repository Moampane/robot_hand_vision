import rclpy
from rclpy.node import Node
from math import acos, sqrt
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class FiniteStateController(Node):
    def __init__(self, image_topic):
        super().__init__("finite_state_controller")
        self.camera_sub = self.create_subscription(
            Image, image_topic, self.process_image, 10
        )


def main():
    fsm_publisher = FiniteStateController()
    rclpy.init()
    rclpy.spin(fsm_publisher)
    fsm_publisher.destroy_node()
    rclpy.shutdown()
    print("Hi from robot_hand_vision.")


if __name__ == "__main__":
    main()
