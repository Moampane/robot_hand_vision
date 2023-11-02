import rclpy
from rclpy.node import Node
from threading import Thread
import pickle
import mediapipe as mp
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry as Odom
from sensor_msgs.msg import Image
import cv2
import math
from cv_bridge import CvBridge
import time


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

        #OpenCV variables -------------------------------------------
        self.cap = cv2.VideoCapture(0)

        #Hand Classification Variables -------------------------------
        self.hand_prediction = None

    def process_image(self, msg):
        """Process image messages from ROS and stash them in an attribute
        called cv_image for subsequent processing"""
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def loop_wrapper(self):
        """This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2"""
        cv2.namedWindow("video_window")
        while True:
            self.classify_hand(model_dict=pickle.load(open("src/robot_hand_vision/robot_hand_vision/new_model.p", "rb")), cap=self.cv_image)
            print(self.hand_prediction)
            self.run_loop()
            time.sleep(0.1)
    
    def classify_hand(self, mp_drawing=mp.solutions.drawing_utils, mp_hands=mp.solutions.hands, model_dict=None, gesture_threshold=0.1, cap=None):
        model = model_dict['model']
        labels_dict = {
        0: "Tog-Teleop",
        1: "Forward",
        2: "Backwards",
        3: "Right",
        4: "Left",
        5: "Stop",
        6: "Spin",
        7: "P-Following",
        8: "Gojo",
        }
        with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
            data_aux = []
            x_ = []
            y_ = []
            frame = cap
            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(frame_rgb)


            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    x_min = self.get_min_dim(hand_landmarks, "x")
                    y_min = self.get_min_dim(hand_landmarks, "y")
                    x_max = self.get_max_dim(hand_landmarks, "x")
                    y_max = self.get_max_dim(hand_landmarks, "y")

                    for i in range(len(hand_landmarks.landmark)):
                        x = (hand_landmarks.landmark[i].x - x_min) / (x_max - x_min)
                        y = (hand_landmarks.landmark[i].y - y_min) / (y_max - y_min)
                        data_aux.append(x)
                        data_aux.append(y)

                x1 = int(x_min * W)
                y1 = int(y_min * H)

                x2 = int(x_max * W)
                y2 = int(y_max * H)

                try:
                    prediction = list(model.predict_proba([np.asarray(data_aux)])[0])
                    max_percent = max(prediction)

                    if max_percent > gesture_threshold:
                        max_idx = prediction.index(max_percent)
                        predicted_gesture = labels_dict[max_idx]
                        self.hand_prediction = max_idx

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            frame,
                            f"{predicted_gesture} {max_percent*100}%",
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )

                    else:
                        self.hand_prediction = 10
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            frame,
                            f"N/A {max_percent*100}%",
                            (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )

                except ValueError:
                    self.hand_prediction = 11
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(
                        frame,
                        "Gojo",
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )

            cv2.imshow("Hand Gesture Classifier", frame)
    
    # get max x or y dimension of a hand
    def get_max_dim(self,hand, dim):
        if dim == 'y':
            return max([landmark.y for landmark in hand.landmark])
        elif dim == 'x':
            return max([landmark.x for landmark in hand.landmark])
    
    # get min x or y dimension of a hand
    def get_min_dim(self,hand, dim):
        if dim == 'y':
            return min([landmark.y for landmark in hand.landmark])
        elif dim == 'x':
            return min([landmark.x for landmark in hand.landmark])

    def robot_angle(self, msg):
        """
        Converts the current angular pose of the robot to degrees.
        """
        w = msg.pose.pose.orientation.w
        self.crnt_angle = math.rad2deg(math.acos(w)*2)

    
    def run_loop(self):
        if not self.cv_image is None:
            cv2.imshow("video_window", self.cv_image)
            cv2.waitKey(5)





if __name__ == "__main__":
    node = HandStateController("/camera/image_raw")
    node.run()

def main():
    """
    Initializes and publishes the HandStateController node.
    """
    rclpy.init()

    handstate_publish = HandStateController('/camera/image_raw')
    rclpy.spin(handstate_publish)
    handstate_publish.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()