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
from gtts import gTTS
from playsound import playsound


class HandStateController(Node):
    """
    A ROS2 Node that controls various Neato behavior based on hand signs.
    """

    def __init__(self, image_topic):
        super().__init__("hand_state_controller")
        # create bridge between OpenCV and ROS ----------------------
        self.bridge = CvBridge()
        self.cv_image = None
        # create subscriptions and publishers -----------------------
        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.vel_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # create a thread to run the run loop on ---------------------
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        soundthread = Thread(target=self.play_sound)
        soundthread.start()
        # global teleop variables ------------------------------------
        self.teleop_direction = {1: 0.3, 2: -0.3, 3: 0.3, 4: -0.3, 5: 0.0}
        self.run_teleop = False
        self.toggle_counter = 0  # counter to ensure that teleop pose is held for a minimum period prior to changing mode.

        # global spin variables --------------------------------------
        self.spin_angle = 360
        self.crnt_angle = None
        # global person follow variables -----------------------------

        # OpenCV variables -------------------------------------------
        self.cap = cv2.VideoCapture(0)

        # Hand Classification Variables -------------------------------
        self.hand_prediction = 5
        self.prev_prediction = None
        self.action_text = {
            0: "Toggling Teleop, Save Driving!",
            1: "Moving Forward",
            2: "Moving Backwards",
            3: "Turning Right",
            4: "Turning Left",
            5: "Stop",
            6: "Now Spinning",
            7: "I Love Daft Punk. Activating the jukebox!",
            8: "Gojo",
        }

    def process_image(self, msg):
        """Process image messages from ROS and stash them in an attribute
        called cv_image for subsequent processing"""
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def loop_wrapper(self):
        """This function takes care of calling the run_loop function repeatedly.
        We are using a separate thread to run the loop_wrapper to work around
        issues with single threaded executors in ROS2"""
        cv2.namedWindow("video_window")
        self.msg = Twist()
        while True:
            # Classify Hand Pose
            self.classify_hand(
                model_dict=pickle.load(
                    open("src/robot_hand_vision/robot_hand_vision/new_model.p", "rb")
                ),
                cap=self.cv_image,
            )

            # Keep count of the amount of times a gesture is repeated
            if self.hand_prediction == self.prev_prediction:
                self.toggle_counter += 1
            else:
                self.toggle_counter = 0

            # If teleop toggle gesture has been held, toggle teleop
            self.check_teleop_toggle()

            # Run teleop code
            if self.run_teleop is True:
                self.teleop()
            if self.hand_prediction == 8 and self.toggle_counter == 20:
                self.play_sound(self.hand_prediction)
            if self.hand_prediction == 7 and self.toggle_counter == 20:
                self.play_sound(self.hand_prediction)

            self.prev_prediction = self.hand_prediction
            self.run_loop()
            time.sleep(0.02)

    def classify_hand(
        self,
        mp_drawing=mp.solutions.drawing_utils,
        mp_hands=mp.solutions.hands,
        model_dict=None,
        gesture_threshold=0.1,
        cap=None,
    ):
        """
        Classifies a recognized hand as a hand gesture and sets class variable hand_prediction to the classified index.

        Args:
            mp_drawing (drawing_utils): Class used to draw in hand landmarks and connections.
            mp_hands (hands): Class that stores hand landmarks and connections.
            model_dict (dictionary): A dictionary containing the trained hand gesture classification model.
            gesture_threshold (float): A float determining how confident the model needs to be for a classification.
            cap (VideoCapture): Neato's video.

        """
        model = model_dict["model"]
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
        with mp_hands.Hands(
            min_detection_confidence=0.6, min_tracking_confidence=0.4
        ) as hands:
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
    def get_max_dim(self, hand, dim):
        if dim == "y":
            return max([landmark.y for landmark in hand.landmark])
        elif dim == "x":
            return max([landmark.x for landmark in hand.landmark])

    # get min x or y dimension of a hand
    def get_min_dim(self, hand, dim):
        if dim == "y":
            return min([landmark.y for landmark in hand.landmark])
        elif dim == "x":
            return min([landmark.x for landmark in hand.landmark])

    def teleop(self):
        if self.hand_prediction in self.teleop_direction.keys():
            direction = self.teleop_direction[self.hand_prediction]
            if self.hand_prediction == 1 or self.hand_prediction == 2:
                if self.toggle_counter == 10:
                    self.msg.linear.x = direction
                    self.play_sound(self.hand_prediction)
            elif self.hand_prediction == 3 or self.hand_prediction == 4:
                if self.toggle_counter == 10:
                    self.msg.angular.z = direction
                    self.play_sound(self.hand_prediction)
            else:
                if self.toggle_counter == 5:
                    self.msg.angular.z = 0.0
                    self.msg.linear.x = 0.0
                    self.play_sound(self.hand_prediction)
        else:
            self.msg.angular.z = 0.0
            self.msg.linear.x = 0.0

    def check_teleop_toggle(self):
        if self.hand_prediction == 0 and self.toggle_counter == 10:
            if self.run_teleop is False:
                self.run_teleop = True
                self.play_sound(self.hand_prediction)
            else:
                self.run_teleop = False

    def robot_angle(self, msg):
        """
        Converts the current angular pose of the robot to degrees.
        """
        w = msg.pose.pose.orientation.w
        self.crnt_angle = math.rad2deg(math.acos(w) * 2)

    def play_sound(self, hand_prediction):
        print(hand_prediction)
        language = "en"
        text = self.action_text[hand_prediction]
        print(text)
        txttospeech = gTTS(text=text, lang=language, slow=False, tld="ie")
        txttospeech.save(
            f"src/robot_hand_vision/robot_hand_vision/robot_hand_vision/Sounds/{hand_prediction}.mp3"
        )
        if hand_prediction == 7:
            playsound(
                f"src/robot_hand_vision/robot_hand_vision/robot_hand_vision/Sounds/{hand_prediction}.mp3"
            )
            playsound(
                "src/robot_hand_vision/robot_hand_vision/robot_hand_vision/daft_punk.mp3",
                block=False,
            )
        elif hand_prediction == 8:
            playsound(
                f"src/robot_hand_vision/robot_hand_vision/robot_hand_vision/Sounds/{hand_prediction}.mp3"
            )
            playsound(
                "src/robot_hand_vision/robot_hand_vision/robot_hand_vision/domain.mp3",
                block=False,
            )
            print("this block")
        else:
            playsound(
                f"src/robot_hand_vision/robot_hand_vision/robot_hand_vision/Sounds/{hand_prediction}.mp3",
                block=False,
            )

    def run_loop(self):
        self.vel_pub.publish(self.msg)
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

    handstate_publish = HandStateController("/camera/image_raw")
    rclpy.spin(handstate_publish)
    handstate_publish.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
