import cv2
import mediapipe as mp


class PoseDetector:
    """
    MediaPipe Pose wrapper.
    Responsibility:
    - Detect human pose landmarks
    - Return landmark positions
    """

    def __init__(self, static_image_mode=False, smooth_landmarks=True, detection_confidence=0.5, tracking_confidence=0.5):
        #self.mp_pose → shortcut to Pose module.
        self.mp_pose = mp.solutions.pose
        #self.mp_draw → shortcut to drawing utilities to overlay skeletons.
        self.mp_draw = mp.solutions.drawing_utils
        #Instantiates the actual MediaPipe Pose detector with the settings above.
        self.pose = self.mp_pose.Pose(static_image_mode=static_image_mode,
                                        smooth_landmarks=smooth_landmarks,
                                        min_detection_confidence=detection_confidence,
                                        min_tracking_confidence=tracking_confidence)

        #Method to process an image or frame, detect landmarks, and optionally draw the skeleton.
    def detect(self, frame, draw=True):
        """
        Detect pose landmarks in a BGR frame.

        :param frame: OpenCV BGR image
        :param draw: Draw landmarks on frame
        :return: list of landmarks [(id, x, y, z)]
        """

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Runs the pose detection model.
        results = self.pose.process(img_rgb)
        #Empty list to store landmarks in (id, x, y, z) format.
        landmarks = []
        #results.pose_landmarks contains detected landmarks (33 points) if a person is found.
        if results.pose_landmarks:
            h, w, _ = frame.shape

            for idx, lm in enumerate(results.pose_landmarks.landmark):
                x = int(lm.x * w)
                y = int(lm.y * h)
                z = lm.z
                landmarks.append((idx, x, y, z))

            if draw:
                #Draws points and lines connecting joints.
                self.mp_draw.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return landmarks
