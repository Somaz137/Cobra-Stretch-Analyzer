import math
import time


def calculate_angle(a, b, c):
    """
    Calculate angle at point b given points a, b, c.
    Points are (x, y)
    Returns angle in degrees
    """
    try:
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])

        dot_product = ba[0]*bc[0] + ba[1]*bc[1]
        mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
        mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

        if mag_ba * mag_bc == 0:
            return 0

        cos_angle = dot_product / (mag_ba * mag_bc)
        # Clamp value to avoid math domain error
        cos_angle = max(min(cos_angle, 1.0), -1.0)

        angle = math.degrees(math.acos(cos_angle))
        return angle
    except:
        return 0


class CobraAnalyzer:
    """
    Analyze landmarks to detect Cobra Stretch pose
    """

    def __init__(self, hold_threshold=5):
        """
        :param hold_threshold: seconds required for a valid hold
        """
        self.hold_threshold = hold_threshold
        # Store computed angles for display
        self.left_elbow_angle = 0
        self.right_elbow_angle = 0
        self.shoulder_angle = 0
        self.hip_angle = 0
        self.shoulder_lift_norm = 0
        self.torso_angle = 0
        self.pose_start_time = None

    def is_cobra_pose(self, landmarks):
        """
        Determine if the person is in Cobra pose
        :param landmarks: list of (id, x, y, z) from PoseDetector
        :return: (pose_detected: bool, quality: str)
        """

        if not landmarks:
            self.pose_start_time = None
            return False, "No landmarks"

        # MediaPipe landmark indices
        left_shoulder = landmarks[11][1:3]
        right_shoulder = landmarks[12][1:3]
        left_elbow = landmarks[13][1:3]
        right_elbow = landmarks[14][1:3]
        left_hip = landmarks[23][1:3]
        right_hip = landmarks[24][1:3]
        left_wrist = landmarks[15][1:3]
        right_wrist = landmarks[16][1:3]

        # Calculate angles
        # Use actual wrist landmarks to compute elbow angles (shoulder-elbow-wrist)
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        shoulder_angle = calculate_angle(left_hip, left_shoulder, right_shoulder)
        hip_angle = calculate_angle(left_shoulder, left_hip, right_hip)

        # Store angles for display
        self.left_elbow_angle = left_elbow_angle
        self.right_elbow_angle = right_elbow_angle
        self.shoulder_angle = shoulder_angle
        self.hip_angle = hip_angle

        # Shoulder lift metric: Compare wrist-to-shoulder distance vs 1.5x wrist-to-elbow distance
        # This measures how far the wrist has moved away from elbow (arm extension/lift)
        left_wrist_to_shoulder = math.hypot(left_wrist[0] - left_shoulder[0], 
                                            left_wrist[1] - left_shoulder[1]) + 1e-6
        left_wrist_to_elbow = math.hypot(left_wrist[0] - left_elbow[0], 
                                         left_wrist[1] - left_elbow[1]) + 1e-6
        left_lift_ratio = left_wrist_to_shoulder / (1.5 * left_wrist_to_elbow)
        
        right_wrist_to_shoulder = math.hypot(right_wrist[0] - right_shoulder[0], 
                                             right_wrist[1] - right_shoulder[1]) + 1e-6 #1e-6 to avoid div by 0
        right_wrist_to_elbow = math.hypot(right_wrist[0] - right_elbow[0], 
                                          right_wrist[1] - right_elbow[1]) + 1e-6
        right_lift_ratio = right_wrist_to_shoulder / (1.5 * right_wrist_to_elbow)
        
        # Average of both arms for overall lift metric
        shoulder_lift_norm = (left_lift_ratio + right_lift_ratio) / 2.0

        # Calculate shoulder and hip midpoints for torso angle
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2.0,
                        (left_shoulder[1] + right_shoulder[1]) / 2.0)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2.0,
                   (left_hip[1] + right_hip[1]) / 2.0)

        # Torso angle relative to vertical (0 = vertical upright) handle camera tilt
        torso_vec = (shoulder_mid[0] - hip_mid[0], shoulder_mid[1] - hip_mid[1])
        mag_torso = math.hypot(torso_vec[0], torso_vec[1]) + 1e-6
        # vertical vector pointing up in image coords
        vertical = (0.0, -1.0)
        dot = torso_vec[0] * vertical[0] + torso_vec[1] * vertical[1]
        cos_val = max(min(dot / (mag_torso * 1.0), 1.0), -1.0)
        torso_angle = math.degrees(math.acos(cos_val))

        # Store torso metrics for display
        self.shoulder_lift_norm = shoulder_lift_norm
        self.torso_angle = torso_angle

        # Simple thresholds for Cobra Stretch
        elbow_threshold = 160

        # Evaluate using elbow extension + wrist-to-shoulder lift ratio
        elbow_ok = (left_elbow_angle >= elbow_threshold) and (right_elbow_angle >= elbow_threshold)
        # With new metric: ratio of wrist-shoulder / (1.5 * wrist-elbow)
        # Values > 1.0 indicate good lift/extension. Typical range: 0.8-1.5
        torso_lift_threshold = 1.45
        torso_ok = (shoulder_lift_norm > torso_lift_threshold) or (torso_angle < 45.0)

        pose_detected = elbow_ok and torso_ok

        # Quality assessment
        if not pose_detected:
            quality = "Poor"
            self.pose_start_time = None
        else:
            # Immediate good if elbows are nearly straight
            elbow_instant_thresh = 165.0
            elbow_instant = (left_elbow_angle >= elbow_instant_thresh) and (right_elbow_angle >= elbow_instant_thresh)

            if elbow_instant and shoulder_lift_norm > 1.40:
                if self.pose_start_time is None:
                    self.pose_start_time = time.time()
                quality = "Good"
            else:
                # Track hold duration to upgrade to Good
                if self.pose_start_time is None:
                    self.pose_start_time = time.time()

                hold_time = time.time() - self.pose_start_time

                if hold_time >= self.hold_threshold:
                    quality = "Good"
                else:
                    quality = "Average"

        return pose_detected, quality
