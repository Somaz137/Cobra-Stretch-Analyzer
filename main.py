import cv2
from pose_detector import PoseDetector
from cobra_analyzer import CobraAnalyzer
import tkinter as tk
from tkinter import filedialog
import numpy as np
import time


class CobraGUI:
    """GUI for Cobra Analyzer with buttons"""
    
    BUTTON_COLOR = (100, 150, 200)
    TEXT_COLOR = (255, 255, 255)
    METRICS_COLOR = (100, 200, 255)
    
    def __init__(self):
        self.detector = PoseDetector()
        self.analyzer = CobraAnalyzer(hold_threshold=5)
        self.current_screen = "MENU" # MENU, LIVE_FEED, SELECT_VIDEO, VIDEO_PLAY, EXIT
        self.cap = cv2.VideoCapture(0)
        time.sleep(0.5)
        
    def _draw_button(self, frame, x, y, width, height, text):
        """Draw a button with centered text"""
        # -1 for filled
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.BUTTON_COLOR, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = x + (width - text_size[0]) // 2
        text_y = y + (height + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, self.TEXT_COLOR, thickness)
    
    def _draw_angle_labels(self, frame, landmarks):
        """Draw angle labels at the points where they are calculated"""
        if not landmarks:
            return
        
        landmark_dict = {lm[0]: (lm[1], lm[2]) for lm in landmarks}
        
        # Left elbow label at landmark 13
        if 13 in landmark_dict:
            x, y = landmark_dict[13]
            cv2.putText(frame, "L_elbow", (x - 40, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Right elbow label at landmark 14
        if 14 in landmark_dict:
            x, y = landmark_dict[14]
            cv2.putText(frame, "R_elbow", (x - 40, y - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Shoulder label at landmark 11
        if 11 in landmark_dict:
            x, y = landmark_dict[11]
            cv2.putText(frame, "Shoulder", (x - 50, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Hip label at landmark 23
        if 23 in landmark_dict:
            x, y = landmark_dict[23]
            cv2.putText(frame, "Hip", (x - 20, y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def _draw_hold_duration(self, frame, quality):
        """Display hold duration for Good/Average poses only"""
        if quality in ["Good", "Average"] and self.analyzer.pose_start_time:
            hold_time = time.time() - self.analyzer.pose_start_time
            duration_text = f"Hold: {hold_time:.1f}s"
            cv2.putText(frame, duration_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _draw_metrics_sidebar(self, composite):
        """Draw metrics sidebar with all measurements"""
        composite[0:480, 480:700] = (0, 0, 0)
        
        metrics_y = 20
        cv2.putText(composite, "METRICS", (490, metrics_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        metrics_y += 35
        cv2.putText(composite, f"L_Elbow: {self.analyzer.left_elbow_angle:.1f}", (490, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.METRICS_COLOR, 1)
        metrics_y += 25
        cv2.putText(composite, f"R_Elbow: {self.analyzer.right_elbow_angle:.1f}", (490, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.METRICS_COLOR, 1)
        
        metrics_y += 35
        cv2.putText(composite, f"Shoulder: {self.analyzer.shoulder_angle:.1f}", (490, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.METRICS_COLOR, 1)
        metrics_y += 25
        cv2.putText(composite, f"Hip: {self.analyzer.hip_angle:.1f}", (490, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.METRICS_COLOR, 1)
        
        metrics_y += 35
        cv2.putText(composite, f"Lift: {self.analyzer.shoulder_lift_norm:.2f}", (490, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.METRICS_COLOR, 1)
        metrics_y += 25
        cv2.putText(composite, f"Torso: {self.analyzer.torso_angle:.1f}", (490, metrics_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.METRICS_COLOR, 1)
    
    def _is_clicked(self, mouse_x, mouse_y, button_x, button_y, width, height):
        """Check if a button was clicked"""
        return (button_x <= mouse_x <= button_x + width and 
                button_y <= mouse_y <= button_y + height)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for button clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_screen == "MENU":
                # Live Feed button
                if self._is_clicked(x, y, 150, 150, 300, 80):
                    self.current_screen = "LIVE_FEED"
                # Recorded Video button
                elif self._is_clicked(x, y, 150, 280, 300, 80):
                    self.current_screen = "SELECT_VIDEO"
                # Exit button
                elif self._is_clicked(x, y, 150, 410, 300, 80):
                    self.current_screen = "EXIT"
            
            elif self.current_screen == "VIDEO_PLAY":
                # Back button
                if self._is_clicked(x, y, 10, 500, 340, 80):
                    self.current_screen = "MENU"
                    if self.cap:
                        self.cap.release()
                        self.cap = None
                
                # Exit button
                elif self._is_clicked(x, y, 360, 500, 330, 80):
                    self.current_screen = "EXIT"
                    if self.cap:
                        self.cap.release()
            
            elif self.current_screen == "LIVE_FEED":
                # Back button
                if self._is_clicked(x, y, 10, 500, 670, 80):
                    self.current_screen = "MENU"
                    if self.cap:
                        self.cap.release()
                        self.cap = None
    
    def _show_menu(self):
        """Display main menu screen"""
        frame = np.ones((600, 600, 3), dtype=np.uint8) * 240
        
        cv2.putText(frame, "COBRA ANALYZER", (100, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        self._draw_button(frame, 150, 150, 300, 80, "LIVE FEED")
        self._draw_button(frame, 150, 280, 300, 80, "RECORDED VIDEO")
        self._draw_button(frame, 150, 410, 300, 80, "EXIT")
        
        return frame
    
    def _show_live_feed(self):
        """Show live webcam feed with pose detection (always active)"""
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        
        ret, frame = self.cap.read()
        
        # Create composite frame: video + sidebar + buttons
        composite = np.ones((600, 700, 3), dtype=np.uint8) * 240
        
        if not ret:
            # Error message in video area
            error_frame = np.ones((480, 480, 3), dtype=np.uint8) * 100
            cv2.putText(error_frame, "ERROR: Cannot", (80, 220),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(error_frame, "access webcam", (80, 260),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            composite[0:480, 0:480] = error_frame
        else:
            # Resize frame to fit video area
            frame_resized = cv2.resize(frame, (480, 480))
            
            # Pose detection - draw default skeleton
            landmarks = self.detector.detect(frame_resized, draw=True)
            _, quality = self.analyzer.is_cobra_pose(landmarks)
            
            # Draw angle labels at calculation points
            self._draw_angle_labels(frame_resized, landmarks)
            
            # Display quality indicator
            if quality == "Good":
                color = (0, 255, 0)
            elif quality == "Average":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.putText(frame_resized, f"Pose: {quality}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display hold duration for good/average poses
            self._draw_hold_duration(frame_resized, quality)
            
            # Place video in composite
            composite[0:480, 0:480] = frame_resized
        
        # Draw metrics sidebar
        self._draw_metrics_sidebar(composite)
        
        # Buttons at bottom
        self._draw_button(composite, 10, 500, 670, 80, "BACK")
        
        return composite
    
    def _show_video_player(self):
        """Show recorded video player with pose detection always active"""
        if not self.cap:
            return np.ones((600, 700, 3), dtype=np.uint8) * 100
        
        # Create composite frame: video + sidebar + buttons
        composite = np.ones((600, 700, 3), dtype=np.uint8) * 240
        
        # Always advance frames
        ret, frame = self.cap.read()
        if not ret:
            error_frame = np.ones((480, 480, 3), dtype=np.uint8) * 100
            cv2.putText(error_frame, "VIDEO ENDED", (140, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            composite[0:480, 0:480] = error_frame
        else:
            # Resize frame to fit video area
            frame_resized = cv2.resize(frame, (480, 480))
            
            # Detect pose - draw default skeleton
            landmarks = self.detector.detect(frame_resized, draw=True)
            _, quality = self.analyzer.is_cobra_pose(landmarks)
            
            # Draw angle labels at calculation points
            self._draw_angle_labels(frame_resized, landmarks)
            
            # Display quality indicator
            if quality == "Good":
                color = (0, 255, 0)
            elif quality == "Average":
                color = (0, 165, 255)
            else:
                color = (0, 0, 255)
            
            cv2.putText(frame_resized, f"Pose: {quality}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display hold duration for good/average poses
            self._draw_hold_duration(frame_resized, quality)
            
            # Place video in composite
            composite[0:480, 0:480] = frame_resized
        
        # Draw metrics sidebar
        self._draw_metrics_sidebar(composite)
        
        # Buttons at bottom - only BACK and EXIT
        self._draw_button(composite, 10, 500, 340, 80, "BACK")
        self._draw_button(composite, 360, 500, 330, 80, "EXIT")
        
        return composite
    
    def run(self):
        """Main application loop"""
        cv2.namedWindow("Cobra Analyzer")
        cv2.setMouseCallback("Cobra Analyzer", self._mouse_callback)
        
        while True:
            if self.current_screen == "MENU":
                frame = self._show_menu()
            
            elif self.current_screen == "LIVE_FEED":
                frame = self._show_live_feed()
            
            elif self.current_screen == "SELECT_VIDEO":
                root = tk.Tk()
                root.withdraw()
                video_path = filedialog.askopenfilename(
                    filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
                )
                root.destroy()
                
                if video_path:
                    self.video_path = video_path
                    self.cap = cv2.VideoCapture(video_path)
                    self.current_screen = "VIDEO_PLAY"
                else:
                    self.current_screen = "MENU"
                continue
            
            elif self.current_screen == "VIDEO_PLAY":
                frame = self._show_video_player()
            
            elif self.current_screen == "EXIT":
                break
            
            cv2.imshow("Cobra Analyzer", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC to exit
                break
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = CobraGUI()
    app.run()
