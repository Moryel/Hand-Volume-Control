"""
Advanced Hand Gesture Control System
Lead Computer Vision Engineer Implementation

Features:
- State Machine (LOCKED/ACTIVE modes with gesture transitions)
- Adaptive Depth Scaling (works at any distance from camera)
- Media Control (Play/Pause with debounce)
- Visual Feedback (Volume bar, state indicators, FPS counter)

Author: Yuksel
"""

import cv2
import numpy as np
import time
import platform
import sys

# Platform-specific imports
if platform.system() == "Windows":
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        USE_PYCAW = True
    except ImportError:
        print("Warning: pycaw not available. Install with: pip install pycaw")
        USE_PYCAW = False
else:
    USE_PYCAW = False

# Import for media control
try:
    import pyautogui
    HAS_PYAUTOGUI = True
except ImportError:
    print("Warning: pyautogui not available. Install with: pip install pyautogui")
    HAS_PYAUTOGUI = False

# MediaPipe import - using the tasks API for newer versions
try:
    import mediapipe
    from mediapipe import Image as mpImage
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except Exception as e:
    print(f"Error: MediaPipe not available: {e}")
    print("Install with: pip install mediapipe")
    HAS_MEDIAPIPE = False
    sys.exit(1)


class StateManager:
    """Manages LOCKED/ACTIVE state transitions"""
    
    LOCKED = "LOCKED"
    ACTIVE = "ACTIVE"
    
    def __init__(self):
        self.current_state = self.LOCKED
        self.state_change_time = time.time()
        self.state_cooldown = 0.5  # Prevent rapid state changes
    
    def can_change_state(self):
        """Check if enough time has passed since last state change"""
        return (time.time() - self.state_change_time) > self.state_cooldown
    
    def set_state(self, new_state):
        """Change state with cooldown protection"""
        if self.current_state != new_state and self.can_change_state():
            self.current_state = new_state
            self.state_change_time = time.time()
            print(f"State changed to: {new_state}")
            return True
        return False
    
    def is_active(self):
        return self.current_state == self.ACTIVE
    
    def is_locked(self):
        return self.current_state == self.LOCKED


class GestureDetector:
    """Advanced gesture recognition with depth-invariant calculations"""
    
    def __init__(self):
        # Create MediaPipe HandLandmarker
        base_options = python.BaseOptions(model_asset_buffer=self._get_model())
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Landmark IDs
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        self.MIDDLE_MCP = 9  # Middle finger base
        
        # Video timestamp counter
        self.frame_timestamp_ms = 0
    
    def _get_model(self):
        """Download and load the hand landmarker model"""
        import urllib.request
        import os
        
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully")
        
        with open(model_path, 'rb') as f:
            return f.read()
    
    def process_frame(self, frame):
        """Process frame and return hand landmarks"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # MediaPipe Image expects contiguous numpy array
        rgb_frame = np.ascontiguousarray(rgb_frame)
        
        # Create MediaPipe Image (format 1 = SRGB)
        mp_image = mpImage(image_format=1, data=rgb_frame)
        
        # Detect hands
        self.frame_timestamp_ms += 33  # Approximate 30fps
        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        
        return result
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks"""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def is_finger_extended(self, landmarks, finger_name):
        """Check if a finger is extended based on landmark positions"""
        if finger_name == 'thumb':
            # Thumb: Check if tip is far from palm center
            thumb_tip = landmarks[4]
            thumb_mcp = landmarks[2]
            palm_center = landmarks[0]
            dist_tip = self.calculate_distance(thumb_tip, palm_center)
            dist_mcp = self.calculate_distance(thumb_mcp, palm_center)
            return dist_tip > dist_mcp * 1.2
        else:
            # Other fingers: Compare tip Y-position with PIP joint
            tips = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
            pips = {'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}
            
            tip_y = landmarks[tips[finger_name]].y
            pip_y = landmarks[pips[finger_name]].y
            
            # Finger is extended if tip is above (lower y value) PIP
            return tip_y < pip_y - 0.05
    
    def get_finger_states(self, landmarks):
        """Get extension state for all fingers"""
        return {
            'thumb': self.is_finger_extended(landmarks, 'thumb'),
            'index': self.is_finger_extended(landmarks, 'index'),
            'middle': self.is_finger_extended(landmarks, 'middle'),
            'ring': self.is_finger_extended(landmarks, 'ring'),
            'pinky': self.is_finger_extended(landmarks, 'pinky')
        }
    
    def detect_open_palm(self, finger_states):
        """All 5 fingers extended"""
        return all(finger_states.values())
    
    def detect_fist(self, finger_states):
        """All fingers folded"""
        return not any(finger_states.values())
    
    def detect_peace_sign(self, finger_states):
        """Only index and middle fingers extended"""
        return (finger_states['index'] and 
                finger_states['middle'] and 
                not finger_states['ring'] and 
                not finger_states['pinky'])
    
    def calculate_depth_invariant_ratio(self, landmarks):
        """
        Calculate pinch ratio normalized by hand size
        This makes volume control work at any distance from camera
        """
        # Reference length: Wrist to Middle MCP (hand size)
        wrist = landmarks[self.WRIST]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        reference_length = self.calculate_distance(wrist, middle_mcp)
        
        if reference_length < 0.01:  # Avoid division by zero
            return 0.0
        
        # Pinch length: Thumb tip to Index tip
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        pinch_length = self.calculate_distance(thumb_tip, index_tip)
        
        # Calculate ratio (depth-invariant)
        ratio = pinch_length / reference_length
        
        return ratio
    
    def ratio_to_volume(self, ratio):
        """
        Map ratio to volume percentage
        Ratio < 0.2 = 0% (fingers touching)
        Ratio > 1.3 = 100% (fingers fully spread)
        """
        # Clamp ratio to expected range
        ratio = np.clip(ratio, 0.2, 1.3)
        
        # Linear interpolation
        volume = np.interp(ratio, [0.2, 1.3], [0, 100])
        
        return int(volume)
    
    def draw_landmarks(self, frame, landmarks):
        """Draw hand landmarks on frame"""
        h, w, _ = frame.shape
        
        # Draw connections
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        # Draw lines
        for connection in connections:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))
            
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw landmarks
        for landmark in landmarks:
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)


class VolumeController:
    """Cross-platform volume control"""
    
    def __init__(self):
        self.platform = platform.system()
        self.volume_interface = None
        
        if self.platform == "Windows" and USE_PYCAW:
            self._init_windows()
        
        # Volume smoothing
        self.volume_history = []
        self.history_size = 5
    
    def _init_windows(self):
        """Initialize Windows volume control"""
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume_interface = cast(interface, POINTER(IAudioEndpointVolume))
            print("Windows volume control initialized")
        except Exception as e:
            print(f"Failed to initialize Windows volume: {e}")
    
    def set_volume(self, volume_percent):
        """Set system volume (0-100)"""
        # Apply smoothing
        self.volume_history.append(volume_percent)
        if len(self.volume_history) > self.history_size:
            self.volume_history.pop(0)
        
        smooth_volume = int(np.mean(self.volume_history))
        
        if self.platform == "Windows" and self.volume_interface:
            try:
                # Convert to 0.0-1.0 range
                volume_scalar = smooth_volume / 100.0
                self.volume_interface.SetMasterVolumeLevelScalar(volume_scalar, None)
            except Exception as e:
                print(f"Error setting volume: {e}")
        
        elif self.platform == "Linux":
            try:
                # Try amixer first
                import subprocess
                subprocess.call(['amixer', '-D', 'pulse', 'sset', 'Master', f'{smooth_volume}%'],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except:
                try:
                    # Fallback to pactl
                    subprocess.call(['pactl', 'set-sink-volume', '@DEFAULT_SINK@', f'{smooth_volume}%'],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except:
                    pass
        
        return smooth_volume


class MediaController:
    """Media playback control with debounce"""
    
    def __init__(self):
        self.last_toggle_time = 0
        self.debounce_duration = 1.0  # 1 second cooldown
    
    def can_toggle(self):
        """Check if enough time has passed since last toggle"""
        return (time.time() - self.last_toggle_time) > self.debounce_duration
    
    def toggle_play_pause(self):
        """Toggle media play/pause with debounce"""
        if HAS_PYAUTOGUI and self.can_toggle():
            try:
                pyautogui.press('playpause')
                self.last_toggle_time = time.time()
                print("Media toggled: Play/Pause")
                return True
            except Exception as e:
                print(f"Error toggling media: {e}")
        return False


class VisualFeedback:
    """Enhanced visual feedback system"""
    
    def __init__(self):
        self.fps_history = []
        self.fps_history_size = 30
    
    def draw_state_indicator(self, frame, state_manager):
        """Draw current state (LOCKED/ACTIVE) with color coding"""
        state_text = state_manager.current_state
        
        if state_manager.is_locked():
            color = (0, 0, 255)  # Red for LOCKED
            instruction = "Make Open Palm to UNLOCK"
        else:
            color = (0, 255, 0)  # Green for ACTIVE
            instruction = "Make Fist to LOCK"
        
        # Draw state
        cv2.putText(frame, state_text, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw instruction
        cv2.putText(frame, instruction, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_volume_bar(self, frame, volume):
        """Draw dynamic volume bar"""
        h, w, _ = frame.shape
        
        # Bar dimensions
        bar_x = w - 60
        bar_y = 100
        bar_width = 40
        bar_height = 300
        
        # Draw background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        
        # Draw filled portion based on volume
        fill_height = int((volume / 100) * bar_height)
        fill_y = bar_y + bar_height - fill_height
        
        # Color gradient based on volume
        if volume < 33:
            color = (0, 255, 0)  # Green
        elif volume < 66:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 140, 255)  # Orange
        
        cv2.rectangle(frame, (bar_x, fill_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     color, -1)
        
        # Draw border
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Draw volume text
        cv2.putText(frame, f"{volume}%", (bar_x - 10, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def draw_connection_line(self, frame, landmarks, h, w):
        """Draw line between thumb and index finger when active"""
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Convert normalized coordinates to pixel coordinates
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
        
        # Draw line
        cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), 
                (255, 0, 255), 3)
        
        # Draw circles at fingertips
        cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 255), -1)
        cv2.circle(frame, (index_x, index_y), 10, (255, 0, 255), -1)
    
    def draw_fps(self, frame, fps):
        """Draw FPS counter"""
        # Update FPS history for smoothing
        self.fps_history.append(fps)
        if len(self.fps_history) > self.fps_history_size:
            self.fps_history.pop(0)
        
        avg_fps = int(np.mean(self.fps_history))
        
        h, w, _ = frame.shape
        cv2.putText(frame, f"FPS: {avg_fps}", (w - 150, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def draw_gesture_info(self, frame, gesture_name):
        """Draw current gesture name"""
        if gesture_name:
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


def initialize_camera():
    """Initialize camera with robust error handling"""
    for cam_index in [0, 1]:
        cap = cv2.VideoCapture(cam_index)
        if cap.isOpened():
            # Set camera properties for better performance
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print(f"Camera initialized successfully (index: {cam_index})")
            return cap
    
    raise RuntimeError("Failed to initialize camera. No camera found at index 0 or 1.")


def main():
    """Main application loop"""
    print("=" * 60)
    print("Advanced Hand Gesture Control System")
    print("=" * 60)
    print("\nGestures:")
    print("  - Open Palm (5 fingers) → UNLOCK")
    print("  - Fist (closed hand) → LOCK")
    print("  - Peace Sign (2 fingers) → Play/Pause Media")
    print("  - Pinch (Thumb + Index) → Volume Control (when ACTIVE)")
    print("\nPress 'q' to quit")
    print("=" * 60)
    
    # Initialize components
    try:
        cap = initialize_camera()
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print("Initializing hand tracking...")
    state_manager = StateManager()
    gesture_detector = GestureDetector()
    volume_controller = VolumeController()
    media_controller = MediaController()
    visual_feedback = VisualFeedback()
    print("System ready!")
    
    # FPS tracking
    prev_time = time.time()
    
    # Main loop
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame from camera")
            break
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Process frame with gesture detector
        result = gesture_detector.process_frame(frame)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 30
        prev_time = current_time
        
        # Default values
        current_volume = 0
        current_gesture = None
        
        # Process hand landmarks
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                # Draw hand skeleton
                gesture_detector.draw_landmarks(frame, hand_landmarks)
                
                # Get finger states
                finger_states = gesture_detector.get_finger_states(hand_landmarks)
                
                # Detect gestures
                is_open_palm = gesture_detector.detect_open_palm(finger_states)
                is_fist = gesture_detector.detect_fist(finger_states)
                is_peace = gesture_detector.detect_peace_sign(finger_states)
                
                # State transitions (always check, regardless of current state)
                if is_open_palm:
                    state_manager.set_state(StateManager.ACTIVE)
                    current_gesture = "Open Palm (UNLOCK)"
                elif is_fist:
                    state_manager.set_state(StateManager.LOCKED)
                    current_gesture = "Fist (LOCK)"
                
                # Actions only when ACTIVE
                if state_manager.is_active():
                    # Media control (Peace sign)
                    if is_peace:
                        if media_controller.toggle_play_pause():
                            current_gesture = "Peace Sign (Play/Pause)"
                    
                    # Volume control (Pinch) - only if NOT doing other gestures
                    elif not is_open_palm and not is_fist:
                        # Calculate depth-invariant ratio
                        ratio = gesture_detector.calculate_depth_invariant_ratio(hand_landmarks)
                        volume = gesture_detector.ratio_to_volume(ratio)
                        
                        # Set volume
                        current_volume = volume_controller.set_volume(volume)
                        
                        # Draw connection line
                        visual_feedback.draw_connection_line(frame, hand_landmarks, h, w)
                        
                        current_gesture = f"Pinch (Volume: {current_volume}%)"
        
        # Draw visual feedback
        visual_feedback.draw_state_indicator(frame, state_manager)
        visual_feedback.draw_volume_bar(frame, current_volume)
        visual_feedback.draw_fps(frame, fps)
        
        if current_gesture:
            visual_feedback.draw_gesture_info(frame, current_gesture)
        
        # Display frame
        cv2.imshow("Hand Gesture Control", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nExiting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed successfully")


if __name__ == "__main__":
    main()

