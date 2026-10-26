# main.py

import cv2
import threading
import time
import config
from object_detection import ObjectDetector
from ocr_module import OCRReader
from voice_module import VoiceCommandProcessor
from audio_feedback import AudioFeedback
from emergency_alert import EmergencyAlert
import gps_server

class AssistiveSystem:
    def __init__(self):
        """Initialize all system modules."""
        print("[INFO] Initializing AI-Based Assistive System...")
        self.detector = ObjectDetector()
        self.ocr = OCRReader()
        self.audio = AudioFeedback()
        self.emergency = EmergencyAlert()
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.current_frame = None
        self.frame_count = 0
        self.mode = "auto"  # auto / manual

        # Start GPS server
        gps_server.start_server()

        # Initialize voice processor with command callback
        self.voice = VoiceCommandProcessor(self.handle_command)
        self.voice.start_continuous_listening()

        self.audio.announce(
            "Assistive system initialized. I am ready to help you.",
            priority=3
        )
        print("[INFO] System initialization complete.")

    def handle_command(self, command):
        """Handle recognized voice commands."""
        if command == "detect_objects":
            self._announce_objects()
        elif command == "read_text":
            self._read_text()
        elif command == "get_location":
            self._announce_location()
        elif command == "emergency_alert":
            self._send_emergency_alert()
        elif command == "stop":
            # Clear audio queue (simplified)
            self.audio.announce("Stopping.", priority=1)
        elif command == "help":
            help_text = ("Available commands: detect objects, read text, "
                         "my location, send alert, stop, and help.")
            self.audio.announce(help_text, priority=3)
        elif command == "describe_surroundings":
            self._announce_objects()

    def _announce_objects(self):
        """Detect and announce objects in current frame."""
        if self.current_frame is not None:
            detections = self.detector.detect(self.current_frame)
            announcement = self.detector.format_announcement(detections)
            self.audio.announce(announcement, priority=5)

    def _read_text(self):
        """Extract and announce text from current frame."""
        if self.current_frame is not None:
            self.audio.announce("Reading text. Please hold still.", priority=4)
            text = self.ocr.extract_text(self.current_frame)
            if text.strip():
                self.audio.announce(f"Text detected: {text}", priority=5)
            else:
                self.audio.announce("No text detected in current view.",
                                    priority=5)

    def _announce_location(self):
        """Retrieve and announce GPS location."""
        location = gps_server.get_latest_location()
        if location['latitude']:
            loc_text = (f"Your current location is latitude "
                        f"{location['latitude']:.4f}, longitude "
                        f"{location['longitude']:.4f}.")
        else:
            loc_text = ("Location not available. "
                        "Please ensure GPS tracking is active.")
        self.audio.announce(loc_text, priority=4)

    def _send_emergency_alert(self):
        """Send emergency SMS alert."""
        self.audio.announce("Sending emergency alert. Please wait.",
                            priority=1)
        success = self.emergency.send_alert()
        if success:
            self.audio.announce("Emergency alert sent successfully.",
                                priority=1)
        else:
            self.audio.announce("Failed to send emergency alert. "
                                "Please try again.", priority=1)

    def run(self):
        """Main processing loop."""
        print("[INFO] Starting main processing loop. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read from camera.")
                break

            self.current_frame = frame.copy()
            self.frame_count += 1

            # Periodic automatic object detection
            if (self.frame_count % config.DETECTION_INTERVAL == 0
                    and self.mode == "auto"):
                detections = self.detector.detect(frame)
                frame = self.detector.draw_detections(frame, detections)

                # Only announce if objects are very close (potential obstacle)
                close_objects = [d for d in detections
                                 if d['distance'] == 'very close']
                if close_objects:
                    announcement = self.detector.format_announcement(
                        close_objects
                    )
                    self.audio.announce(
                        f"Warning: {announcement}", priority=3
                    )

            cv2.imshow("Assistive System - Camera Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] System shutdown.")

if __name__ == "__main__":
    system = AssistiveSystem()
    system.run()
