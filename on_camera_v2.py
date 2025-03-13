import cv2
import numpy as np
import tensorflow as tf
import os
from collections import deque

class DigitDetector:
    def __init__(self, model_path='saved_model/my_model.keras'):
        """Initialize the digit detector"""
        self.model = self._load_model(model_path)
        self.prediction_queue = deque(maxlen=5)  # Increased smoothing window
        self.min_contour_area = 1000  # Increased minimum area
        self.confidence_threshold = 0.6  # Increased confidence threshold
        
        # Initialize display settings
        cv2.namedWindow('Main View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processing View', cv2.WINDOW_NORMAL)
        
    def _load_model(self, model_path):
        """Load the TensorFlow model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        return tf.keras.models.load_model(model_path)
    
    def preprocess_frame(self, frame):
        """Preprocess the frame for digit detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better handling of lighting variations
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )
        
        # Apply morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return thresh
    
    def find_digit_contours(self, thresh):
        """Find and filter digit contours"""
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Calculate contour properties for better filtering
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # More strict filtering criteria
            if (0.2 <= aspect_ratio <= 1.2 and  # Slightly relaxed aspect ratio
                solidity > 0.5):  # Must be reasonably solid
                valid_contours.append((contour, area, (x, y, w, h)))
        
        return valid_contours
    
    def prepare_digit_image(self, frame, bbox):
        """Prepare digit image for classification"""
        x, y, w, h = bbox
        
        # Add padding
        padding = int(max(w, h) * 0.2)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2*padding)
        h = min(frame.shape[0] - y, h + 2*padding)
        
        # Extract and preprocess region
        digit_region = frame[y:y+h, x:x+w]
        
        # Convert to grayscale if needed
        if len(digit_region.shape) == 3:
            digit_region = cv2.cvtColor(digit_region, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        digit_region = cv2.resize(digit_region, (28, 28))
        
        # Normalize
        digit_region = digit_region.astype('float32') / 255.0
        
        # Reshape for model (flatten to 784)
        digit_region = digit_region.reshape(1, 784)
        
        return digit_region
    
    def smooth_predictions(self, digit, confidence):
        """Smooth predictions over time"""
        self.prediction_queue.append((digit, confidence))
        
        if len(self.prediction_queue) < self.prediction_queue.maxlen:
            return None, 0
        
        # Count predictions and average confidence
        pred_counts = {}
        for d, c in self.prediction_queue:
            if d not in pred_counts:
                pred_counts[d] = [0, 0]
            pred_counts[d][0] += 1
            pred_counts[d][1] += c
        
        # Get most common prediction with highest confidence
        most_common = max(
            pred_counts.items(),
            key=lambda x: (x[1][0], x[1][1]/x[1][0])
        )
        
        digit = most_common[0]
        confidence = most_common[1][1] / most_common[1][0]
        
        return digit, confidence
    
    def draw_results(self, frame, thresh, contours_info):
        """Draw detection results"""
        # Create displays
        main_display = frame.copy()
        debug_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        for contour, _, bbox in contours_info:
            x, y, w, h = bbox
            
            # Prepare digit image and get prediction
            digit_image = self.prepare_digit_image(frame, bbox)
            pred = self.model.predict(digit_image, verbose=0)
            digit = np.argmax(pred[0])
            confidence = float(pred[0][digit])
            
            # Smooth predictions
            smooth_digit, smooth_conf = self.smooth_predictions(digit, confidence)
            
            if smooth_digit is not None and smooth_conf > self.confidence_threshold:
                # Draw on main display
                cv2.drawContours(main_display, [contour], -1, (0, 255, 0), 2)
                
                # Draw on debug display
                cv2.drawContours(debug_display, [contour], -1, (0, 255, 0), 2)
                cv2.rectangle(debug_display, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Add text to debug display
                text = f"{smooth_digit} ({smooth_conf:.2f})"
                cv2.putText(debug_display, text,
                          (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, (255, 255, 255), 2)
        
        return main_display, debug_display
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Preprocess frame
        thresh = self.preprocess_frame(frame)
        
        # Find contours
        contours_info = self.find_digit_contours(thresh)
        
        if not contours_info:
            return frame, cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        # Draw results
        return self.draw_results(frame, thresh, contours_info)
    
    def run(self):
        """Run the digit detector"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        print("Starting video capture... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                main_display, debug_display = self.process_frame(frame)
                
                # Show results
                cv2.imshow('Main View', main_display)
                cv2.imshow('Processing View', debug_display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        detector = DigitDetector()
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
