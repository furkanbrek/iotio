import cv2
import numpy as np
import tensorflow as tf
import os

def detect_digit(frame):
    """
    Detect and isolate potential digit regions in the frame
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Create debug visualization
    debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None, None, debug, thresh
    
    # Filter and sort contours by area
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:  # Minimum area threshold
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Check if the contour has reasonable dimensions for a digit
        if 0.2 <= aspect_ratio <= 1.0:
            valid_contours.append((contour, area))
    
    if not valid_contours:
        return None, None, None, debug, thresh
    
    # Get the largest valid contour
    contour, area = max(valid_contours, key=lambda x: x[1])
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(contour)
    
    # Add padding
    padding = int(max(w, h) * 0.2)  # 20% padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(frame.shape[1] - x, w + 2*padding)
    h = min(frame.shape[0] - y, h + 2*padding)
    
    # Extract region
    digit_region = frame[y:y+h, x:x+w]
    
    # Draw contour and bounding box on debug view
    cv2.drawContours(debug, [contour], -1, (0, 255, 0), 2)
    cv2.rectangle(debug, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return (x, y, w, h), digit_region, contour, debug, thresh

def preprocess_image(image):
    """
    Preprocess image for the model
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image

def main():
    # Load model
    model_path = 'saved_model/my_model.keras'
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Starting video capture... Press 'q' to quit")
    
    # For prediction smoothing
    predictions = []
    smooth_window = 3
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        try:
            # Create a copy for main display
            main_display = frame.copy()
            
            # Detect digit
            bbox, digit_region, contour, debug, thresh = detect_digit(frame)
            
            # Show threshold window
            cv2.imshow('Threshold', thresh)
            
            # Show debug window with all visualizations
            cv2.imshow('Debug View', debug)
            
            if bbox is not None and digit_region is not None:
                # Draw only contour on main display
                cv2.drawContours(main_display, [contour], -1, (0, 255, 0), 2)
                
                # Preprocess and predict
                processed = preprocess_image(digit_region)
                pred = model.predict(processed, verbose=0)
                digit = np.argmax(pred[0])
                confidence = float(pred[0][digit])
                
                # Smooth predictions
                predictions.append((digit, confidence))
                if len(predictions) > smooth_window:
                    predictions.pop(0)
                
                if len(predictions) == smooth_window:
                    # Count occurrences and average confidence
                    pred_counts = {}
                    for d, c in predictions:
                        if d not in pred_counts:
                            pred_counts[d] = [0, 0]
                        pred_counts[d][0] += 1
                        pred_counts[d][1] += c
                    
                    # Get most common prediction
                    most_common = max(pred_counts.items(),
                                   key=lambda x: (x[1][0], x[1][1]/x[1][0]))
                    digit = most_common[0]
                    confidence = most_common[1][1] / most_common[1][0]
                    
                    # Add prediction text to debug view only
                    if confidence > 0.5:
                        text = f"{digit} ({confidence:.2f})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(debug, text, (10, 30), font, 1, (255, 255, 255), 2)
                        # Update debug window with text
                        cv2.imshow('Debug View', debug)
        
        except Exception as e:
            print(f"Error during processing: {e}")
            main_display = frame.copy()
        
        # Show main display with only contours
        cv2.imshow('Digit Detection', main_display)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()