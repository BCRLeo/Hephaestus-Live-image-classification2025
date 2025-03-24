import torch
import cv2
import warnings

# Suppress the specific FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    # Load YOLOv5 model
    print("Loading YOLOv5 model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()
    
    # Initialize webcam
    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)  # 0 for default webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting detection... Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break
            
        # Convert frame to RGB (YOLOv5 expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = model(frame_rgb)
        
        # Process results
        for det in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            
            # Convert coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Get class name and confidence
            class_name = results.names[int(cls)]
            confidence = float(conf)
            
            # Draw bounding box
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label with class name and confidence
            label = f'{class_name} {confidence:.2f}'
            # Calculate text size
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Draw background rectangle for better text visibility
            cv2.rectangle(frame, (x1, y1 - text_height - baseline - 5), 
                         (x1 + text_width, y1), color, -1)
            
            # Draw text in white for better contrast
            cv2.putText(frame, label, (x1, y1 - baseline - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Display FPS
        cv2.imshow('YOLOv5 Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 