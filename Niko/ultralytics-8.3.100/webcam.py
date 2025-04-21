import cv2
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import initialize_weights
from ultralytics.utils.loss import v8DetectionLoss

# Load the student model with random weights
student_model = YOLO('yolov8n.yaml')  # Replace with your desired YOLO model
initialize_weights(student_model)

# Load the pre-trained teacher model
teacher_model = YOLO('yolov8n.pt')

# Initialize the detection loss
detection_loss = v8DetectionLoss(student_model)

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define an optimizer for the student model
optimizer = torch.optim.Adam(student_model.model.parameters(), lr=1e-4)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB (YOLO expects RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform prediction with the teacher model
    teacher_results = teacher_model.predict(source=frame_rgb, save=False, show=False)

    # Extract teacher predictions (bounding boxes, confidence scores, etc.)
    teacher_preds = teacher_results[0].boxes.xyxy  # Bounding boxes
    teacher_classes = teacher_results[0].boxes.cls  # Class predictions

    # Perform a forward pass with the student model
    student_results = student_model(frame_rgb)  # Direct forward pass

    # Handle empty predictions from the student model
    if student_results[0].boxes.xyxy.shape[0] == 0:
        # Create a dummy tensor for student predictions
        student_preds = torch.zeros_like(teacher_preds)
    else:
        student_preds = student_results[0].boxes.xyxy

    # Prepare the batch for the loss function
    batch = {
        "bboxes": teacher_preds,  # Ground truth bounding boxes from the teacher model
        "batch_idx": torch.zeros(teacher_preds.shape[0], dtype=torch.int64),  # Dummy batch indices
        "cls": teacher_classes,  # Ground truth class labels from the teacher model
    }

    # Compute the loss
    loss, _ = detection_loss(student_preds, batch)

    # Compute the teacher's full predictions (logits or probabilities)
    teacher_full_preds = teacher_results[0].full_preds  # Assuming `probs` contains the teacher's prediction probabilities

    # Ensure the teacher's predictions are not empty
    if teacher_full_preds is None or teacher_full_preds.shape[0] == 0:
        print("Warning: No predictions from the teacher model.")
        continue

    # Compute the student's full predictions
    student_full_preds = student_results[0].full_preds  # Assuming `probs` contains the student's prediction probabilities

    # Ensure the student's predictions are not empty
    if student_full_preds is None or student_full_preds.shape[0] == 0:
        print("Warning: No predictions from the student model.")
        continue

    # Compute the knowledge distillation loss (e.g., KL divergence)
    temperature = 3.0  # Temperature for softening the logits
    kd_loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(student_full_preds / temperature, dim=1),
        torch.nn.functional.softmax(teacher_full_preds / temperature, dim=1),
        reduction='batchmean'
    )

    # Combine the detection loss and the knowledge distillation loss
    total_loss = loss + kd_loss

    # Backpropagate the total loss and update the student model
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Print the losses for debugging
    print(f"Detection Loss: {loss.item()}, KD Loss: {kd_loss.item()}, Total Loss: {total_loss.item()}")

    # Visualize the teacher's predictions on the frame
    annotated_frame = teacher_results[0].plot()

    # Display the frame
    cv2.imshow("YOLO Webcam (Teacher Predictions)", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the student model weights
torch.save(student_model.model.state_dict(), "student_model_weights.pth")
print("Student model weights saved to 'student_model_weights.pth'")