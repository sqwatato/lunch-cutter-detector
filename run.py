import cv2
from ultralytics import YOLO
from boxmot import DeepOCSORT, StrongSORT, BoTSORT # Different tracking algorithms
from pathlib import Path
import numpy as np
import colors
import random

# Gets the center of a bounding box
def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

# Load the YOLOv8 model
model = YOLO('best.pt')

# Initialize the tracker
tracker = BoTSORT(
  model_weights=Path('mobilenetv2_x1_4_dukemtmcreid.pt'),  # which ReID model to use, when applicable
  device='cpu',  # 'cpu', 'cuda:0', 'cuda:1', ... 'cuda:N'
  fp16=False,  # wether to run the ReID model with half precision or not
#   det_thresh=0.2  # minimum valid detection confidence
)

# Open the video file
cap = cv2.VideoCapture("vid2.mp4")

# Dictionary to store the color of each object ID, previous location, and whether it has crossed the line or not
id_map = {}

# Create a VideoWriter object to save the output video
writer = None

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # Break the loop if the frame was not successfully read
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, device="cpu")
        
        # Extract the bounding boxes, IDs, confidences and classes from the YOLOv8 results
        annotated_frame = results[0]
        
        # Run the tracker on the frame
        tracker_outputs = tracker.update(annotated_frame.boxes.data.detach().cpu(), frame)
        
        # Extract the bounding boxes, IDs, confidences and classes from the tracker outputs
        bboxes = tracker_outputs[:, :4]
        id = tracker_outputs[:, 4]
        confidences = tracker_outputs[:, 5]
        classes = tracker_outputs[:, 6]
        
        # Loop through the bounding boxes, IDs, confidences and classes
        for bbox, id, conf, cls in zip(bboxes, id, confidences, classes):
            # Ignore if class is head
            if cls == 0:
                continue
            
            
            
            # Draw the bounding box and ID on the frame
            bbox = bbox.astype(np.int32)
            x1, y1, x2, y2 = bbox
            
            # Set ID to a color
            if id not in id_map:
                id_map[id] = (random.choice(colors.COLORS), get_center(bbox), False)
            
            if id_map[id][2] == False and ((get_center(bbox)[1] < frame.shape[0] * 3 / 5 and id_map[id][1][1] > frame.shape[0] * 3 / 5) or (get_center(bbox)[1] > frame.shape[0] * 3 / 5 and id_map[id][1][1] < frame.shape[0] * 3 / 5)):
                id_map[id] = ((0,0,0), id_map[id][1], True)
                print(f'ID {int(id)} has crossed the line')
                
            color = id_map[id][0]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'id: {int(id)}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, 30)
        
        # Draw Line
        cv2.line(frame, (0, int(frame.shape[0] * 3 / 5)), (frame.shape[1], int(frame.shape[0] * 3 / 5)), (0, 255, 0), 2)
        
        # Display the number of people who have crossed the line
        cv2.putText(frame, f'Crossed: {sum([1 for id in id_map if id_map[id][2] == True])}', (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 30)

        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference",frame)
        
        # Write the frame to the output video
        if writer == None:
            writer = cv2.VideoWriter('output2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
writer.release()
