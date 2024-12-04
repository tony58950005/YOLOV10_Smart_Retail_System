from flask import Flask, render_template, Response, jsonify
import cv2
import time
import torch
from ultralytics import YOLOv10
import supervision as sv

app = Flask(__name__)

# Initialize YOLO model and annotators
model = YOLOv10('best.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
cap = cv2.VideoCapture(0)  # Use default camera

# Define all items, prices, and inventory
all_items = {"Essential": 5, "GoodGather": 3}
inventory_list = {}
detection_timeout = 5  # 5 seconds timeout

# CUDA profiling variables
cuda_start_event = torch.cuda.Event(enable_timing=True)
cuda_end_event = torch.cuda.Event(enable_timing=True)

# Generate video frames with profiling
def generate_frames():
    global inventory_list
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start CUDA profiling
        cuda_start_event.record()

        # Perform object detection with YOLO
        results = model(frame)[0]

        # Stop CUDA profiling
        cuda_end_event.record()
        torch.cuda.synchronize()  # Wait for the event to be recorded

        # Calculate elapsed time for GPU inference
        inference_time = cuda_start_event.elapsed_time(cuda_end_event)
        print(f"CUDA Inference Time: {inference_time} ms")

        # Process current detections
        current_detections = {}
        for detection in results.boxes:
            class_id = int(detection.cls[0])  # Get class ID
            label = model.names[class_id] if class_id in model.names else 'Unknown'

            if label in all_items:
                current_detections[label] = time.time()  # Record detection time
                if label not in inventory_list:
                    inventory_list[label] = {"price": all_items[label], "last_seen": time.time()}

        # Remove timed-out items from inventory
        for item in list(inventory_list.keys()):
            if item not in current_detections and time.time() - inventory_list[item]["last_seen"] > detection_timeout:
                del inventory_list[item]
            elif item in current_detections:
                inventory_list[item]["last_seen"] = current_detections[item]

        # Annotate frame with bounding boxes and labels
        detections = sv.Detections.from_ultralytics(results)
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Convert annotated frame to JPEG and return as video stream
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask routes and other application logic...
@app.route('/inventory_and_shopping_list')
def get_inventory_and_shopping_list():
    # Calculate shopping list as the difference between all items and inventory list
    shopping_list = {item: price for item, price in all_items.items() if item not in inventory_list}
    total_price = sum(shopping_list.values())
    return jsonify({
        "inventory_list": list(inventory_list.keys()),
        "shopping_list": list(shopping_list.keys()),
        "total_price": total_price
    })

@app.route('/cuda_performance')
def get_cuda_performance():
    # Return profiling information
    inference_time = cuda_start_event.elapsed_time(cuda_end_event)
    allocated_memory = torch.cuda.memory_allocated()
    return jsonify({
        "inference_time": inference_time,
        "allocated_memory": allocated_memory
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
