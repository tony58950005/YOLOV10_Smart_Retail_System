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

# FPS Calculation Variables
frame_count = 0
start_time = time.time()

# Generate video frames with profiling
def generate_frames():
    global inventory_list, frame_count, start_time
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

        # FPS Calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        else:
            fps = 0

        # Memory Stats (Detailed Memory Profiling)
        memory_allocated = torch.cuda.memory_allocated()
        memory_reserved = torch.cuda.memory_reserved()
        memory_cached = torch.cuda.memory_cached()
        memory_stats = torch.cuda.memory_stats()

        # Extract shared, pinned memory, UVM
        shared_memory = memory_stats.get('cuda.allocations.shared', 0)
        pinned_memory = memory_stats.get('cuda.allocations.pinned', 0)
        uvm_memory = memory_stats.get('cuda.allocations.uvm', 0)

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
    fps = frame_count / (time.time() - start_time) if start_time != 0 else 0
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    cached_memory = torch.cuda.memory_cached()
    
    # Detailed memory usage
    memory_stats = torch.cuda.memory_stats()
    shared_memory = memory_stats.get('cuda.allocations.shared', 0)
    pinned_memory = memory_stats.get('cuda.allocations.pinned', 0)
    uvm_memory = memory_stats.get('cuda.allocations.uvm', 0)
    
    return jsonify({
        "inference_time": inference_time,
        "fps": fps,
        "allocated_memory": allocated_memory,
        "reserved_memory": reserved_memory,
        "cached_memory": cached_memory,
        "shared_memory": shared_memory,
        "pinned_memory": pinned_memory,
        "uvm_memory": uvm_memory
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
