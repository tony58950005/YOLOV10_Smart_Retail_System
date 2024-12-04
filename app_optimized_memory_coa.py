from flask import Flask, render_template, Response, jsonify
import cv2
import time
import torch
from ultralytics import YOLOv10
import supervision as sv
import numpy as np

app = Flask(__name__)

# Initialize YOLO model and annotators
model = YOLOv10('best.pt')  # Remove unsupported 'half' argument
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

# Optimized batch frame generator
def generate_frames():
    global inventory_list, frame_count, start_time
    batch_frames = []  # For batch processing
    batch_size = 4  # Number of frames per batch

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add frame to batch
        batch_frames.append(frame)

        # Process batch if batch_size is reached
        if len(batch_frames) == batch_size:
            # Prepare batch for GPU memory coalescing
            batch_array = np.stack(batch_frames)  # Shape: (batch_size, H, W, C)
            batch_tensor = torch.from_numpy(batch_array).cuda().permute(0, 3, 1, 2).float() / 255.0
            batch_tensor = batch_tensor.half()  # Convert to FP16

            # Start CUDA profiling
            cuda_start_event.record()

            # Perform object detection with YOLO
            results = model(batch_tensor)

            # Stop CUDA profiling
            cuda_end_event.record()
            torch.cuda.synchronize()

            # Calculate elapsed time for GPU inference
            inference_time = cuda_start_event.elapsed_time(cuda_end_event)

            # FPS Calculation
            frame_count += batch_size
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 1.0 else 0

            # Process detections for each frame in batch
            for frame, result in zip(batch_frames, results):
                detections = sv.Detections.from_ultralytics(result)
                annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

                # Convert annotated frame to JPEG and yield
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # Clear batch_frames for next batch
            batch_frames.clear()

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
    
    return jsonify({
        "inference_time_ms": inference_time,
        "fps": fps,
        "allocated_memory_mb": allocated_memory / 1024**2,
        "reserved_memory_mb": reserved_memory / 1024**2
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
