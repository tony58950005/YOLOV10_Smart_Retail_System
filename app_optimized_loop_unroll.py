from flask import Flask, render_template, Response, jsonify
import cv2
import time
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from ultralytics import YOLOv10
import supervision as sv

# 初始化 Flask 应用
app = Flask(__name__)

# 初始化 YOLO 模型和注释器
model = YOLOv10('best.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 使用默认摄像头
cap = cv2.VideoCapture(0)

# CUDA profiling 相关变量
cuda_start_event = torch.cuda.Event(enable_timing=True)
cuda_end_event = torch.cuda.Event(enable_timing=True)

# FPS 计算变量
frame_count = 0
start_time = time.time()

# 使用 GPU loop unrolling 优化的处理逻辑
def process_frame_gpu(frame_tensor):
    """
    使用 GPU loop unrolling 加速处理张量计算。
    """
    # 假设 frame_tensor 为 4D 的 (batch_size, channels, height, width)
    frame_tensor = frame_tensor.to('cuda', non_blocking=True)
    
    # 模拟简单的 loop unrolling，应用 YOLO 模型
    batch_size = frame_tensor.size(0)
    results = []
    for i in range(0, batch_size, 4):  # 每次处理 4 个 frame
        batch = frame_tensor[i:i+4]
        if batch.size(0) > 0:
            results.extend(model(batch))
    
    return results

# 实时生成视频流
def generate_frames():
    global frame_count, start_time
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧转换为张量
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # 开始 CUDA profiling
        cuda_start_event.record()

        # 使用 GPU loop unrolling 处理帧
        results = process_frame_gpu(frame_tensor)

        # 停止 CUDA profiling
        cuda_end_event.record()
        torch.cuda.synchronize()

        # 计算推理时间
        inference_time = cuda_start_event.elapsed_time(cuda_end_event)

        # FPS 计算
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        else:
            fps = 0

        # 将检测结果注释到帧上
        detections = sv.Detections.from_ultralytics(results[0])  # 假设 batch size 为 1
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # 转换为 JPEG 并返回流
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Flask 路由
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cuda_performance')
def get_cuda_performance():
    # 返回 profiling 信息
    inference_time = cuda_start_event.elapsed_time(cuda_end_event)
    fps = frame_count / (time.time() - start_time) if start_time != 0 else 0
    allocated_memory = torch.cuda.memory_allocated()
    reserved_memory = torch.cuda.memory_reserved()
    
    return jsonify({
        "inference_time": inference_time,
        "fps": fps,
        "allocated_memory": allocated_memory,
        "reserved_memory": reserved_memory
    })

@app.route('/')
def index():
    return render_template('index.html')

# 启动应用
if __name__ == "__main__":
    app.run(debug=True)
