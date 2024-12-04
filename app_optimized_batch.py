from flask import Flask, render_template, Response, jsonify
import cv2
import time
import torch
from ultralytics import YOLOv10
import supervision as sv
import queue

# Flask应用初始化
app = Flask(__name__)

# 初始化YOLO模型和注释器
model = YOLOv10('best.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# 视频捕获
cap = cv2.VideoCapture(0)  # 使用默认摄像头

# 批处理设置
batch_size = 4  # 每批处理的帧数
frame_queue = queue.Queue(maxsize=batch_size)  # 用队列管理帧
detection_timeout = 5  # 超时时间

# CUDA变量
cuda_start_event = torch.cuda.Event(enable_timing=True)
cuda_end_event = torch.cuda.Event(enable_timing=True)

# FPS计算变量
frame_count = 0
start_time = time.time()

# 存储当前的检测和库存
all_items = {"Essential": 5, "GoodGather": 3}
inventory_list = {}

def generate_frames():
    global inventory_list, frame_count, start_time
    batch_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将帧加入队列
        if frame_queue.qsize() < batch_size:
            frame_queue.put(frame)
        else:
            # 从队列中提取一批帧
            for _ in range(batch_size):
                batch_frames.append(frame_queue.get())

            # 开始CUDA计时
            cuda_start_event.record()

            # 批量推理
            results = model(batch_frames)

            # 停止CUDA计时
            cuda_end_event.record()
            torch.cuda.synchronize()  # 等待事件记录完成

            # 计算推理时间
            inference_time = cuda_start_event.elapsed_time(cuda_end_event)

            # FPS计算
            frame_count += batch_size
            elapsed_time = time.time() - start_time
            if elapsed_time > 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            else:
                fps = 0

            # 处理每帧的检测结果
            annotated_batch = []
            for i, frame_results in enumerate(results):
                # 获取单帧结果
                detections = sv.Detections.from_ultralytics(frame_results)
                
                # 注释框和标签
                annotated_frame = bounding_box_annotator.annotate(scene=batch_frames[i], detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
                annotated_batch.append(annotated_frame)

                # 更新库存逻辑
                current_detections = {}
                for detection in frame_results.boxes:
                    class_id = int(detection.cls[0])
                    label = model.names[class_id] if class_id in model.names else 'Unknown'

                    if label in all_items:
                        current_detections[label] = time.time()
                        if label not in inventory_list:
                            inventory_list[label] = {"price": all_items[label], "last_seen": time.time()}

                # 移除超时的库存项
                for item in list(inventory_list.keys()):
                    if item not in current_detections and time.time() - inventory_list[item]["last_seen"] > detection_timeout:
                        del inventory_list[item]
                    elif item in current_detections:
                        inventory_list[item]["last_seen"] = current_detections[item]

            # 返回注释后的帧
            for annotated_frame in annotated_batch:
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            batch_frames = []

# 其他API端点
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

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
