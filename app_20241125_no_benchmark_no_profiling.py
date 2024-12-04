from flask import Flask, render_template, Response, jsonify, request
import cv2
from ultralytics import YOLOv10
import supervision as sv
import time

app = Flask(__name__)

# 初始化 YOLO 模型和标注器
model = YOLOv10('best.pt')
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头

# 定义全集、物品价格、检测到的 inventory list 及其超时逻辑
all_items = {"Essential": 5, "GoodGather": 3}  # 定义全集及其价格
inventory_list = {}
detection_timeout = 5  # 超过5秒未检测到则认为物品已被移除

# 生成检测后的视频流
def generate_frames():
    global inventory_list
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 对每帧进行对象检测
        results = model(frame)[0]
        
        # 当前帧检测到的物品及其时间
        current_detections = {}
        for detection in results.boxes:
            class_id = int(detection.cls[0])  # 获取类别ID
            label = model.names[class_id] if class_id in model.names else 'Unknown'

            if label in all_items:  # 只处理全集中的物品
                current_detections[label] = time.time()  # 记录物品检测时间

                # 将物品添加到 inventory list
                if label not in inventory_list:
                    inventory_list[label] = {"price": all_items[label], "last_seen": time.time()}

        # 移除超时的物品
        for item in list(inventory_list.keys()):
            if item not in current_detections and time.time() - inventory_list[item]["last_seen"] > detection_timeout:
                del inventory_list[item]
            elif item in current_detections:
                inventory_list[item]["last_seen"] = current_detections[item]

        # 标注检测结果
        detections = sv.Detections.from_ultralytics(results)
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # 将标注后的帧转换为 JPEG 格式并作为视频流返回
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 获取 inventory list 和 shopping list 以及总价
@app.route('/inventory_and_shopping_list')
def get_inventory_and_shopping_list():
    # 计算 shopping list 为全集减去 inventory list
    shopping_list = {item: price for item, price in all_items.items() if item not in inventory_list}
    total_price = sum(shopping_list.values())
    return jsonify({
        "inventory_list": list(inventory_list.keys()),
        "shopping_list": list(shopping_list.keys()),
        "total_price": total_price
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
