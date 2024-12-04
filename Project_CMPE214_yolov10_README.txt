First step: install Python3.10 and Git

Second step: Create a new yolo10 folder

Third step: Execute the following command
    pip install  supervision labelme   labelme2yolo huggingface_hub  google_cloud_audit_log
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install  git+https://github.com/THU-MIG/yolov10.git

Fourth step: Detect the object with camera and yolo10
a.     python yolov10-infer.py
b.     python gen-imgs.py  (get the pictures by camera)
c.     roboflow for labelling
d.    yolo detect train data=yolo10-test/data.yaml model=yolov10n.pt epochs=30 batch=8 imgsz=640 device=0
e.    python yolov10-detect.py  （Testing）

Appendix: Label the pictures
a.    labelme (online)
b.    labelme2yolo --json_dir D:\..yolo10\output_images