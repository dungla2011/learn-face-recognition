import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker

# === CẤU HÌNH ===
VIDEO_SOURCE = r"c:/Users/pc2/Downloads/video1.mp4"  # Hoặc 0 nếu dùng webcam
ID_FILE = "id_tracker.json"
SAVE_DIR_NEW = "saved_images/new"
SAVE_DIR_KNOWN = "saved_images/known"
DISPLAY_WIDTH = 960  # Half HD width
DISPLAY_HEIGHT = 540  # Half HD height

# === KHỞI TẠO ===
os.makedirs(SAVE_DIR_NEW, exist_ok=True)
os.makedirs(SAVE_DIR_KNOWN, exist_ok=True)

# Load model YOLOv8 (dạng nhỏ nhất)
model = YOLO("yolov8n.pt")  # Đã huấn luyện sẵn, nhận dạng 'person'

# Khởi tạo tracker Norfair
tracker = Tracker(
    distance_function="euclidean", 
    distance_threshold=50,
    initialization_delay=3
)

# Load hoặc tạo dữ liệu ID
if os.path.exists(ID_FILE):
    try:
        with open(ID_FILE) as f:
            id_data = json.load(f)
        # Make sure the next_id key exists
        if "next_id" not in id_data:
            print("Warning: 'next_id' key not found in ID file. Initializing to 1.")
            id_data["next_id"] = 1
    except (json.JSONDecodeError, FileNotFoundError):
        print("Error loading ID file. Creating new one.")
        id_data = {"next_id": 1}
else:
    id_data = {"next_id": 1}

# Mở video
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_id = 0
total_person_count = 0  # Tổng số người đã phát hiện được
total_images_saved = 0

# Tạo cửa sổ với kích thước cố định
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("=== Starting video processing ===")
print("Frame\tPersons\tNew\tTotal")
print("------------------------------")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1

    # Nhận dạng người
    results = model(frame)[0]
    detections = []
    person_count = 0  # Số người trong frame hiện tại

    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 0:  # Chỉ lấy 'person'
            person_count += 1  # Đếm số người trong frame
            # Sử dụng centroids thay vì corners
            centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            detections.append(Detection(
                points=centroid,
                scores=np.array([float(conf)]),
                data={"bbox": (int(x1), int(y1), int(x2), int(y2))}
            ))

    tracked_objects = tracker.update(detections=detections)

    # Hiển thị số lượng người trong frame
    cv2.putText(frame, f"Current count: {person_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Hiển thị số frame
    cv2.putText(frame, f"Frame: {frame_id}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Kiểm tra xem có ID mới không để cập nhật tổng số người
    new_persons_in_frame = 0
    
    for obj in tracked_objects:
        # Lấy bounding box từ data
        if not hasattr(obj, "last_detection") or obj.last_detection is None:
            continue
            
        x1, y1, x2, y2 = obj.last_detection.data["bbox"]
        person_id = obj.id
        id_key = f"id_{person_id}"

        # Lưu ảnh tùy theo ID mới/cũ
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:  # Kiểm tra crop rỗng
            continue
            
        if id_key not in id_data:
            # Ensure next_id exists
            if "next_id" not in id_data:
                id_data["next_id"] = 1
                
            id_data[id_key] = id_data["next_id"]
            id_data["next_id"] += 1
            new_persons_in_frame += 1  # Tăng số người mới trong frame
            path = os.path.join(SAVE_DIR_NEW, f"{id_key}_f{frame_id}.jpg")
        else:
            path = os.path.join(SAVE_DIR_KNOWN, f"{id_key}_f{frame_id}.jpg")
        cv2.imwrite(path, crop)
        total_images_saved += 1

        # Vẽ khung & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id_data[id_key]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Cập nhật tổng số người đã phát hiện
    total_person_count += new_persons_in_frame
    
    # In thông tin ra terminal
    print(f" --- Frame count: {frame_id}\t Found Person: {person_count}\t new: {new_persons_in_frame}\t count per: {total_person_count}")
    
    # Hiển thị tổng số người đã phát hiện
    cv2.putText(frame, f"Total unique persons: {total_person_count}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Hiển thị frame với kích thước đã điều chỉnh
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("=== Video processing completed ===")
print(f"Total frames processed: {frame_id}")
print(f"Total unique persons detected: {total_person_count}")
print(f"Total images saved: {total_images_saved}")

# Lưu lại ID
with open(ID_FILE, "w") as f:
    json.dump(id_data, f)

cap.release()
cv2.destroyAllWindows()
