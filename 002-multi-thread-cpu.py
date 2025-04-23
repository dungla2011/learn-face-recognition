import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
import threading
import queue
import time
from datetime import timedelta

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

# Tạo queue để lưu các ảnh cần xử lý
image_queue = queue.Queue(maxsize=100)  # Giới hạn queue size để tránh dùng quá nhiều RAM

# Hàm xử lý lưu ảnh chạy trong thread riêng
def image_saver_thread():
    while True:
        try:
            # Lấy dữ liệu từ queue (path và ảnh cần lưu)
            item = image_queue.get(timeout=5)
            if item is None:  # Signal để dừng thread
                break
                
            path, img = item
            cv2.imwrite(path, img)
            image_queue.task_done()
        except queue.Empty:
            # Tiếp tục nếu queue rỗng (không cần dừng thread)
            continue
        except Exception as e:
            print(f"Lỗi khi lưu ảnh: {e}")
            image_queue.task_done()

# Khởi động thread xử lý lưu ảnh (ngay sau khi khởi tạo các thư mục)
saver_thread = threading.Thread(target=image_saver_thread, daemon=True)
saver_thread.start()

# Đo thời gian chi tiết - thêm vào đầu chương trình
yolo_time = 0
tracker_time = 0
save_time = 0
other_time = 0
total_time = 0

# Thêm trước vòng lặp chính
start_time = time.time()
processing_times = []  # Lưu thời gian xử lý từng frame

while True:
    loop_start = time.time()
    
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    
    # Đo YOLO
    yolo_start = time.time()
    results = model(frame)[0]
    yolo_end = time.time()
    yolo_time += (yolo_end - yolo_start)

    # Nhận dạng người
    detections = []
    person_count = 0  # Số người trong frame hiện tại

    for result in results.boxes.data:
        x1, y1, x2, y2, conf, cls = result.cpu()  # Add .cpu() to move tensor to CPU
        if int(cls) == 0:  # Chỉ lấy 'person'
            person_count += 1  # Đếm số người trong frame
            # Sử dụng centroids thay vì corners
            centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
            detections.append(Detection(
                points=centroid,
                scores=np.array([float(conf)]),
                data={"bbox": (int(x1), int(y1), int(x2), int(y2))}
            ))

    # Đo Tracker
    tracker_start = time.time()
    tracked_objects = tracker.update(detections=detections)
    tracker_end = time.time()
    tracker_time += (tracker_end - tracker_start)

    # Hiển thị số lượng người trong frame
    cv2.putText(frame, f"Current count: {person_count}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    # Hiển thị số frame
    cv2.putText(frame, f"Frame: {frame_id}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
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
        image_queue.put((path, crop.copy()))  # .copy() để tránh tham chiếu thay đổi
        total_images_saved += 1

        # Vẽ khung & ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {id_data[id_key]}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # Cập nhật tổng số người đã phát hiện
    total_person_count += new_persons_in_frame
    
    # In thông tin ra terminal
    print(f" --- Frame count: {frame_id}\t Found Person: {person_count}\t new: {new_persons_in_frame}\t count per: {total_person_count}")
    
    # Hiển thị tổng số người đã phát hiện
    cv2.putText(frame, f"Total unique persons: {total_person_count}", (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    # Đo Save Images
    save_start = time.time()
    # Code xử lý và lưu ảnh
    save_end = time.time()
    save_time += (save_end - save_start)
    
    loop_end = time.time()
    frame_duration = loop_end - loop_start
    total_time += frame_duration
    
    # Add the current frame duration to processing_times list
    processing_times.append(frame_duration)
    
    # Tính thời gian khác (vẽ frame, hiển thị...)
    other_duration = frame_duration - (yolo_end - yolo_start) - (tracker_end - tracker_start) - (save_end - save_start)
    other_time += other_duration
    
    # Tính và hiển thị thời gian xử lý trên frame
    avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)  # Trung bình 30 frame
    fps = 1.0 / avg_time if avg_time > 0 else 0
    
    cv2.putText(frame, f"Process time: {frame_duration:.4f}s", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Average FPS: {fps:.1f}", (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Hiển thị thông tin thời gian lên frame
    cv2.putText(frame, f"YOLO: {(yolo_end - yolo_start)*1000:.1f}ms", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Tracker: {(tracker_end - tracker_start)*1000:.1f}ms", (10, 180), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"Save: {(save_end - save_start)*1000:.1f}ms", (10, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Hiển thị frame với kích thước đã điều chỉnh
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Sau khi vòng lặp kết thúc
end_time = time.time()
total_time = end_time - start_time
video_length = frame_id / cap.get(cv2.CAP_PROP_FPS)  # Thời lượng video theo giây

print("=== Video processing completed ===")
print(f"Total frames processed: {frame_id}")
print(f"Total processing time: {timedelta(seconds=total_time)}")
print(f"Video duration: {timedelta(seconds=video_length)}")
print(f"Processing ratio: {total_time/video_length:.2f}x real-time")
print(f"Average processing time per frame: {sum(processing_times)/len(processing_times):.4f}s")
print(f"Min processing time: {min(processing_times):.4f}s")
print(f"Max processing time: {max(processing_times):.4f}s")

# Lưu lại ID
with open(ID_FILE, "w") as f:
    json.dump(id_data, f)

cap.release()
cv2.destroyAllWindows()

# Cuối chương trình - thêm đoạn này trước khi đóng cửa sổ
print("Đang chờ lưu hết ảnh...")
image_queue.put(None)  # Signal dừng thread
saver_thread.join(timeout=10)  # Chờ tối đa 10 giây
print(f"Đã lưu xong. Số ảnh còn trong queue: {image_queue.qsize()}")

# Sau khi kết thúc vòng lặp, hiển thị thống kê
print("=== Performance Statistics ===")
print(f"Total time: {total_time:.2f}s")
print(f"YOLO: {yolo_time:.2f}s ({yolo_time/total_time*100:.1f}%)")
print(f"Tracker: {tracker_time:.2f}s ({tracker_time/total_time*100:.1f}%)")
print(f"Save Images: {save_time:.2f}s ({save_time/total_time*100:.1f}%)")
print(f"Other: {other_time:.2f}s ({other_time/total_time*100:.1f}%)")
print(f"Average time per frame: {total_time/frame_id*1000:.1f}ms")
