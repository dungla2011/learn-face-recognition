# LAD: video 11s, YOLO cpu : 14s, YOLO gpu: 5s

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
from norfair import Detection, Tracker
import threading
import queue
import time
import torch
from datetime import timedelta
import concurrent.futures

# === CẤU HÌNH ===
VIDEO_SOURCE = r"c:/Users/pc2/Downloads/video1.mp4"  # Hoặc 0 nếu dùng webcam
ID_FILE = "id_tracker.json"
SAVE_DIR_NEW = "saved_images/new"
SAVE_DIR_KNOWN = "saved_images/known"
DISPLAY_WIDTH = 960  # Half HD width
DISPLAY_HEIGHT = 540  # Half HD height

# GPU optimization settings
BATCH_SIZE = 16  # Xử lý nhiều frame cùng lúc
PREFETCH_SIZE = 32  # Số frame đọc trước và xếp hàng đợi
NMS_CONF_THRESHOLD = 0.25  # Non-max suppression confidence threshold
NMS_IOU_THRESHOLD = 0.45  # Non-max suppression IoU threshold
DISPLAY_OUTPUT = True  # Hiển thị frame output (tắt để tăng tốc)
SAVE_CROPS = False  # Lưu hình ảnh người (tắt để tăng tốc)

# Kiểm tra và sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

input("=== Press Enter to start video processing ===")

# === KHỞI TẠO ===
os.makedirs(SAVE_DIR_NEW, exist_ok=True)
os.makedirs(SAVE_DIR_KNOWN, exist_ok=True)

# Load model YOLOv8 với GPU nếu có sẵn
model = YOLO("yolov8n.pt")
if device.type == 'cuda':
    model.to(device)
    print("Model moved to GPU")

# Thiết lập YOLOv8 để sử dụng batch processing
model.conf = NMS_CONF_THRESHOLD
model.iou = NMS_IOU_THRESHOLD
model.max_det = 50  # Maximum number of detections per image

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

# Tạo queue để lưu các ảnh cần xử lý
image_queue = queue.Queue(maxsize=100)  # Giới hạn queue size để tránh dùng quá nhiều RAM

# Queue cho prefetching frame
frame_queue = queue.Queue(maxsize=PREFETCH_SIZE)

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

# Hàm đọc frame từ video và đưa vào queue
def frame_reader():
    frame_id = 0
    skip_frames = 0  # Skip frames nếu cần tăng tốc
    
    # Tối ưu đọc frame
    while True:
        ret, frame = cap.read()
        if not ret:
            # Đánh dấu kết thúc
            for _ in range(PREFETCH_SIZE):
                frame_queue.put((None, frame_id))
            break
        
        frame_id += 1
        
        # Có thể bỏ qua frames để tăng tốc (nếu cần)
        if skip_frames > 0 and frame_id % skip_frames != 0:
            continue
            
        # Có thể resize frame để giảm kích thước tính toán
        # frame = cv2.resize(frame, (640, 480))
            
        frame_queue.put((frame, frame_id))
        
        # Hiển thị tiến trình
        if frame_id % 100 == 0:
            print(f"Reading frame: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%)")
        
        # Tạm dừng nếu queue đầy
        if frame_queue.qsize() >= PREFETCH_SIZE:
            time.sleep(0.001)

# Khởi động thread xử lý lưu ảnh
saver_thread = threading.Thread(target=image_saver_thread, daemon=True)
saver_thread.start()

# Mở video
cap = cv2.VideoCapture(VIDEO_SOURCE)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total_frames}, Video FPS: {fps}")

# Tối ưu buffer sizes cho video capture
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Tăng buffer size để đọc video nhanh hơn

frame_id = 0
total_person_count = 0  # Tổng số người đã phát hiện được
total_images_saved = 0

# Tạo cửa sổ với kích thước cố định (chỉ khi cần hiển thị)
if DISPLAY_OUTPUT:
    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("=== Starting video processing ===")
print("Frame\tPersons\tNew\tTotal")
print("------------------------------")

# Đo thời gian chi tiết
yolo_time = 0
tracker_time = 0
save_time = 0
display_time = 0  # Thêm biến đo thời gian hiển thị
io_time = 0  # Thêm biến đo thời gian I/O
total_time = 0
processing_count = 0  # Đếm số frame đã xử lý
processing_times = []  # Lưu thời gian xử lý từng frame

# Thêm trước vòng lặp chính
start_time = time.time()

# Khởi động thread đọc frame
reader_thread = threading.Thread(target=frame_reader, daemon=True)
reader_thread.start()

# Thread pool cho bước tiền xử lý
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Để tránh hiệu ứng khởi động lạnh của GPU
print("Warming up GPU...")
dummy_input = torch.zeros((1, 3, 640, 640), device=device)
for _ in range(10):
    _ = model(dummy_input, verbose=False)
print("GPU warm-up complete")

# Biến lưu thời gian idle
idle_time = 0

frames_batch = []
frame_ids_batch = []

while True:
    loop_start = time.time()
    
    # Đọc frame từ queue
    try:
        frame, current_frame_id = frame_queue.get(timeout=1)
        if frame is None:
            break
        frame_id = current_frame_id
    except queue.Empty:
        continue
        
    # Tạo batch frames
    frames_batch.append(frame)
    frame_ids_batch.append(frame_id)
    
    # Chỉ xử lý khi đã đủ batch hoặc cuối video
    if len(frames_batch) < BATCH_SIZE and not frame_queue.empty():
        # Lưu thời gian idle khi chờ đủ batch
        idle_now = time.time()
        # Tiếp tục thu thập frames cho batch đủ lớn
        continue
    
    # Xử lý batch
    batch_start = time.time()
    
    # Đo YOLO
    yolo_start = time.time()
    # Sử dụng batch processing cho model nếu có nhiều frame
    if len(frames_batch) > 1:
        results = model(frames_batch, verbose=False)
    else:
        results = [model(frames_batch[0], verbose=False)[0]]
    yolo_end = time.time()
    
    # Tính thời gian YOLO trên batch
    yolo_batch_time = (yolo_end - yolo_start)
    # Tính thời gian YOLO trên mỗi frame
    yolo_per_frame = yolo_batch_time / len(frames_batch)
    yolo_time += yolo_batch_time
    
    # Xử lý từng frame trong batch
    for batch_idx, (frame, frame_id, result) in enumerate(zip(frames_batch, frame_ids_batch, results)):
        # Đảm bảo result có dạng đúng
        if isinstance(result, list):
            result = result[0]
            
        frame_start_time = time.time()
        
        # Nhận dạng người
        detections = []
        person_count = 0  # Số người trong frame hiện tại

        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.cpu().numpy()  # Chuyển sang CPU trước khi đổi thành numpy
            if int(cls) == 0 and conf >= NMS_CONF_THRESHOLD:  # Chỉ lấy 'person' có confidence tốt
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
        tracker_frame_time = (tracker_end - tracker_start)
        tracker_time += tracker_frame_time

        # Hiển thị số lượng người trong frame
        cv2.putText(frame, f"Current count: {person_count}", (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # Hiển thị số frame
        cv2.putText(frame, f"Frame: {frame_id}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        # Kiểm tra xem có ID mới không để cập nhật tổng số người
        new_persons_in_frame = 0
        
        # Chỉ lưu ảnh nếu được yêu cầu
        if SAVE_CROPS:
            save_start = time.time()
            
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
                
                
                image_queue.put((path, crop.copy()))
                total_images_saved += 1
                
                # Vẽ khung & ID nếu hiển thị
                if DISPLAY_OUTPUT:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {id_data[id_key]}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            
            save_end = time.time()
            save_frame_time = (save_end - save_start)
            save_time += save_frame_time
        else:
            # Định nghĩa các biến với giá trị mặc định khi SAVE_CROPS=False
            save_frame_time = 0
            save_start = time.time()  # Thêm dòng này để tránh lỗi
            save_end = save_start     # Thêm dòng này để tránh lỗi
            
            # Vẽ khung luôn nếu không lưu ảnh nhưng cần hiển thị
            if DISPLAY_OUTPUT:
                for obj in tracked_objects:
                    if not hasattr(obj, "last_detection") or obj.last_detection is None:
                        continue
                        
                    x1, y1, x2, y2 = obj.last_detection.data["bbox"]
                    person_id = obj.id
                    id_key = f"id_{person_id}"
                    
                    if id_key not in id_data:
                        if "next_id" not in id_data:
                            id_data["next_id"] = 1
                            
                        id_data[id_key] = id_data["next_id"]
                        id_data["next_id"] += 1
                        new_persons_in_frame += 1
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {id_data[id_key]}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Cập nhật tổng số người đã phát hiện
        total_person_count += new_persons_in_frame
        save_end = time.time()
        save_frame_time = (save_end - save_start)
        save_time += save_frame_time
        
        # In thông tin ra terminal
        print(f" --- Frame count: {frame_id}\t Found Person: {person_count}\t new: {new_persons_in_frame}\t count per: {total_person_count}")
        
        # Hiển thị tổng số người đã phát hiện
        # cv2.putText(frame, f"Total unique persons: {total_person_count}", (10, 130), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        
        frame_end_time = time.time()
        frame_duration = frame_end_time - frame_start_time
        processing_count += 1
        
        # Thời gian tổng cộng cho tất cả các frame
        total_time += frame_duration
        
        # Add the current frame duration to processing_times list
        processing_times.append(frame_duration)
        
        # Thời gian hiển thị
        if DISPLAY_OUTPUT:
            display_start = time.time()
            
            # Thông tin về GPU memory usage
            if device.type == 'cuda':
                gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
                gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
            else:
                gpu_mem_alloc = 0
                gpu_mem_reserved = 0
                
            # Phần hiển thị thông tin và visualization
            # Hiển thị thông tin thời gian lên frame
            cv2.putText(frame, f"YOLO: {yolo_per_frame*1000:.1f}ms/frame", (10, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Tracker: {tracker_frame_time*1000:.1f}ms", (10, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 210), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Device: {device.type}", (10, 230), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"GPU Mem: {gpu_mem_alloc:.1f}/{gpu_mem_reserved:.1f} MB", (10, 250), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Batch: {len(frames_batch)}", (10, 270), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Progress: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%)", (10, 290), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Tracking", frame)
            key = cv2.waitKey(1) & 0xFF
            display_end = time.time()
            display_time += (display_end - display_start)
            
            if key == ord("q"):
                break

    # Xóa frames đã xử lý
    frames_batch = []
    frame_ids_batch = []

# Sau khi vòng lặp kết thúc
end_time = time.time()
total_processing_time = end_time - start_time
video_length = frame_id / cap.get(cv2.CAP_PROP_FPS)  # Thời lượng video theo giây

# Đảm bảo total_time không vượt quá total_processing_time
# Sử dụng phương pháp phân chia thời gian hợp lý hơn
measured_time = yolo_time + tracker_time + save_time + display_time

# Tính thời gian IO là phần còn lại, đảm bảo nó không âm
io_time = max(0, total_processing_time - measured_time)

# Other time là phần cần tính lại từ total_time
other_time = 0  # Không tính được chính xác other_time trong trường hợp này

print("=== Video processing completed ===")
print(f"Total frames processed: {frame_id}")
print(f"Total processing time: {timedelta(seconds=total_processing_time)}")
print(f"Video duration: {timedelta(seconds=video_length)}")
print(f"Processing ratio: {total_processing_time/video_length:.2f}x real-time")
print(f"Average processing time per frame: {total_processing_time/frame_id:.4f}s")
print(f"Min processing time: {min(processing_times) if processing_times else 0:.4f}s")
print(f"Max processing time: {max(processing_times) if processing_times else 0:.4f}s")

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
print(f"Total time: {total_processing_time:.2f}s")
print(f"YOLO: {yolo_time:.2f}s ({yolo_time/total_processing_time*100:.1f}%)")
print(f"Tracker: {tracker_time:.2f}s ({tracker_time/total_processing_time*100:.1f}%)")
print(f"Save Images: {save_time:.2f}s ({save_time/total_processing_time*100:.1f}%)")
print(f"Display: {display_time:.2f}s ({display_time/total_processing_time*100:.1f}%)")
print(f"Main Components: {measured_time:.2f}s ({measured_time/total_processing_time*100:.1f}%)")
print(f"I/O & Overhead: {io_time:.2f}s ({io_time/total_processing_time*100:.1f}%)")
print(f"Average time per frame: {total_processing_time/frame_id*1000:.1f}ms")

# Hiển thị thông tin chi tiết hơn về sự phân chia thời gian
print("\n=== Detailed Timing ===")
print(f"YOLO/frame: {yolo_time/frame_id*1000:.2f}ms")
print(f"Tracker/frame: {tracker_time/frame_id*1000:.2f}ms")
print(f"Save/frame: {save_time/frame_id*1000:.2f}ms")
print(f"Display/frame: {display_time/frame_id*1000:.2f}ms")

# Hiển thị thông tin GPU
if device.type == 'cuda':
    print("\n=== GPU Statistics ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved(0) / 1024**2:.2f} MB")

# Tối ưu hóa model cho inference
if device.type == 'cuda':
    # Tối ưu hóa CUDA cho YOLOv8
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Thiết lập memory allocation
    torch.cuda.empty_cache()
    
    # Thiết lập model cho tối ưu inference
    model.fuse()  # Fuse các layers cho inference nhanh hơn 