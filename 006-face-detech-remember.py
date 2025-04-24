# LAD: video 11s, YOLO cpu : 14s, YOLO gpu: 3-5s
# Face detection and recognition across multiple videos

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
import shutil
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path
import uuid

# === CẤU HÌNH ===
VIDEO_SOURCE = r"c:/Users/pc2/Downloads/video1.mp4"  # Hoặc 0 nếu dùng webcam
VIDEO_SOURCE = r"c:/Users/pc2/Downloads/video2-lanhdaoVN.mp4"  # Hoặc 0 nếu dùng webcam
ID_FILE = "id_tracker.json"
EMBEDDINGS_DIR = "face_embeddings"  # Thư mục chứa embedding vectors
FACES_DIR = "face_images"  # Thư mục chứa ảnh khuôn mặt
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# GPU optimization settings
BATCH_SIZE = 32  # Xử lý nhiều frame cùng lúc
PREFETCH_SIZE = 64  # Số frame đọc trước và xếp hàng đợi
NMS_CONF_THRESHOLD = 0.25  # Non-max suppression confidence threshold
NMS_IOU_THRESHOLD = 0.45  # Non-max suppression IoU threshold
DISPLAY_OUTPUT = True  # Hiển thị frame output
SAVE_CROPS = True  # Lưu hình ảnh khuôn mặt
FACE_RECOGNITION_THRESHOLD = 0.7  # Ngưỡng cosine similarity để nhận diện cùng một khuôn mặt
MAX_FACES_PER_PERSON = 20  # Số lượng khuôn mặt tối đa lưu cho mỗi người

# Kiểm tra và sử dụng GPU nếu có
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# === KHỞI TẠO ===
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# Load model YOLOv8 mặt với GPU nếu có sẵn
model = YOLO("yolov8n_face.pt")
if device.type == 'cuda':
    model.to(device)
    print("YOLO model moved to GPU")

# Khởi tạo các model với cùng một device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MTCNN
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

# FaceNet
facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

print(f"MTCNN device: {next(mtcnn.parameters()).device}")
print(f"FaceNet device: {next(facenet.parameters()).device}")

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

# Hàm tạo thư mục cho mỗi người
def create_person_directory(person_id):
    person_dir = os.path.join(FACES_DIR, f"person_{person_id}")
    os.makedirs(person_dir, exist_ok=True)
    return person_dir

# Hàm để trích xuất embedding từ khuôn mặt
def extract_face_embedding(face_img):
    try:
        # Tiền xử lý
        if face_img.shape[0] < 20 or face_img.shape[1] < 20:
            return None
            
        # Sử dụng MTCNN để căn chỉnh khuôn mặt
        face = mtcnn(face_img)
        if face is None:
            return None
            
        # Chuyển face tensor sang cùng device với model FaceNet
        face = face.to(device)
            
        # Tạo embedding vector với FaceNet
        with torch.no_grad():
            embedding = facenet(face.unsqueeze(0)).detach().cpu().numpy()[0]
        return embedding
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

# Load dữ liệu embeddings đã lưu
embeddings_data = {}
previous_persons_count = 0  # Số lượng người đã phát hiện trước đây

if os.path.exists(os.path.join(EMBEDDINGS_DIR, "embeddings.pkl")):
    try:
        with open(os.path.join(EMBEDDINGS_DIR, "embeddings.pkl"), "rb") as f:
            embeddings_data = pickle.load(f)
        previous_persons_count = len(embeddings_data)  # Đếm số lượng người đã phát hiện trước đây
        print(f"Loaded {previous_persons_count} person embeddings")
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        embeddings_data = {}
        previous_persons_count = 0
else:
    print("No existing embeddings found. Starting fresh.")

# Khởi tạo ID data
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

# Hàm so sánh khuôn mặt với database để tìm người giống nhất
def find_matching_person(face_embedding, threshold=FACE_RECOGNITION_THRESHOLD):
    best_match_id = None
    best_match_score = threshold
    
    for person_id, embeddings in embeddings_data.items():
        for emb in embeddings:
            # Tính toán cosine similarity giữa 2 embedding
            similarity = cosine_similarity([face_embedding], [emb])[0][0]
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = person_id
    
    return best_match_id, best_match_score

# Hàm cập nhật dữ liệu người
def update_person_data(person_id, face_img, face_embedding, frame_id):
    # Đảm bảo person_id là string
    person_id = str(person_id)
    
    # Tạo thư mục cho người này nếu chưa có
    person_dir = create_person_directory(person_id)
    
    # Lưu ảnh khuôn mặt
    face_filename = f"face_{uuid.uuid4()}_f{frame_id}.jpg"
    face_path = os.path.join(person_dir, face_filename)
    cv2.imwrite(face_path, face_img)
    
    # Cập nhật embeddings
    if person_id not in embeddings_data:
        embeddings_data[person_id] = []
    
    # Giới hạn số lượng embeddings lưu trữ cho mỗi người
    if len(embeddings_data[person_id]) < MAX_FACES_PER_PERSON:
        embeddings_data[person_id].append(face_embedding)
    
    # Lưu lại dữ liệu embeddings
    with open(os.path.join(EMBEDDINGS_DIR, "embeddings.pkl"), "wb") as f:
        pickle.dump(embeddings_data, f)
    
    return face_path

# Tạo queue để lưu các ảnh cần xử lý
image_queue = queue.Queue(maxsize=100)

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
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

frame_id = 0
total_person_count = 0
new_person_count = 0
recognized_person_count = 0

# Tạo cửa sổ với kích thước cố định
if DISPLAY_OUTPUT:
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Recognition", DISPLAY_WIDTH, DISPLAY_HEIGHT)

print("=== Starting video processing ===")
print(f"Persons detected before: {previous_persons_count}")
print("Frame\tFaces\tNew\tRecognized")
print("------------------------------")

# Đo thời gian chi tiết
yolo_time = 0
recognition_time = 0
tracking_time = 0
save_time = 0
display_time = 0
total_time = 0

# Thêm trước vòng lặp chính
start_time = time.time()

# Khởi động thread đọc frame
reader_thread = threading.Thread(target=frame_reader, daemon=True)
reader_thread.start()

# Thread pool cho bước tiền xử lý
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Để tránh hiệu ứng khởi động lạnh của GPU
if device.type == 'cuda':
    print("Warming up GPU...")
    dummy_input = torch.zeros((1, 3, 640, 640), device=device)
    for _ in range(5):
        _ = model(dummy_input, verbose=False)
    print("GPU warm-up complete")

frames_batch = []
frame_ids_batch = []

# Dictionary để theo dõi ID của tracker và person_id từ nhận dạng
tracker_to_person_id = {}

while True:
    try:
        # Đọc frame từ queue
        frame, current_frame_id = frame_queue.get(timeout=1)
        if frame is None:
            break
        frame_id = current_frame_id
            
        # Tạo batch frames
        frames_batch.append(frame)
        frame_ids_batch.append(frame_id)
        
        # Chỉ xử lý khi đã đủ batch hoặc cuối video
        if len(frames_batch) < BATCH_SIZE and not frame_queue.empty():
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
        
        # Tính thời gian YOLO
        yolo_batch_time = (yolo_end - yolo_start)
        yolo_time += yolo_batch_time
        
        # Xử lý từng frame trong batch
        for batch_idx, (frame, frame_id, result) in enumerate(zip(frames_batch, frame_ids_batch, results)):
            # Đảm bảo result có dạng đúng
            if isinstance(result, list):
                result = result[0]
                
            frame_start_time = time.time()
            
            # Phát hiện khuôn mặt
            detections = []
            face_count = 0
            
            track_start = time.time()
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box.cpu().numpy()
                # Chỉ lấy khuôn mặt có confidence tốt
                if conf >= NMS_CONF_THRESHOLD:
                    face_count += 1
                    # Sử dụng centroids cho tracking
                    centroid = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]])
                    detections.append(Detection(
                        points=centroid,
                        scores=np.array([float(conf)]),
                        data={"bbox": (int(x1), int(y1), int(x2), int(y2))}
                    ))
            
            # Cập nhật tracker
            tracked_objects = tracker.update(detections=detections)
            track_end = time.time()
            tracking_time += (track_end - track_start)
            
            # Xử lý nhận dạng khuôn mặt
            recog_start = time.time()
            frame_new_persons = 0
            frame_recognized_persons = 0
            
            for obj in tracked_objects:
                # Lấy bounding box từ data
                if not hasattr(obj, "last_detection") or obj.last_detection is None:
                    continue
                    
                x1, y1, x2, y2 = obj.last_detection.data["bbox"]
                tracker_id = obj.id
                
                # Crop khuôn mặt
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                
                # Trích xuất embedding từ khuôn mặt
                face_embedding = extract_face_embedding(face_crop)
                if face_embedding is None:
                    continue
                
                # Kiểm tra nếu tracker_id đã được gán với person_id
                if tracker_id in tracker_to_person_id:
                    person_id = tracker_to_person_id[tracker_id]
                    frame_recognized_persons += 1
                    
                    # Cập nhật dữ liệu cho person này
                    if SAVE_CROPS and frame_id % 5 == 0:  # Lưu mỗi 5 frame để giảm số lượng ảnh
                        face_path = update_person_data(person_id, face_crop, face_embedding, frame_id)
                else:
                    # Tìm người trùng khớp trong database
                    matching_id, similarity_score = find_matching_person(face_embedding)
                    
                    if matching_id is not None:
                        # Đã tìm thấy người trong database
                        person_id = matching_id
                        frame_recognized_persons += 1
                        print(f"Recognized person {person_id} with similarity {similarity_score:.3f}")
                    else:
                        # Người mới chưa có trong database
                        person_id = id_data["next_id"]
                        id_data["next_id"] += 1
                        frame_new_persons += 1
                        print(f"New person detected! Assigned ID: {person_id}")
                    
                    # Lưu mapping giữa tracker_id và person_id
                    tracker_to_person_id[tracker_id] = person_id
                    
                    # Lưu dữ liệu người mới
                    if SAVE_CROPS:
                        face_path = update_person_data(person_id, face_crop, face_embedding, frame_id)
                
                # Vẽ bounding box và ID
                if DISPLAY_OUTPUT:
                    color = (0, 255, 0) if matching_id is not None else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"ID: {person_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            recog_end = time.time()
            recognition_time += (recog_end - recog_start)
            
            # Cập nhật tổng số người
            new_person_count += frame_new_persons
            recognized_person_count += frame_recognized_persons
            
            # In thông tin ra terminal
            print(f"Frame {frame_id}: Faces: {face_count}, New: {frame_new_persons}, Recognized: {frame_recognized_persons}")
            
            # Hiển thị thông tin trên frame
            if DISPLAY_OUTPUT:
                # Thông tin về GPU memory usage
                if device.type == 'cuda':
                    gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**2
                else:
                    gpu_mem_alloc = 0
                    gpu_mem_reserved = 0
                
                # Hiển thị thông tin
                cv2.putText(frame, f"Frame: {frame_id}/{total_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, f"Faces: {face_count}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, f"Total new persons: {new_person_count}", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, f"Total recognized: {recognized_person_count}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, f"Device: {device.type}", (10, 190), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Persons detected before: {previous_persons_count}", (10, 230), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Hiển thị frame
                cv2.imshow("Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
        
        # Xóa frames đã xử lý
        frames_batch = []
        frame_ids_batch = []
        
    except queue.Empty:
        continue
    except KeyboardInterrupt:
        print("Interrupted by user")
        break
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()

# Sau khi vòng lặp kết thúc
end_time = time.time()
total_processing_time = end_time - start_time

# Lưu ID data
with open(ID_FILE, "w") as f:
    json.dump(id_data, f)

# Lưu embeddings data
with open(os.path.join(EMBEDDINGS_DIR, "embeddings.pkl"), "wb") as f:
    pickle.dump(embeddings_data, f)

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("=== Video processing completed ===")
print(f"Total frames processed: {frame_id}")
print(f"Total processing time: {timedelta(seconds=total_processing_time)}")
print(f"Persons detected before: {previous_persons_count}")
print(f"New persons detected: {new_person_count}")
print(f"Recognized persons: {recognized_person_count}")

# Đang chờ lưu hết ảnh
print("Đang chờ lưu hết ảnh...")
image_queue.put(None)  # Signal dừng thread
saver_thread.join(timeout=10)  # Chờ tối đa 10 giây
print("Đã lưu xong.")

# Hiển thị thống kê
print("\n=== Performance Statistics ===")
print(f"Total time: {total_processing_time:.2f}s")
print(f"YOLO detection: {yolo_time:.2f}s ({yolo_time/total_processing_time*100:.1f}%)")
print(f"Face recognition: {recognition_time:.2f}s ({recognition_time/total_processing_time*100:.1f}%)")
print(f"Tracking: {tracking_time:.2f}s ({tracking_time/total_processing_time*100:.1f}%)")