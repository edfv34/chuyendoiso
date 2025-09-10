from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2
import os
import json
from datetime import datetime
import base64
import numpy as np
from ultralytics import YOLO
import threading
import time
from werkzeug.utils import secure_filename
import platform
import random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['VIOLATION_FOLDER'] = 'violations'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Tạo thư mục cần thiết
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIOLATION_FOLDER'], exist_ok=True)

# NEW: detection thresholds (tune để tăng độ chính xác)
CONF_THRESHOLD = 0.4      # confidence threshold for any detected box
HEAD_REGION_RATIO = 0.35  # top 35% of person bbox considered "head"
IOU_HEAD_THRESHOLD = 0.03 # IoU threshold between hat bbox and head region to count as hat

# Heuristic thresholds (tuned to reduce false positives)
SKIN_RATIO_THRESHOLD = 0.15   # nếu tỉ lệ da < 15% => khả năng được che bởi mũ
MEAN_SAT_THRESHOLD = 110.0    # nếu độ bão hòa trung bình > 110 => khả năng mũ màu
MEAN_V_LOW_HAIR = 60.0        # nếu value thấp và có nhiều da => có thể là tóc tối (NO HAT)
DEBUG_HAT = True              # bật debug để trả skin_ratio/mean_sat/mean_v trong detections

class SafetyDetectionApp:
    def __init__(self):
        try:
            self.model = YOLO('yolov8n.pt')  # Model sẽ tự động tải về
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            self.model = None
            
        self.camera = None
        self.is_detecting = False
        self.violation_count = 0
        self.total_detections = 0
        self.detection_results = []
        self.camera_thread = None
        self.frame_buffer = None
        # NEW: lock để an toàn thread
        self.lock = threading.Lock()
        # last detection summary for frontend (timestamped)
        self.last_detection = None
        # Tự động tìm camera khả dụng
        self.working_camera_id = self.find_working_camera()

    def find_working_camera(self):
        """Tự động tìm camera khả dụng (loại bỏ -1, probe an toàn)"""
        print("🔍 Đang tìm camera khả dụng...")
        # bỏ -1 (gây lỗi trên nhiều hệ thống)
        camera_ids = [0, 1, 2, 3]
        # build backends an toàn
        backends = []
        if platform.system() == "Windows":
            if hasattr(cv2, 'CAP_DSHOW'): backends.append(cv2.CAP_DSHOW)
            if hasattr(cv2, 'CAP_MSMF'): backends.append(cv2.CAP_MSMF)
        elif platform.system() == "Linux":
            if hasattr(cv2, 'CAP_V4L2'): backends.append(cv2.CAP_V4L2)
        elif platform.system() == "Darwin":
            if hasattr(cv2, 'CAP_AVFOUNDATION'): backends.append(cv2.CAP_AVFOUNDATION)
        if hasattr(cv2, 'CAP_ANY'):
            backends.append(cv2.CAP_ANY)
        if not backends:
            backends = [cv2.CAP_ANY]

        for camera_id in camera_ids:
            for backend in backends:
                cap = None
                try:
                    print(f"  Thử camera {camera_id} với backend {backend}")
                    # nếu backend == CAP_ANY, gọi không truyền backend
                    if backend == cv2.CAP_ANY:
                        cap = cv2.VideoCapture(camera_id)
                    else:
                        cap = cv2.VideoCapture(camera_id, backend)

                    if not (cap and cap.isOpened()):
                        continue

                    # thử đọc vài frame ngắn
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # thử grab/read thêm 1 lần
                        cap.grab()
                        ret, frame = cap.read()

                    if ret and frame is not None:
                        print(f"✅ Tìm thấy camera hoạt động: ID {camera_id}")
                        cap.release()
                        return camera_id
                except Exception as e:
                    print(f"  Lỗi thử camera {camera_id}: {e}")
                finally:
                    try:
                        if cap is not None:
                            cap.release()
                    except:
                        pass

        print("❌ Không tìm thấy camera nào hoạt động")
        return None
    
    def open_camera(self):
        """Mở camera với error handling tốt hơn"""
        if self.working_camera_id is None:
            return False, "Không tìm thấy camera khả dụng"
        
        try:
            # Thử mở với backend phù hợp
            if platform.system() == "Windows":
                self.camera = cv2.VideoCapture(self.working_camera_id, cv2.CAP_DSHOW)
            else:
                self.camera = cv2.VideoCapture(self.working_camera_id)
            
            if not self.camera.isOpened():
                return False, "Không thể mở camera"
            
            # Thiết lập thuộc tính camera
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test đọc frame
            ret, frame = self.camera.read()
            if not ret:
                self.camera.release()
                return False, "Camera không trả về frame"
            
            print("✅ Camera đã được mở thành công")
            return True, "Camera đã sẵn sàng"
            
        except Exception as e:
            print(f"❌ Lỗi mở camera: {e}")
            if self.camera:
                self.camera.release()
                self.camera = None
            return False, str(e)
    
    # INSERT: hàm tính IoU (dùng cho head vs hat overlap)
    def _iou(self, a, b):
        xA = max(a[0], b[0])
        yA = max(a[1], b[1])
        xB = min(a[2], b[2])
        yB = min(a[3], b[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
        boxBArea = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
        denom = float(boxAArea + boxBArea - interArea)
        if denom <= 0:
            return 0.0
        return interArea / denom

    # NEW: compute head region (top portion of person bbox)
    def _head_region(self, person_bbox, ratio=HEAD_REGION_RATIO):
        x1, y1, x2, y2 = person_bbox
        h = y2 - y1
        head_y2 = int(y1 + max(8, h * ratio))  # at least small height
        return [x1, y1, x2, head_y2]

    # REPLACE detect_in_frame: deterministic "hat" check (bất kể loại mũ)
    def detect_in_frame(self, frame):
        """Detect persons and any hat-like objects. Trả về (frame_with_drawings, detections)
        detections: list of {bbox, confidence, hat: True/False, timestamp, debug?}
        """
        if self.model is None:
            return self.fake_detection(frame)

        try:
            results = self.model(frame)
            persons = []
            hats = []

            # Mở rộng từ khóa để bao phủ nhiều kiểu "mũ"
            hat_keywords = ('helmet', 'hardhat', 'hat', 'cap', 'beanie', 'hood', 'turban', 'scarf', 'bonnet', 'visor')

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # lấy dữ liệu box
                        xy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3])
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = str(self.model.names.get(cls, cls)).lower()

                        if conf < CONF_THRESHOLD:
                            continue

                        if 'person' in class_name or class_name == 'person':
                            persons.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
                        elif any(k in class_name for k in hat_keywords):
                            hats.append({'bbox': [x1, y1, x2, y2], 'confidence': conf})
                        else:
                            # ignore other classes
                            continue

            # Xác định hat per person bằng head region + IoU / center check
            detections = []
            for p in persons:
                has_hat = False
                head = self._head_region(p['bbox'])
                debug_info = {}

                # check model hat boxes first (require IoU / center)
                for h in hats:
                    iou = self._iou(head, h['bbox'])
                    hx1, hy1, hx2, hy2 = h['bbox']
                    hcx = (hx1 + hx2) / 2.0
                    hcy = (hy1 + hy2) / 2.0
                    hx1_h, hy1_h, hx2_h, hy2_h = head
                    center_inside = (hcx >= hx1_h and hcx <= hx2_h and hcy >= hy1_h and hcy <= hy2_h)
                    if iou >= IOU_HEAD_THRESHOLD or center_inside:
                        has_hat = True
                        debug_info['model_hat_iou'] = float(iou)
                        break

                # Fallback heuristic only if model didn't detect hat
                if not has_hat:
                    try:
                        x1h, y1h, x2h, y2h = head
                        h_img_h, h_img_w = frame.shape[:2]
                        x1h = max(0, min(h_img_w-1, int(x1h)))
                        x2h = max(0, min(h_img_w-1, int(x2h)))
                        y1h = max(0, min(h_img_h-1, int(y1h)))
                        y2h = max(0, min(h_img_h-1, int(y2h)))
                        if x2h - x1h > 8 and y2h - y1h > 8:
                            head_roi = frame[y1h:y2h, x1h:x2h]
                            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)

                            # skin detection rough ranges in HSV (tunable)
                            lower_skin = np.array([0, 10, 60], dtype=np.uint8)
                            upper_skin = np.array([50, 160, 255], dtype=np.uint8)
                            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                            skin_pixels = int(np.count_nonzero(skin_mask))
                            total_pixels = head_roi.shape[0] * head_roi.shape[1]
                            skin_ratio = skin_pixels / max(1, total_pixels)

                            # mean saturation and value in head region
                            mean_sat = float(np.mean(hsv[:,:,1]))
                            mean_v = float(np.mean(hsv[:,:,2]))

                            debug_info.update({
                                'skin_ratio': float(skin_ratio),
                                'mean_sat': float(mean_sat),
                                'mean_v': float(mean_v),
                                'head_bbox': head
                            })

                            # Decision (stricter):
                            # - nếu rất ít da (rất được che) -> coi là mũ
                            # - hoặc nếu da thấp AND saturation rất cao -> mũ màu
                            # - nếu vùng tối (value thấp) và da tươm nhiều -> có thể là tóc tối -> NO HAT
                            if skin_ratio < SKIN_RATIO_THRESHOLD and mean_sat > MEAN_SAT_THRESHOLD:
                                has_hat = True
                            elif skin_ratio < (SKIN_RATIO_THRESHOLD * 0.5):
                                # very low skin coverage -> strong hat signal
                                has_hat = True
                            elif mean_v < MEAN_V_LOW_HAIR and skin_ratio > 0.20:
                                # likely dark hair visible -> not a hat
                                has_hat = False
                            # otherwise keep has_hat False
                    except Exception:
                        pass

                det = {
                    'bbox': p['bbox'],
                    'confidence': p.get('confidence', 0.0),
                    'hat': has_hat,
                    'timestamp': datetime.now().isoformat()
                }
                if DEBUG_HAT:
                    det['debug'] = debug_info
                detections.append(det)

            # Vẽ kết quả lên frame
            return self.draw_detections(frame, detections, hats), detections

        except Exception as e:
            print(f"❌ Lỗi detection: {e}")
            return self.fake_detection(frame)

    def fake_detection(self, frame):
        """Fake detection để test giao diện (không báo vi phạm)."""
        cv2.putText(frame, "DEMO MODE - No Model Loaded",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(frame, f"Violations: {self.violation_count}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, f"Total Detections: {self.total_detections}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # return empty detections to avoid false positives
        return frame, []

    # UPDATE draw_detections signature and drawing logic: show HAT (xanh) / NO HAT (đỏ)
    def draw_detections(self, frame, detections, hats):
        """Vẽ detection results: detections là danh sách persons với 'hat' flag"""
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            if d.get('hat'):
                color = (0, 200, 0)  # green
                label = f"HAT {d['confidence']:.2f}"
            else:
                color = (0, 0, 255)  # red
                label = f"NO HAT {d['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # filled label background for readability
            ((w, h), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Vẽ các bbox của "hats" nhẹ hơn để minh họa
        for h in hats:
            hx1, hy1, hx2, hy2 = h['bbox']
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 215, 0), 1)

        # Thống kê
        stats_text = [
            f"Violations: {self.violation_count}",
            f"Total: {self.total_detections}",
            f"Rate: {(self.violation_count/max(1,self.total_detections)*100):.1f}%"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (10, frame.shape[0] - 80 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

# Global detector
detector = SafetyDetectionApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live')
def live_detection():
    return render_template('live_detection.html')

@app.route('/test_camera')
def test_camera():
    """Endpoint test camera"""
    result = detector.find_working_camera()
    if result is not None:
        return jsonify({
            'success': True,
            'camera_id': result,
            'message': f'Camera ID {result} hoạt động bình thường'
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Không tìm thấy camera nào hoạt động'
        })

@app.route('/start_camera')
def start_camera():
    """Bắt đầu camera với error handling tốt hơn"""
    global detector

    if detector.is_detecting:
        return jsonify({
            'success': False, 
            'message': 'Camera đang chạy rồi'
        })

    # Kiểm tra và mở camera
    success, message = detector.open_camera()

    if not success:
        return jsonify({
            'success': False,
            'message': f'Lỗi mở camera: {message}',
            'suggestions': [
                'Kiểm tra camera có được kết nối',
                'Đóng các ứng dụng khác đang dùng camera',
                'Khởi động lại trình duyệt',
                'Kiểm tra quyền truy cập camera'
            ]
        })

    detector.is_detecting = True

    return jsonify({
        'success': True,
        'message': 'Camera đã được khởi động',
        'camera_id': detector.working_camera_id
    })

@app.route('/stop_camera')
def stop_camera():
    """Dừng camera"""
    global detector

    detector.is_detecting = False

    if detector.camera:
        detector.camera.release()
        detector.camera = None

    return jsonify({
        'success': True,
        'message': 'Camera đã được dừng'
    })

def generate_frames():
    """Generate video frames với error handling"""
    global detector

    while detector.is_detecting:
        try:
            if detector.camera is None or not detector.camera.isOpened():
                success, message = detector.open_camera()
                if not success:
                    time.sleep(1)
                    continue

            success, frame = detector.camera.read()
            if not success:
                print("❌ Không đọc được frame từ camera")
                time.sleep(0.1)
                continue

            # Process frame -> returns detections list per-person with 'hat' flag
            processed_frame, detections = detector.detect_in_frame(frame)

            # Count violations = persons without hat
            violations = [d for d in detections if not d.get('hat', False)]

            # build last_detection summary (immediate reporting)
            if detections:
                compliant = (len(violations) == 0)
                message = "Đã đạt yêu cầu" if compliant else "Yêu cầu đội mũ"
            else:
                compliant = True
                message = "Không có người trong khung"

            detector.last_detection = {
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'violations': len(violations),
                'compliant': compliant,
                'message': message
            }

            if violations:
                detector.violation_count += len(violations)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                violation_filename = f"live_violation_{timestamp}.jpg"
                violation_path = os.path.join(app.config['VIOLATION_FOLDER'], violation_filename)
                cv2.imwrite(violation_path, processed_frame)
                print(f"⚠️  Vi phạm phát hiện! Đã lưu: {violation_filename}")
                detector.detection_results.append({
                    'timestamp': timestamp,
                    'type': 'live',
                    'violations': len(violations),
                    'detections': violations,
                    'image': violation_filename
                })
                detector.detection_results = detector.detection_results[-50:]

            detector.total_detections += 1

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print(f"❌ Lỗi trong generate_frames: {e}")
            time.sleep(0.1)
            continue

@app.route('/video_feed')
def video_feed():
    """Video stream endpoint"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    """Lấy thống kê"""
    return jsonify({
        'violation_count': detector.violation_count,
        'total_detections': detector.total_detections,
        'violation_rate': (detector.violation_count/max(1,detector.total_detections)*100),
        'is_detecting': detector.is_detecting,
        'camera_available': detector.working_camera_id is not None
    })

@app.route('/reset_stats')
def reset_stats():
    """Reset thống kê"""
    detector.violation_count = 0
    detector.total_detections = 0
    detector.detection_results = []
    return jsonify({'success': True, 'message': 'Stats reset successfully'})

@app.route('/camera_info')
def camera_info():
    """Thông tin camera"""
    info = {
        'available_camera': detector.working_camera_id,
        'system': platform.system(),
        'opencv_version': cv2.__version__,
        'is_detecting': detector.is_detecting
    }

    if detector.camera and detector.camera.isOpened():
        info.update({
            'width': int(detector.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(detector.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': detector.camera.get(cv2.CAP_PROP_FPS)
        })

    return jsonify(info)

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    """Nhận ảnh chụp màn hình từ client, phân tích và trả về kết quả"""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'success': False, 'error': 'No image data'}), 400

    # Giải mã base64 từ dataURL
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'success': False, 'error': 'Cannot decode image'}), 400

    # Phân tích frame
    processed_frame, detections = detector.detect_in_frame(frame)
    detector.total_detections += 1

    # Count violations (NO HAT)
    violations = [d for d in detections if not d.get('hat', False)]
    detector.violation_count += len(violations)

    # last_detection summary
    if detections:
        compliant = (len(violations) == 0)
        message = "Đã đạt yêu cầu" if compliant else "Yêu cầu đội mũ"
    else:
        compliant = True
        message = "Không có người trong khung"

    detector.last_detection = {
        'timestamp': datetime.now().isoformat(),
        'detections': detections,
        'violations': len(violations),
        'compliant': compliant,
        'message': message
    }

    # Lưu ảnh kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_filename = f"capture_{timestamp}.jpg"
    result_path = os.path.join(app.config['VIOLATION_FOLDER'], result_filename)
    cv2.imwrite(result_path, processed_frame)

    # Lưu kết quả vào lịch sử
    detector.detection_results.append({
        'timestamp': timestamp,
        'type': 'capture',
        'violations': len(violations),
        'detections': detections,
        'confidence': max([d['confidence'] for d in detections]) if detections else 0,
        'image': result_filename
    })
    detector.detection_results = detector.detection_results[-50:]

    # Encode kết quả để trả về cho client
    _, buffer = cv2.imencode('.jpg', processed_frame)
    result_image_b64 = base64.b64encode(buffer).decode('utf-8')
    result_image_url = f"data:image/jpeg;base64,{result_image_b64}"

    return jsonify({
        'success': True,
        'violations_detected': len(violations),
        'detections': detections,
        'violation_count': detector.violation_count,
        'total_detections': detector.total_detections,
        'violation_rate': (detector.violation_count / max(1, detector.total_detections) * 100),
        'result_image': result_image_url,
        'result_filename': result_filename
    })

# --- NEW: serve violation images from violations folder ---
@app.route('/violations/<path:filename>')
def serve_violation_image(filename):
    """Serve saved violation images"""
    return send_from_directory(app.config['VIOLATION_FOLDER'], filename)

# REPLACE /upload endpoint: synchronous processing for image or video (return result immediately)
@app.route('/upload', methods=['POST'])
def upload_file():
    """Nhận file upload (ảnh hoặc video), phân tích ngay và trả kết quả JSON tức thì."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Không tìm thấy file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tên file rỗng'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Lỗi lưu file: {e}'}), 500

    # Đọc file: thử như ảnh, nếu không phải ảnh -> thử mở như video và lấy frame đầu
    frame = None
    try:
        arr = np.fromfile(save_path, dtype=np.uint8)
        if arr.size:
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        frame = None

    if frame is None:
        # thử đọc video (lấy frame đầu)
        try:
            cap = cv2.VideoCapture(save_path)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
            else:
                frame = None
        except Exception:
            frame = None

    if frame is None:
        return jsonify({'success': False, 'error': 'Không thể đọc file (không phải ảnh hợp lệ hoặc video không đọc được).'}), 400

    # Phân tích frame (một lần quét duy nhất)
    processed_frame, detections = detector.detect_in_frame(frame)

    with detector.lock:
        detector.total_detections += 1
        violations = [d for d in detections if not d.get('hat', False)]
        detector.violation_count += len(violations)

        # Lưu lịch sử
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"upload_{timestamp}_{filename}"
        # lưu ảnh result
        try:
            cv2.imwrite(os.path.join(app.config['VIOLATION_FOLDER'], result_filename), processed_frame)
        except Exception:
            _, buf = cv2.imencode('.jpg', processed_frame)
            with open(os.path.join(app.config['VIOLATION_FOLDER'], result_filename), 'wb') as f:
                f.write(buf.tobytes())

        # last_detection summary for upload
        if detections:
            compliant = (len(violations) == 0)
            message = "Đã đạt yêu cầu" if compliant else "Yêu cầu đội mũ"
        else:
            compliant = True
            message = "Không có người trong ảnh"

        detector.last_detection = {
            'timestamp': datetime.now().isoformat(),
            'detections': detections,
            'violations': len(violations),
            'compliant': compliant,
            'message': message
        }

        detector.detection_results.append({
            'timestamp': timestamp,
            'type': 'upload',
            'violations': len(violations),
            'detections': detections,
            'confidence': max([d['confidence'] for d in detections]) if detections else 0,
            'image': result_filename
        })
        detector.detection_results = detector.detection_results[-50:]

    # Trả kết quả ngay (response includes 'detections' with hat flag)
    _, buffer = cv2.imencode('.jpg', processed_frame)
    result_image_b64 = base64.b64encode(buffer).decode('utf-8')
    result_image_url = f"data:image/jpeg;base64,{result_image_b64}"

    return jsonify({
        'success': True,
        'violations_detected': len(violations),
        'detections': detections,
        'violation_count': detector.violation_count,
        'total_detections': detector.total_detections,
        'violation_rate': (detector.violation_count / max(1, detector.total_detections) * 100),
        'result_image': result_image_url,
        'result_filename': result_filename
    })

@app.route('/recent_results')
def recent_results():
    """Trả về danh sách kết quả phát hiện gần đây cho mục thống kê"""
    return jsonify({
        'results': detector.detection_results[-10:]  # Trả về 10 kết quả mới nhất
    })

# --- NEW: results page route ---
@app.route('/results')
def results_page():
    """Hiển thị trang kết quả (sử dụng cùng template index.html)"""
    # Prepare violations list for template
    violations = detector.detection_results[:]  # copy
    # Prepare list of saved violation images
    try:
        images = sorted(os.listdir(app.config['VIOLATION_FOLDER']))
        # filter common image extensions
        images = [f for f in images if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    except Exception:
        images = []

    return render_template('index.html',
                           violations=violations,
                           violation_images=images)

@app.route('/latest_detection')
def latest_detection():
    """Return last_detection summary for live UI to poll"""
    if detector.last_detection is None:
        return jsonify({'success': True, 'latest': None})
    return jsonify({'success': True, 'latest': detector.last_detection})

if __name__ == '__main__':
    print("🚀 Khởi động Helmet Detection System")
    print(f"📹 Camera khả dụng: {detector.working_camera_id}")
    print("🌐 Truy cập: http://localhost:5000")
    print("🔧 Test camera: http://localhost:5000/test_camera")

    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)