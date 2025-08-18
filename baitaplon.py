import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from datetime import datetime

class SafetyHelmetDetector:
    def __init__(self, model_path="yolov8n.pt"):
        """
        Khởi tạo detector với model YOLO
        Args:
            model_path: Đường dẫn đến model YOLO
        """
        self.model = YOLO(model_path)
        self.violation_count = 0
        self.total_detections = 0
        
        # Tạo thư mục lưu ảnh vi phạm
        self.violation_dir = "violations"
        if not os.path.exists(self.violation_dir):
            os.makedirs(self.violation_dir)
    
    def detect_safety_equipment(self, frame):
        """
        Phát hiện người và mũ bảo hộ trong frame
        Args:
            frame: Khung hình từ camera
        Returns:
            processed_frame: Khung hình đã xử lý với bounding boxes
            violations: Danh sách các vi phạm
        """
        results = self.model(frame)
        violations = []
        
        # Xử lý kết quả
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Lấy tọa độ và confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Lấy tên class
                    class_name = self.model.names[cls]
                    
                    # Vẽ bounding box
                    if conf > 0.5:  # Threshold confidence
                        if class_name == 'person':
                            # Kiểm tra xem có mũ bảo hộ không
                            has_helmet = self.check_helmet_in_person_area(frame, x1, y1, x2, y2, results)
                            
                            if has_helmet:
                                color = (0, 255, 0)  # Xanh lá - An toàn
                                label = f"Safe Worker: {conf:.2f}"
                            else:
                                color = (0, 0, 255)  # Đỏ - Vi phạm
                                label = f"NO HELMET: {conf:.2f}"
                                violations.append({
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': conf,
                                    'timestamp': datetime.now()
                                })
                            
                            # Vẽ bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, label, (int(x1), int(y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        elif class_name in ['helmet', 'hard-hat', 'safety-helmet']:
                            # Vẽ mũ bảo hộ
                            color = (255, 0, 0)  # Xanh dương
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"Helmet: {conf:.2f}", (int(x1), int(y1-10)), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame, violations
    
    def check_helmet_in_person_area(self, frame, px1, py1, px2, py2, results):
        """
        Kiểm tra xem có mũ bảo hộ trong vùng đầu của người không
        """
        head_area_y2 = py1 + (py2 - py1) * 0.3  # 30% phần trên của bounding box
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls]
                    
                    if class_name in ['helmet', 'hard-hat', 'safety-helmet']:
                        # Kiểm tra overlap với vùng đầu
                        helmet_center_x = (x1 + x2) / 2
                        helmet_center_y = (y1 + y2) / 2
                        
                        if (px1 <= helmet_center_x <= px2 and 
                            py1 <= helmet_center_y <= head_area_y2):
                            return True
        return False
    
    def save_violation_image(self, frame, violation_info):
        """
        Lưu ảnh vi phạm
        """
        timestamp = violation_info['timestamp'].strftime("%Y%m%d_%H%M%S")
        filename = f"violation_{timestamp}.jpg"
        filepath = os.path.join(self.violation_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def process_video_stream(self, source=0):
        """
        Xử lý video stream từ camera
        Args:
            source: Nguồn video (0 cho webcam, hoặc đường dẫn file video)
        """
        cap = cv2.VideoCapture(source)
        
        # Thiết lập độ phân giải
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        print("Bắt đầu phát hiện mũ bảo hộ...")
        print("Nhấn 'q' để thoát, 's' để chụp ảnh, 'r' để reset thống kê")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Phát hiện
            processed_frame, violations = self.detect_safety_equipment(frame)
            
            # Xử lý vi phạm
            if violations:
                self.violation_count += len(violations)
                for violation in violations:
                    self.save_violation_image(processed_frame, violation)
                    print(f"⚠️  CẢNH BÁO: Phát hiện nhân viên không đội mũ bảo hộ!")
            
            self.total_detections += 1
            
            # Tính FPS
            if frame_count % 30 == 0:
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Hiển thị thông tin trên frame
            info_text = [
                f"FPS: {fps:.1f}",
                f"Vi pham: {self.violation_count}",
                f"Tong quet: {self.total_detections}",
                f"Ti le vi pham: {(self.violation_count/max(1,self.total_detections)*100):.1f}%"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(processed_frame, text, (10, 30 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Hiển thị frame
            cv2.imshow('Safety Helmet Detection System', processed_frame)
            
            # Xử lý phím bấm
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Chụp ảnh
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"screenshot_{timestamp}.jpg", processed_frame)
                print(f"Đã lưu ảnh: screenshot_{timestamp}.jpg")
            elif key == ord('r'):
                # Reset thống kê
                self.violation_count = 0
                self.total_detections = 0
                print("Đã reset thống kê!")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Báo cáo cuối cùng
        print(f"\n=== BÁO CÁO KẾT THÚC ===")
        print(f"Tổng số lần quét: {self.total_detections}")
        print(f"Số vi phạm phát hiện: {self.violation_count}")
        print(f"Tỷ lệ vi phạm: {(self.violation_count/max(1,self.total_detections)*100):.2f}%")
        print(f"Ảnh vi phạm lưu tại: {self.violation_dir}/")

    def process_video_file(self, video_path, output_path=None):
        """
        Xử lý file video
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            # Thiết lập video writer
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            processed_frame, violations = self.detect_safety_equipment(frame)
            
            if violations:
                self.violation_count += len(violations)
                print(f"Frame {frame_num}: Phát hiện {len(violations)} vi phạm")
            
            if output_path:
                out.write(processed_frame)
            
            # Hiển thị tiến độ
            if frame_num % 30 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Đã xử lý: {progress:.1f}% ({frame_num}/{total_frames})")
        
        cap.release()
        if output_path:
            out.release()
            print(f"Video đã được lưu: {output_path}")

def main():
    """
    Hàm main để chạy ứng dụng
    """
    # Khởi tạo detector
    detector = SafetyHelmetDetector()
    
    print("=== HỆ THỐNG PHÁT HIỆN MŨ BẢO HỘ ===")
    print("1. Camera trực tiếp")
    print("2. Xử lý file video") 
    print("3. Thoát")
    
    choice = input("Chọn chức năng (1-3): ")
    
    if choice == '1':
        # Sử dụng camera
        camera_id = input("Nhập ID camera (0 cho webcam mặc định): ")
        try:
            camera_id = int(camera_id) if camera_id else 0
        except:
            camera_id = 0
        
        detector.process_video_stream(camera_id)
    
    elif choice == '2':
        # Xử lý file video
        video_path = input("Nhập đường dẫn file video: ")
        if os.path.exists(video_path):
            save_output = input("Lưu video kết quả? (y/n): ").lower() == 'y'
            output_path = None
            if save_output:
                output_path = f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            detector.process_video_file(video_path, output_path)
        else:
            print("File không tồn tại!")
    
    elif choice == '3':
        print("Thoát chương trình.")
    
    else:
        print("Lựa chọn không hợp lệ!")

if __name__ == "__main__":
    main()