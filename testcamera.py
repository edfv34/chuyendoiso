import cv2
import platform
import numpy as np

def test_camera():
    """Kiểm tra camera có hoạt động không"""
    print("🔍 KIỂM TRA CAMERA SYSTEM")
    print("=" * 40)
    # Hiển thị thông tin hệ thống
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OpenCV version: {cv2.__version__}")
    
    # Danh sách các camera ID để thử
    camera_ids = [0, 1, 2, -1]  # -1 cho một số hệ thống
    
    working_cameras = []
    
    for camera_id in camera_ids:
        print(f"\n🎥 Thử camera ID: {camera_id}")
        
        try:
            # Tạo VideoCapture object
            cap = cv2.VideoCapture(camera_id)
            
            # Thử các backend khác nhau
            backends = [
                (cv2.CAP_DSHOW, "DirectShow (Windows)"),
                (cv2.CAP_V4L2, "Video4Linux2 (Linux)"),
                (cv2.CAP_AVFOUNDATION, "AVFoundation (Mac)"),
                (cv2.CAP_ANY, "Auto detect")
            ]
            
            for backend, name in backends:
                print(f"  Thử backend: {name}")
                cap = cv2.VideoCapture(camera_id, backend)
                
                if cap.isOpened():
                    # Thử đọc frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        height, width = frame.shape[:2]
                        print(f"  ✅ THÀNH CÔNG! Độ phân giải: {width}x{height}")
                        
                        working_cameras.append({
                            'id': camera_id,
                            'backend': backend,
                            'backend_name': name,
                            'resolution': (width, height)
                        })
                        
                        # Hiển thị frame trong 3 giây
                        print(f"  📺 Hiển thị camera trong 3 giây...")
                        for i in range(90):  # 30 FPS x 3 seconds
                            ret, frame = cap.read()
                            if ret:
                                cv2.putText(frame, f"Camera {camera_id} - {name}", 
                                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(frame, "Press 'q' to skip", 
                                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                                cv2.imshow(f'Camera Test - ID {camera_id}', frame)
                                
                                if cv2.waitKey(33) & 0xFF == ord('q'):
                                    break
                        
                        cv2.destroyAllWindows()
                        break
                    else:
                        print(f"  ❌ Không đọc được frame")
                else:
                    print(f"  ❌ Không mở được camera")
                
                cap.release()
        
        except Exception as e:
            print(f"  ❌ Lỗi: {e}")
    
    print("\n" + "=" * 40)
    print("📊 KẾT QUA KIỂM TRA")
    print("=" * 40)
    
    if working_cameras:
        print("✅ Tìm thấy camera hoạt động:")
        for cam in working_cameras:
            print(f"  - Camera ID: {cam['id']}")
            print(f"    Backend: {cam['backend_name']}")
            print(f"    Độ phân giải: {cam['resolution'][0]}x{cam['resolution'][1]}")
        
        return working_cameras
    else:
        print("❌ KHÔNG TÌM THẤY CAMERA NÀO HOẠT động!")
        print("\n🔧 GỢI Ý KHẮC PHỤC:")
        print("1. Kiểm tra camera có được cắm đúng không")
        print("2. Kiểm tra driver camera")
        print("3. Đóng các ứng dụng khác đang dùng camera (Skype, Teams, Zoom)")
        print("4. Thử cắm lại camera")
        print("5. Khởi động lại máy tính")
        
        return []

def test_camera_permissions():
    """Kiểm tra quyền truy cập camera"""
    print("\n🔐 KIỂM TRA QUYỀN TRUY CẬP CAMERA")
    print("=" * 40)
    
    system = platform.system()
    
    if system == "Windows":
        print("Windows - Kiểm tra Camera Privacy Settings:")
        print("1. Settings > Privacy & Security > Camera")
        print("2. Cho phép ứng dụng desktop truy cập camera")
        print("3. Cho phép Python.exe truy cập camera")
    
    elif system == "Darwin":  # macOS
        print("macOS - Kiểm tra quyền camera:")
        print("1. System Preferences > Security & Privacy > Camera")
        print("2. Cho phép Terminal hoặc Python truy cập camera")
    
    elif system == "Linux":
        print("Linux - Kiểm tra quyền camera:")
        print("1. Kiểm tra user có trong group video: groups $USER")
        print("2. Thêm user vào group video: sudo usermod -a -G video $USER")
        print("3. Kiểm tra camera device: ls -la /dev/video*")

def advanced_camera_test():
    """Test camera nâng cao với các tùy chọn"""
    print("\n🔬 KIỂM TRA CAMERA NÂNG CAO")
    print("=" * 40)
    
    # Thử với DirectShow trên Windows
    if platform.system() == "Windows":
        print("Thử với DirectShow backend...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không mở được camera với backend mặc định")
        return False
    
    # Thiết lập các thuộc tính camera
    properties_to_test = [
        (cv2.CAP_PROP_FRAME_WIDTH, "Width"),
        (cv2.CAP_PROP_FRAME_HEIGHT, "Height"),
        (cv2.CAP_PROP_FPS, "FPS"),
        (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),
        (cv2.CAP_PROP_CONTRAST, "Contrast"),
        (cv2.CAP_PROP_SATURATION, "Saturation"),
        (cv2.CAP_PROP_HUE, "Hue"),
    ]
    
    print("📋 Thuộc tính camera hiện tại:")
    for prop, name in properties_to_test:
        value = cap.get(prop)
        print(f"  {name}: {value}")
    
    # Thử thiết lập độ phân giải
    resolutions = [(640, 480), (1280, 720), (1920, 1080)]
    
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Yêu cầu: {width}x{height} -> Thực tế: {actual_width}x{actual_height}")
    
    cap.release()
    return True

if __name__ == "__main__":
    try:
        # Test cơ bản
        working_cameras = test_camera()
        
        # Test quyền truy cập
        test_camera_permissions()
        
        # Test nâng cao nếu có camera
        if working_cameras:
            advanced_camera_test()
        
        print("\n" + "=" * 40)
        print("✅ HOÀN TẤT KIỂM TRA CAMERA")
        
        if working_cameras:
            print(f"Sử dụng Camera ID {working_cameras[0]['id']} cho ứng dụng")
        
    except KeyboardInterrupt:
        print("\n👋 Đã dừng kiểm tra")
    except Exception as e:
        print(f"\n❌ Lỗi không mong đợi: {e}")