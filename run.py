#!/usr/bin/env python3
"""
Script khởi động ứng dụng Helmet Detection
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_requirements():
    """Kiểm tra và cài đặt requirements"""
    print("🔍 Kiểm tra requirements...")
    
    try:
        import flask
        import cv2
        import ultralytics
        print("✅ Tất cả requirements đã được cài đặt")
        return True
    except ImportError as e:
        print(f"❌ Thiếu package: {e}")
        return False

def install_requirements():
    """Cài đặt requirements"""
    print("📦 Cài đặt requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Cài đặt requirements thành công")
        return True
    except subprocess.CalledProcessError:
        print("❌ Lỗi cài đặt requirements")
        return False

def setup_directories():
    """Tạo các thư mục cần thiết"""
    print("📁 Tạo cấu trúc thư mục...")
    
    directories = [
        "static/uploads",
        "violations",
        "models",
        "templates"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Tạo thư mục: {directory}")

def download_model():
    """Tải model YOLO nếu chưa có"""
    model_path = "models/yolov8n.pt"
    
    if not os.path.exists(model_path):
        print("🔄 Đang tải model YOLOv8...")
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')  # Sẽ tự động tải về
            
            # Di chuyển model vào thư mục models
            if os.path.exists('yolov8n.pt'):
                shutil.move('yolov8n.pt', model_path)
                print("✅ Model đã được tải về và cấu hình")
            
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            return False
    else:
        print("✅ Model đã tồn tại")
    
    return True

def create_template_files():
    """Tạo file template nếu chưa có"""
    template_dir = Path("templates")
    
    # Template cơ bản cho index.html
    index_template = """<!DOCTYPE html>
<html>
<head>
    <title>Helmet Detection System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="text-center mb-5">Helmet Detection System</h1>
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h5>Live Detection</h5>
                        <a href="/live" class="btn btn-primary">Start</a>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h5>Upload File</h5>
                        <input type="file" id="fileInput" class="form-control mb-2">
                        <button onclick="uploadFile()" class="btn btn-success">Upload</button>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-body text-center">
                        <h5>Results</h5>
                        <a href="/results" class="btn btn-info">View</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    templates = {
        "index.html": index_template,
        "live_detection.html": """<!DOCTYPE html>
<html>
<head><title>Live Detection</title></head>
<body>
    <div class="container">
        <h2>Live Camera Detection</h2>
        <img src="/video_feed" width="640" height="480">
        <br>
        <button onclick="startCamera()">Start</button>
        <button onclick="stopCamera()">Stop</button>
    </div>
</body>
</html>""",
        "results.html": """<!DOCTYPE html>
<html>
<head><title>Results</title></head>
<body>
    <div class="container">
        <h2>Detection Results</h2>
        <div id="results">Results will appear here</div>
    </div>
</body>
</html>"""
    }
    
    for filename, content in templates.items():
        file_path = template_dir / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Tạo template: {filename}")

def main():
    """Hàm main"""
    print("🚀 Khởi động Helmet Detection System")
    print("=" * 50)
    
    # Kiểm tra Python version
    if sys.version_info < (3, 7):
        print("❌ Cần Python 3.7 hoặc cao hơn")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Tạo thư mục
    setup_directories()
    
    # Kiểm tra requirements
    if not check_requirements():
        if not install_requirements():
            print("❌ Không thể cài đặt requirements")
            sys.exit(1)
    
    # Tải model
    if not download_model():
        print("❌ Không thể tải model")
        sys.exit(1)
    
    # Tạo template files cơ bản
    create_template_files()
    
    print("\n" + "=" * 50)
    print("✅ Setup hoàn tất!")
    print("🌐 Khởi động web server...")
    print("📱 Truy cập: http://localhost:5000")
    print("🛑 Nhấn Ctrl+C để dừng")
    print("=" * 50)
    
    # Khởi động Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Ứng dụng đã được dừng")
    except Exception as e:
        print(f"❌ Lỗi khởi động: {e}")

if __name__ == "__main__":
    main()