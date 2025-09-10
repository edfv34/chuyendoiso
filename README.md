                                 🎓HỆ THỐNG PHÁT HIỆN AN TOÀN MŨ BẢO HỘ CỦA CÔNG NHÂN
![ưe](https://github.com/user-attachments/assets/bafc7daf-df47-4132-94e7-e5de88e4518a)
## 📌 Giới thiệu
Trong môi trường xây dựng, tai nạn lao động thường xảy ra do công nhân không trang bị đầy đủ phương tiện bảo hộ, đặc biệt là **mũ bảo hộ**.  
Đề tài **Helmet Detection System** nhằm mục tiêu phát hiện tự động việc công nhân có đội mũ bảo hộ hay không, từ đó giúp nâng cao an toàn lao động tại công trường.

## 🎯 Mục tiêu & Ý nghĩa
- Phát hiện trong thời gian thực việc đội mũ bảo hộ.
- Giúp ban quản lý giám sát công trường một cách tự động.
- Hạn chế tai nạn lao động, nâng cao ý thức của công nhân.
- Có thể tích hợp vào hệ thống camera giám sát hoặc cảnh báo tự động.

## 🏗️ Kiến trúc hệ thống
1. **Dataset**: Sử dụng bộ dữ liệu hình ảnh công nhân có/không đội mũ bảo hộ (nguồn từ Roboflow/Kaggle).  
2. **Mô hình**: Áp dụng mô hình **YOLO / CNN** để nhận diện.  
3. **Xử lý hình ảnh**: OpenCV dùng để đọc camera, tiền xử lý ảnh.  
4. **Kết quả**: Hiển thị khung bao quanh (Bounding Box) đối tượng và nhãn (`Helmet` / `No Helmet`).  

## 🛠️ Công nghệ sử dụng
- Python 3.10+
- OpenCV
- TensorFlow / PyTorch
- Numpy, Pandas, Matplotlib
- Flask (tùy chọn: để triển khai web demo)

## 📥 Cài đặt
### 1. Clone dự án
```bash
https://github.com/uytthh/GIAM-SAT-AN-TOAN-TAI-NHA-MAY-CAMERA-PHAT-HIEN-NHAN-VIEN-KHONG-DOI-MU-BAO-HO
cd helmet_detection_app
2. Tạo môi trường ảo (khuyến nghị)
bash
Sao chép
Chỉnh sửa
python -m venv venv
source venv/bin/activate   # trên Linux/Mac
venv\Scripts\activate      # trên Windows
3. Cài đặt thư viện
bash
Sao chép
Chỉnh sửa
pip install -r requirements.txt
🚀 Chạy chương trình
1. Chạy phát hiện trên ảnh
bash
Sao chép
Chỉnh sửa
python detect.py --image path/to/image.jpg
2. Chạy phát hiện bằng webcam
bash
Sao chép
Chỉnh sửa
python detect.py --webcam
3. Chạy demo web (nếu có Flask)
bash
Sao chép
Chỉnh sửa
python app.py
Truy cập tại: http://localhost:5000

📊 Kết quả
Phát hiện và phân loại công nhân có đội mũ (Helmet) và không đội mũ (No Helmet).

Xuất ra ảnh/video với khung bao quanh.

![h1](https://github.com/user-attachments/assets/e918ca26-5fec-4f31-92a8-72cdaaa4f780)
![h2](https://github.com/user-attachments/assets/36c75ee9-fbd2-4803-bc05-e7eea32f012f)
📈 Đánh giá chi tiết

Ưu điểm:
✅ Hệ thống phát hiện nhanh, chính xác, hoạt động được với nhiều loại dữ liệu.
✅ Có thể triển khai trong thực tế để nâng cao giám sát an toàn.
✅ Mở rộng dễ dàng, có thể thêm nhiều đối tượng PPE khác (áo phản quang, găng tay).

Nhược điểm:
❌ Cần tập dữ liệu đủ lớn để tăng độ chính xác.
❌ Yêu cầu phần cứng GPU để chạy nhanh trong môi trường thực tế lớn.
❌ Có thể gặp khó khăn trong điều kiện ánh sáng yếu hoặc camera chất lượng thấp.

🔮 Hướng phát triển

Tích hợp cảnh báo âm thanh/ký hiệu khi phát hiện người không đội mũ.

Nhận diện nhiều loại thiết bị bảo hộ khác.

Tối ưu mô hình để chạy trên thiết bị IoT (Raspberry Pi, Jetson Nano).

Phát triển dashboard web để quản lý và theo dõi nhiều camera.

👨‍💻 Tác giả
Nhóm sinh viên: 1 bao gồm
Trần Hoàng Công Tuyển
Dương Đức Cường
Nguyễn Văn Hội

