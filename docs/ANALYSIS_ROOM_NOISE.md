# Báo cáo ngắn: vì sao PLY phòng bị nhiễu/lệch

Dữ liệu: ZIP `20261029_135827` (job SCAN4, phòng ~2,15 × 1,64 × 2,19 m). Phân tích chạy trên máy cục bộ, không đụng RunPod.

## 1. Hiện tượng (đo trên `project_point_cloud_clean.ply`, 580.798 điểm)
- Mỗi mặt phẳng (tường/sàn/trần) dày ~5 cm (độ lệch chuẩn); chỉ ~60% điểm nằm trong 3 cm quanh mặt phẳng.
- Nhìn từ trên xuống thấy 2 lớp tường lệch nhau ~20 cm, và nhiều điểm lơ lửng giữa phòng.
- `quality_report.json`: cảnh báo "Manhattan refinement is large" (xoay 36,6°), bộ lọc chỉ loại 7% điểm.

## 2. Pipeline dựng phòng từ "LiDAR" như thế nào
Cảm biến là VL53L8CX: **chỉ 8×8 = 64 điểm/khung, 14 khung/giây, ~63.500 điểm thật cả phiên**.
Các lớp xử lý sau đó:
1. VIO (ESKF + loop closure + pose graph) ra `processed_vio.csv`.
2. **Depth completion Depthor+ (mạng nơ-ron)**: RGB + 64 điểm ToF -> depth dày 518×518 mỗi keyframe. Đây là lớp làm "dày" điểm: ~63.500 điểm thật thành ~580.000 điểm, phần lớn do mạng suy diễn.
3. TSDF (voxel 2 cm) -> mesh, trộn với VGGT prior, rồi tách mặt phẳng + Manhattan.

## 3. Kết quả kiểm tra nguyên nhân
| Giả thuyết | Kết quả | Kết luận |
|---|---|---|
| ARCore thô bị đứt pose (37 s, 51 s, 65 s; nhảy 14/23/41 cm) | Có, nhưng VIO của pipeline đã làm mượt (bước nhảy lớn nhất 13,5 cm; mọi cổng chất lượng đạt) | Không phải nguyên nhân chính |
| Pose (ARCore hay VIO) quyết định độ dày | Dựng chỉ từ điểm ToF thật: cả hai nguồn pose cho kết quả gần như giống nhau | Pose không phải điểm nghẽn |
| **Sai hướng lưới ToF / hiệu chỉnh LiDAR–camera** | Thử các hướng lưới 8×8: tỉ lệ điểm nằm trong 3 cm quanh mặt phẳng tăng **28% -> ~52%** khi lưới ToF được xoay 270° + lật gương (cả ARCore lẫn VIO đều cho cùng đỉnh) | **Nguyên nhân chính tìm được** |
| Depth completion tạo điểm giả | 63.500 điểm thật -> 580.798 điểm trong PLY | Nguyên nhân thứ hai (đám sương giữa phòng) |

Bằng chứng nằm ở `lidar_camera_extrinsics.json`: `source: "bootstrap"`, `applied: false`, config `rot180z` là **giá trị khởi tạo, chưa hiệu chuẩn thật**. Pipeline vẫn dùng giá trị này và không báo lỗi.

## 4. Giới hạn của kết luận
- 52% vẫn thấp hơn mức mong đợi của dữ liệu sạch; còn sai số nhỏ khác (độ nghiêng, tịnh tiến, rộng chùm tia ~10 cm ở 1 m, nhiễu ToF 1–5 cm).
- Phép thử dùng cách nhìn thô (RANSAC trên mẫu), chưa chạy lại toàn pipeline với hướng lưới đã sửa.
- Một ZIP duy nhất; cần xác nhận lại trên vài ZIP khác.

## 5. Phương pháp giải quyết (theo thứ tự ưu tiên)
1. **Hiệu chuẩn thật LiDAR–camera** (xoay + tịnh tiến), thay config bootstrap; cấm pipeline chạy im lặng khi `applied: false` / `source: bootstrap` (ít nhất phải cảnh báo rõ trong kết quả).
2. **Kiểm tra tự động hướng lưới ToF**: trước khi dựng, thử các hướng (xoay/lật) và chọn hướng cho độ phẳng cao nhất; ghi kết quả vào `quality_report.json`.
3. **Siết depth completion**: lọc ô ToF theo mã trạng thái (giữ 5 và 9), bỏ điểm không có điểm đo thật gần; giảm số điểm suy diễn.
4. **Dựng từ mặt phẳng**: ép tường/sàn/trần theo mặt phẳng Manhattan rồi mới xuất mesh/PLY sạch; làm sạch mạnh hơn (hiện chỉ loại 7%).
5. Hướng dẫn người quét: đi chậm, quét rộng hơn (quỹ đạo hiện chỉ ~0,7 m).
6. (Tuỳ chọn, chỉ để xem đẹp) 2DGS/PGSR sau khi pose và hiệu chuẩn đã chuẩn. Gaussian Splatting không giải quyết sai hình học.

Việc 1–4 sửa trong code pipeline GPU (`app/pipeline/*`, `app/services/*`), thuộc người phụ trách GPU. Phần Temporal không đổi.
