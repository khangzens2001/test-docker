# Báo cáo tóm tắt: PLY phòng bị nhiễu/lệch (ZIP 20261029_135827, job SCAN4)

## 1. Hiện trạng và số liệu chứng minh

| Hạng mục | Số liệu đo | Ý nghĩa |
|---|---|---|
| Kích thước phòng | 2146 × 1636 × 2191 mm, 3,511 m² | Số đo tổng thể hợp lý |
| Độ dày mặt phẳng trong PLY | Độ lệch chuẩn ~5 cm; chỉ ~60% điểm trong 3 cm quanh mặt phẳng | Tường dày, không phẳng |
| Tường nhân đôi | 2 lớp tường lệch ~20 cm (nhìn từ trên xuống) | Dữ liệu không trùng khít |
| Điểm lơ lửng | Đám điểm giữa phòng đáng lẽ trống | Điểm depth sai đi vào TSDF |
| Số điểm | 580.798 điểm trong PLY so với ~63.500 điểm đo thật | Khoảng 9 lần là điểm do mạng suy diễn |
| Cảm biến | VL53L8CX, 8×8 = 64 điểm/khung, 14 khung/giây; nhiễu ToF trung vị 12 mm, p90 48 mm | Rất thưa và nhiễu |
| Hiệu chuẩn | `lidar_camera_extrinsics.json`: `source: bootstrap`, `applied: false` | Chưa hiệu chuẩn thật |
| Báo cáo chất lượng | `quality: warning`, Manhattan xoay 36,6°, bộ lọc chỉ loại 7% điểm | Pipeline tự cảnh báo nhưng vẫn xuất kết quả |
| ARCore thô | Mất tracking 3 lần (37 s, 51 s, 65 s), nhảy 14/23/41 cm | Pose thô bị đứt |
| VIO của pipeline | Bước nhảy lớn nhất 13,5 cm (so với 40,5 cm); quãng đường 5,73 m (so với 6,32 m); đầu–cuối lệch 15 cm (so với 13 cm) | Chỉ giảm được bước nhảy, không cải thiện độ khép vòng |
| Quỹ đạo | Quãng đường ~6 m nhưng span chỉ 0,7–0,8 m | Người quét đứng một chỗ, rung và đưa qua lại |
| Phép thử dựng chỉ từ điểm ToF thật | Pose ARCore và VIO cho độ phẳng tường gần như giống nhau (~28% trong 3 cm) | Pose không phải điểm nghẽn chính |
| Thử hướng lưới ToF | Hoán đổi hàng/cột (xoay 270° + lật gương): 28% lên ~52% với cả hai nguồn pose | Hướng lưới hoặc hiệu chuẩn sai |

## 2. Nguyên nhân khả dĩ

| # | Nguyên nhân | Mức chắc chắn | Căn cứ |
|---|---|---|---|
| 1 | Hướng lưới ToF và hiệu chuẩn LiDAR–camera sai (đang dùng bootstrap) | Khá cao | Sửa hướng lưới tăng độ phẳng 28% lên 52%, độc lập với nguồn pose. Chưa chạy lại toàn pipeline để xác nhận mức cải thiện PLY |
| 2 | Depth completion tạo hàng trăm nghìn điểm suy diễn từ 64 điểm/khung | Cao | 63.500 điểm thật thành 580.798 điểm; đám sương giữa phòng |
| 3 | VIO không kiểm soát độ khép vòng, chỉ làm mượt bước nhảy | Trung bình | Đầu–cuối lệch 15 cm trên span 0,7–0,8 m; VIO vẫn "đạt" mọi cổng |
| 4 | Nhiễu ToF và chùm tia rộng (~10 cm ở 1 m); tầm đo thực tế chỉ 0,4–1,5 m | Trung bình | Nhiễu giữa 2 khung 12–48 mm; tường xa không được phủ |
| 5 | Cách quét: quỹ đạo nhỏ, rung, xoay nhanh | Trung bình | Span 0,7 m; quãng đường 6 m; tốc độ xoay đỉnh trước các chỗ đứt |
| 6 | Pipeline nuốt cảnh báo (fail-open), vẫn xuất kết quả khi dữ liệu kém | Cao | `quality: warning` nhưng job vẫn `COMPLETED` |

## 3. Mục tiêu và kế hoạch xử lý

Mục tiêu: tường/sàn/trần phẳng (độ lệch chuẩn dưới 1–2 cm), không nhân đôi, không điểm lơ lửng; số đo phòng giữ ổn định.

| Bước | Việc làm | Tiêu chí hoàn thành | Phụ trách |
|---|---|---|---|
| 1 | Hiệu chuẩn thật LiDAR–camera (xoay, tịnh tiến) và cảnh báo rõ khi extrinsics còn là `bootstrap` | `applied: true`; tỉ lệ điểm trong 3 cm quanh mặt phẳng tăng rõ so với 28% | Người phụ trách GPU |
| 2 | Tự kiểm tra hướng lưới ToF trước khi dựng, chọn hướng cho độ phẳng cao nhất, ghi vào `quality_report.json` | Báo cáo ghi hướng đã chọn và điểm độ phẳng | Người phụ trách GPU |
| 3 | Chạy lại toàn pipeline với hướng lưới đã sửa, so PLY trước và sau | Đo lại độ dày mặt phẳng và tường nhân đôi trên ZIP này và vài ZIP khác | Người phụ trách GPU, kèm mình hỗ trợ đo |
| 4 | Siết depth completion: chỉ giữ ô ToF mã trạng thái 5 và 9; giảm điểm suy diễn | Số điểm suy diễn giảm; hết đám sương giữa phòng | Người phụ trách GPU |
| 5 | Dựng từ mặt phẳng đã ép Manhattan, làm sạch mạnh hơn mức 7% hiện tại | Mesh/PLY sạch, tường 1 lớp | Người phụ trách GPU |
| 6 | Thêm cổng chất lượng khép vòng cho VIO (lệch đầu–cuối) và dừng hoặc cảnh báo khi chất lượng kém | Job báo lỗi hoặc cảnh báo rõ khi pose không đạt | Người phụ trách GPU |
| 7 | Hướng dẫn người quét: đi chậm, quét rộng, tránh xoay nhanh, quay về điểm xuất phát | Quỹ đạo span lớn hơn, quãng đường ngắn hơn | Đội vận hành, ứng dụng quét |
| 8 | (Tùy chọn) 2DGS hoặc PGSR chỉ để xem đẹp, sau khi hiệu chuẩn chuẩn | Có bản xem đẹp, không dùng để đo | Sau bước 1–6 |

Ghi chú: phần Temporal không phải đổi. Bước 3 cần chạy trên cùng dữ liệu để có số so sánh; mình có thể giúp đo kết quả.
