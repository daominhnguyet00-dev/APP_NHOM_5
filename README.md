# 📈 Hệ Thống Kiểm Định Chiến Lược Định Lượng SMA & OBV Trên Sàn HOSE

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Plotly Interactive](https://img.shields.io/badge/Plotly-Interactive%20Charts-orange)](https://plotly.com/)

> **Nghiên cứu & Xây dựng bởi:** Nhóm 5  
> **Dữ liệu phân tích:** Toàn bộ rổ cổ phiếu niêm yết trên Sở Giao dịch Chứng khoán TP.HCM (HOSE) giai đoạn 2020 – 2023.

---

## 📌 Giới Thiệu Dự Án

Web App này được phát triển nhằm mục đích **kiểm định tính hiệu quả và kiểm chứng thực nghiệm (Empirical Backtesting)** của chiến lược đầu tư định lượng kết hợp giữa tín hiệu xu hướng giá (**SMA - Simple Moving Average**) và áp lực dòng tiền tích lũy (**OBV - On-Balance Volume**) trên thị trường chứng khoán Việt Nam.

Ứng dụng số hóa toàn bộ quy trình nghiên cứu từ notebook `NHOM_5_tieu_luan_SMA_OBV.ipynb` thành một bảng điều khiển tương tác trực quan (**Interactive Quantitative Dashboard**) trên nền tảng **Streamlit**, cho phép các nhà đầu tư, chuyên gia phân tích và giảng viên kiểm tra độ vững, điều chỉnh tham số và phân tích độ nhạy trong các điều kiện thị trường khắc nghiệt.

---

## 🏗️ Kiến Trúc Hệ Thống Định Lượng Ba Tầng (3-Tier Framework)

Hệ thống được thiết kế theo tư duy định lượng chuẩn mực, nhằm loại bỏ hoàn toàn các lỗi thiên lệch thường gặp trong tài chính như *Look-ahead bias* (nhìn trước tương lai) và *Data Snooping bias* (khai thác quá mức dữ liệu):

```
┌────────────────────────────────────────────────────────────────────────┐
│  TẦNG 1: SÀNG LỌC CỔ PHIẾU BẰNG TA SCORE                              │
│  - Trend R²: Độ rõ xu hướng từ hồi quy log(giá) theo thời gian (30%)   │
│  - Momentum 6M: Động lượng tỷ suất sinh lời 6 tháng (25%)              │
│  - Volume: Giá trị giao dịch bình quân - Thanh khoản thị trường (25%)   │
│  - Risk: Nghịch đảo biến động hàng năm & Max Drawdown (20%)            │
│  ===> Sàng lọc Top 5 cổ phiếu dẫn đầu: DIG, DCM, MSN, SSI, PDR         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TẦNG 2: TÍN HIỆU GIAO DỊCH KỸ THUẬT SMA & OBV                         │
│  - Tín hiệu SMA: Giao cắt giữa đường trung bình động ngắn và dài       │
│  - Tín hiệu OBV: Giao cắt giữa chỉ báo On-Balance Volume và đường MA   │
│  - Quy tắc kiểm định: SMA đơn lẻ, OBV đơn lẻ, OR (Hoặc), AND (Và)      │
│  - Tối ưu hóa tham số bằng Hyperopt TPE trên tập Train (2020 - 2021)   │
│  - Ràng buộc thanh toán thực tế: Giữ vị thế tối thiểu 3 phiên (T+2.5)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  TẦNG 3: QUẢN TRỊ DANH MỤC & PHÂN BỔ VỐN                              │
│  - Phân bổ đều Equal Weight (EW): 20% mỗi mã                           │
│  - Phân bổ Hiện đại MPT (Modern Portfolio Theory): Tối đa Sharpe       │
│  - Tái cân bằng định kỳ đầu mỗi tháng (Monthly Rebalancing)            │
│  - Khấu trừ chi phí giao dịch tái cơ cấu danh mục thực tế              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Kết Quả Thực Nghiệm Nổi Bật (Tập Test 2022 - Bear Market)

Năm 2022 là giai đoạn thị trường chứng khoán Việt Nam trải qua đợt sụt giảm nghiêm trọng nhất trong thập kỷ. Đây là môi trường lý tưởng để kiểm chứng năng lực phòng vệ vốn của chiến lược:

| Chiến lược / Benchmark | Tổng Lợi Nhuận (%) | Sharpe Ratio | Max Drawdown (%) | Beta vs VN-Index | Alpha vs VN-Index (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SMA + OBV (AND) - MPT** ⭐ | **-13.09%** | **-0.99** | **-18.41%** | **0.12** | **+20.13%** |
| **SMA + OBV (AND) - EW** | **-15.44%** | **-1.20** | **-19.18%** | **0.12** | **+17.79%** |
| SMA + OBV (OR) - MPT | -24.85% | -1.03 | -33.37% | 0.51 | +8.37% |
| SMA Crossover - MPT | -24.34% | -1.68 | -27.37% | 0.25 | +8.89% |
| OBV Filter - MPT | -24.17% | -0.95 | -32.97% | 0.53 | +9.05% |
| **VN-Index (Thị trường chung)** | **-33.22%** | **-1.52** | **-40.34%** | **1.00** | **0.00%** |
| **EW Buy & Hold Toàn rổ (99 mã)** | **-49.63%** | **-1.73** | **-61.55%** | **1.35** | **-16.41%** |
| **EW Buy & Hold Top 5** | **-56.81%** | **-2.10** | **-59.34%** | **1.26** | **-23.58%** |
| **TA Score Xoay vòng tháng** | **-60.10%** | **-2.46** | **-63.35%** | **1.08** | **-26.88%** |

### 💡 Phát hiện đắt giá:
1. **Bộ lọc kép (AND Rule) tạo lá chắn phòng vệ xuất sắc**: Bằng cách đòi hỏi cả tín hiệu giá (SMA) và áp lực khối lượng (OBV) đồng thuận, chiến lược chỉ giữ vị thế cổ phiếu trong **6% - 34%** số phiên trong năm 2022, tự động giữ phần lớn vốn ở trạng thái tiền mặt để né cú sập của thị trường.
2. **Chiến lược Xoay vòng TA Score thất bại (-60.10%)**: Việc chỉ chọn lại Top cổ phiếu đà tăng mỗi tháng mà không có bộ lọc tín hiệu kỹ thuật sẽ liên tục bắt đỉnh các cổ phiếu tăng nóng cuối chu kỳ.

---

## 🚀 Các Tính Năng Chính Trên Web App

1. **📊 Bảng Điều Khiển Hiệu Năng & So Sánh Đa Chiều**:
   - Biểu đồ tăng trưởng tài sản (Normalized Equity Curves) tương tác với Plotly.
   - Đồ thị độ sụt giảm vốn từ đỉnh (Underwater Drawdown Chart).
   - Bảng phân tích đầy đủ các chỉ số: Total Return, Annual Volatility, Sharpe Ratio, Calmar Ratio, Beta, Alpha.
   - Xuất dữ liệu bảng kết quả ra file CSV chỉ với 1 click.

2. **🎯 Tầng 1: Tương Tác Trọng Số TA Score**:
   - Tùy chỉnh trọng số Trend, Momentum, Volume, Risk trực tiếp qua thanh trượt.
   - Xem bảng xếp hạng Top N cổ phiếu, ma trận tương quan Spearman giữa các tín hiệu con.
   - Biểu đồ phân tích độ vững của Top 5 trên lưới 285 tổ hợp trọng số.

3. **⚡ Tầng 2: Soi Tín Hiệu Kỹ Thuật Chi Tiết**:
   - Đồ thị nến Nhật (Candlestick) kèm các đường SMA ngắn/dài.
   - Điểm đánh dấu tín hiệu MUA (Tam giác xanh) và BÁN (Tam giác đỏ) trực quan trên từng phiên.
   - Panel phụ hiển thị đường chỉ báo OBV và OBV MA.
   - Cho phép điều chỉnh nhanh tham số `ma_short`, `ma_long`, `obv_window` để kiểm thử giả thuyết.

4. **⚖️ Tầng 3: Phân Bổ MPT & Tái Cân Bằng Danh Mục**:
   - Biểu đồ cơ cấu tỷ trọng danh mục MPT (Donut chart).
   - Lịch sử tái cân bằng danh mục định kỳ đầu mỗi tháng.

5. **🔬 Phân Tích Độ Nhạy Toàn Diện**:
   - Kiểm định độ nhạy giả định khớp lệnh (Giá đóng cửa $t$ vs Giá mở cửa $t+1$, có/không T+2, có/không phí).
   - Đo lường bước nhảy giá qua đêm (Overnight Gap / Slippage).
   - Đánh giá khả năng chịu đựng chi phí giao dịch (từ 0% đến 0.8% 2 chiều).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```
APP_NHOM_5/
│
├── app.py                         # Ứng dụng Web App Streamlit chính
├── requirements.txt               # Danh sách thư viện phụ thuộc Python
├── README.md                      # Tài liệu hướng dẫn sử dụng & báo cáo
├── HOSE_2020_2023_in.csv          # Dữ liệu giá & khối lượng HOSE 2020-2023
└── NHOM_5_tieu_luan_SMA_OBV.ipynb # Jupyter Notebook nghiên cứu gốc
```

---

## 💻 Hướng Dẫn Cài Đặt & Chạy Cục Bộ (Local)

### 1. Yêu cầu hệ thống:
- Python 3.9 trở lên
- Git

### 2. Các bước cài đặt:

```bash
# 1. Clone repository về máy tính
git clone https://github.com/<tai-khoan-github-cua-ban>/APP_NHOM_5.git
cd APP_NHOM_5

# 2. Tạo môi trường ảo (Khuyến nghị)
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate
# Kích hoạt trên macOS/Linux:
source venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 4. Khởi chạy ứng dụng Streamlit
streamlit run app.py
```

Sau khi chạy lệnh trên, ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: `http://localhost:8501`.

---

## ☁️ Hướng Dẫn Triển Khai (Deploy) Lên Streamlit Community Cloud

Bạn có thể đưa ứng dụng lên mạng Internet hoàn toàn miễn phí chỉ với 3 bước:

### Bước 1: Đẩy code lên GitHub
1. Tạo một repository mới trên GitHub (ví dụ: `hose-sma-obv-quant`).
2. Tải toàn bộ các file trong thư mục dự án lên repository (bao gồm `app.py`, `requirements.txt`, `README.md`, và `HOSE_2020_2023_in.csv`).

```bash
git init
git add .
git commit -m "Khoi tao web app kiem dinh dinh luong SMA OBV"
git branch -M main
git remote add origin https://github.com/<tai-khoan-cua-ban>/hose-sma-obv-quant.git
git push -u origin main
```

### Bước 2: Đăng nhập Streamlit Community Cloud
1. Truy cập vào trang: [share.streamlit.io](https://share.streamlit.io/)
2. Đăng nhập bằng tài khoản GitHub của bạn.

### Bước 3: Deploy ứng dụng
1. Nhấn nút **"New app"**.
2. Chọn Repository, Branch (`main`), và đường dẫn Main file path là: `app.py`.
3. Nhấn **"Deploy!"**.
4. Chờ 1–2 phút để hệ thống build môi trường. Ứng dụng của bạn sẽ có một đường link công khai (ví dụ: `https://your-app.streamlit.app`) để gửi cho bạn bè, giảng viên hoặc hội đồng đánh giá!

---

## 📜 Giấy Phép & Đóng Góp

Dự án được phân phối dưới giấy phép **MIT License**. Mọi đóng góp, phản hồi hoặc cải tiến xin vui lòng mở một Issue hoặc tạo một Pull Request trên GitHub.

*Chúc bạn có những trải nghiệm khám phá thú vị cùng Hệ thống Kiểm định Định lượng của Nhóm 5!*
