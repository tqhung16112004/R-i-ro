
import gradio as gr
import joblib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import re

def generate_recommendation(contrib, feature_names, y_pred):
    # Sắp xếp theo độ lớn tuyệt đối
    idx = np.argsort(np.abs(contrib))[::-1]
    names_sorted = [feature_names[i] for i in idx]
    vals_sorted = contrib[idx]

    # Mapping tên biến → mô tả dễ hiểu
    mapping = {
        "ln_REV": "Doanh thu (log)",
        "DER": "Tỷ lệ nợ trên vốn chủ sở hữu (DER)",
        "ROA": "Khả năng sinh lợi trên tài sản (ROA)",
        "RETA": "Khả năng giữ lại lợi nhuận (RETA)",
        "CR": "Khả năng thanh toán hiện hành (CR)",
        "WCTA": "Vốn lưu động trên tổng tài sản (WCTA)",
        "gCPI": "Mức tăng CPI (lạm phát)",
        "Rck": "Lãi suất",
        "ln_DTC3": "Mức đầu tư công (log)"
    }

    # Phần mở đầu
    report = []

    if y_pred < 1.8:
        report.append("Kết luận: Doanh nghiệp đang trong vùng rủi ro cao.")
    elif y_pred < 3.0:
        report.append("Kết luận: Doanh nghiệp trong vùng xám, cần theo dõi.")
    else:
        report.append("Kết luận: Doanh nghiệp trong vùng an toàn.")

    # Phân tích yếu tố
    neg_vars = [(mapping.get(n, n), v) for n, v in zip(names_sorted, vals_sorted) if v < 0]
    pos_vars = [(mapping.get(n, n), v) for n, v in zip(names_sorted, vals_sorted) if v > 0]

    if neg_vars and y_pred < 3.0:  # chỉ nhấn mạnh yếu tố xấu khi còn rủi ro
        report.append("\nCác yếu tố gây bất lợi:")
        for name, v in neg_vars[:3]:
            report.append(f"- {name}: ảnh hưởng tiêu cực ({abs(v):.3f}).")

    if pos_vars:
        report.append("\nCác yếu tố hỗ trợ:")
        for name, v in pos_vars[:3]:
            report.append(f"- {name}: ảnh hưởng tích cực ({v:.3f}).")

    # Khuyến nghị
    advice = []
    if y_pred < 1.8:  # doanh nghiệp rủi ro cao
        for name, v in neg_vars[:3]:
            if "nợ" in name.lower() or "DER" in name:
                advice.append("Giảm tỷ lệ nợ, tái cấu trúc vốn để giảm áp lực tài chính.")
            elif "ROA" in name:
                advice.append("Tăng hiệu quả sử dụng tài sản để cải thiện ROA.")
            elif "thanh toán" in name:
                advice.append("Cải thiện khả năng thanh khoản bằng cách quản lý tài sản ngắn hạn.")
            elif "doanh thu" in name:
                advice.append("Đẩy mạnh tăng trưởng doanh thu qua mở rộng thị trường/sản phẩm.")
            elif "lạm phát" in name or "CPI" in name:
                advice.append("Theo dõi biến động lạm phát và kiểm soát chi phí.")
            elif "lãi suất" in name:
                advice.append("Xem xét tái cơ cấu khoản vay để giảm rủi ro lãi suất.")
            else:
                advice.append(f"Chú ý cải thiện chỉ số {name}.")
    elif y_pred < 3.0:  # vùng xám
        advice.append("Doanh nghiệp cần cân bằng lại tài chính, tránh để rủi ro gia tăng.")
        advice.append("Ưu tiên cải thiện các chỉ số tài chính có ảnh hưởng tiêu cực.")
    else:  # an toàn
        advice.append("Doanh nghiệp đang an toàn, nên xem xét tái đầu tư lợi nhuận.")
        advice.append("Tận dụng nguồn vốn để mở rộng hoạt động kinh doanh hoặc đầu tư mới.")
        advice.append("Có thể giảm bớt thanh khoản dư thừa để tối ưu hiệu quả sử dụng vốn.")

    if advice:
        report.append("\nKhuyến nghị:")
        report.extend([f"- {a}" for a in advice])

    return "\n".join(report)
    
# Dataset dữ liệu vĩ mô
macro_data = {
    2017: {"CPI":153.68},
    2018: {"CPI":159.11, "RCK":4.25, "DTC":425225000000000},
    2019: {"CPI":163.56, "RCK":4.2, "DTC":453106000000000},
    2020: {"CPI":168.83, "RCK":3.25, "DTC":468411000000000},
    2021: {"CPI":171.93, "RCK":2.5, "DTC":463566000000000},
    2022: {"CPI":177.35, "RCK":3.5, "DTC":461107000000000},
    2023: {"CPI":183.12, "RCK":3.0, "DTC":518787000000000},
    2024: {"CPI":189.76, "RCK":3.0, "DTC":491046000000000},
    2025: {"CPI":197.63, "RCK":2.75, "DTC":539345000000000},
}

# Hàm format VND
def format_vnd(value):
    try:
        num = re.sub(r"[^\d]", "", str(value))
        if num == "":
            return ""
        formatted = "{:,}".format(int(num)).replace(",", ".")
        return formatted
    except:
        return value

# Hàm parse VND về float
def parse_vnd(value):
    try:
        return float(value.replace(".", ""))
    except:
        return 0.0
        
# Theo báo cáo chính phủ dự báo CPI 2025 tăng 4.15% so với năm 2024. RCK giảm 25 điểm phần trăm so với 2024
def autofill_macro(nam):
    if nam in macro_data and (nam-1) in macro_data:
        cpi_t = macro_data[nam]["CPI"]
        cpi_t1 = macro_data[nam-1]["CPI"]
        rck = macro_data[nam]["RCK"]
        dtc_t3 = macro_data[nam]["DTC"]
        return cpi_t, cpi_t1, rck, dtc_t3
    else:
        return None, None, None, None
# Load model
model = joblib.load("xgb_model.pkl")
feature_names = ["ln_REV", "DER", "ROA", "RETA", "CR", "WCTA", "gCPI", "Rck", "ln_DTC3"]

# Import shap và tạo explainer
import shap
explainer = shap.TreeExplainer(model)

def predict_and_plot(doanh_thu, no_phai_tra, von_chu_so_huu,
                     loi_nhuan_sau_thue, tong_ts_t, tong_ts_t1,
                     loi_nhuan_giu_lai, ts_ngan_han, no_ngan_han,
                    nam):
    # Chuyển các input text có dấu chấm thành số float
    doanh_thu = parse_vnd(doanh_thu)
    loi_nhuan_sau_thue = parse_vnd(loi_nhuan_sau_thue)
    no_phai_tra = parse_vnd(no_phai_tra)
    von_chu_so_huu = parse_vnd(von_chu_so_huu)
    loi_nhuan_giu_lai = parse_vnd(loi_nhuan_giu_lai)
    ts_ngan_han = parse_vnd(ts_ngan_han)
    tong_ts_t = parse_vnd(tong_ts_t)
    tong_ts_t1 = parse_vnd(tong_ts_t1)
    no_ngan_han = parse_vnd(no_ngan_han)
    # Tự động lấy dữ liệu vĩ mô
    cpi_t, cpi_t1, rck, dtc_t3 = autofill_macro(nam)
    
    # Feature engineering
    ln_REV = np.log1p(doanh_thu)
    DER = no_phai_tra / von_chu_so_huu if von_chu_so_huu else 0
    denom = (tong_ts_t + tong_ts_t1) / 2 or 1
    ROA = loi_nhuan_sau_thue / denom
    RETA = loi_nhuan_giu_lai / (tong_ts_t or 1)
    CR = ts_ngan_han / (no_ngan_han or 1)
    WCTA = (ts_ngan_han - no_ngan_han) / (tong_ts_t or 1)
    ln_DTC3 = np.log1p(dtc_t3)
    gCPI = ((cpi_t - cpi_t1)/cpi_t1)

    x = np.array([[ln_REV, DER, ROA, RETA, CR, WCTA, gCPI, rck, ln_DTC3]])
    y_pred = model.predict(x)[0]

    # Risk label & color
    if y_pred < 1.8:
        risk_label = " Nguy cơ phá sản"
        color = "#f44336"
    elif y_pred < 3.0:
        risk_label = " Vùng xám"
        color = "#ff9800"
    else:
        risk_label = " An toàn"
        color = "#4caf50"

    # Gauge chart (Plotly)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=y_pred,
        gauge={
            'axis': {'range': [0, 5]},
            'bar': {'color': '#37474f','thickness': 0.3},
            'steps': [
                {'range': [0, 1.8], 'color': '#ef9a9a'},
                {'range': [1.8, 3.0], 'color': '#ffe082'},
                {'range': [3.0, 5], 'color': '#a5d6a7'},
            ]
        },
        title={'text': "<b>Z-score</b>", 'font': {'size': 16}}
    ))
    fig_gauge.update_layout(height=250, margin={'t':40,'b':20,'l':20,'r':20})

    # --- NEW: Local importance bằng SHAP ---
    shap_values = explainer.shap_values(x)
    contrib = shap_values[0]  # vì chỉ có 1 quan sát
    idx = np.argsort(np.abs(contrib))[::-1]
    
    fig_bar, ax = plt.subplots(figsize=(6,6))
    colors = ['#ef5350' if v < 0 else '#66bb6a' for v in contrib[idx]]
    ax.barh([feature_names[i] for i in idx], contrib[idx], color=colors)
    ax.set_xlabel("SHAP value")
    ax.set_title("Đóng góp của biến (local importance)", fontsize=16, weight="bold", pad=30)
    ax.invert_yaxis()
    
    fig_bar.tight_layout(rect=[0, 0, 1, 0.9])
    

    # --- NEW: Gọi hàm generate_recommendation ---
    recommendation = generate_recommendation(contrib, feature_names, y_pred)
    
    return f"{y_pred:.4f}", risk_label, fig_gauge, fig_bar, recommendation
    
# Material CSS
material_css = """
/* Body toàn trang*/
body {
  font-family: 'Roboto', sans-serif;
  margin: 0; padding: 0;
  min-height: 100vh;
  background: linear-gradient(-45deg, #fffffa, #f5fff6, #fffffa, #f5fff6);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
  color: #333;
}
/* Animation đổi màu nền */
@keyframes gradientBG {0%   {background-position: 0% 50%;}50%  {background-position: 100% 50%;}100% {background-position: 0% 50%;}}
/* Header */
.header {
  background: #0C626B;
  color: white;padding: 20px;
  text-align: center;
  font-size: 28px;
  font-weight: bold
}
/* Card nhập liệu */
.card {
  background: #f2ffff;
  padding: 20px;
  border-radius: 22px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  margin: 10px
}
/* Card kết quả */
.card-alt {
  background: #f2ffff;
  padding: 20px;
  border-radius: 22px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
  margin: 10px;
}
/* Nút bấm */
.btn-primary {
  background: #ffffff;
  color: #1565c0 !important;
  padding: 12px 28px;
  border-radius: 28px;
  border: 2px solid #1565c0;
  font-size: 22px;
  font-weight: 700;
  cursor: pointer;
  display: inline-block !important;
  width: auto !important;
  min-width: 180px;
  max-width: fit-content;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.btn-primary:hover {
  background: #1565c0;
  color: #ffffff !important;
  transform: scale(1.05);
}
.center-btn {
  display: flex !important;
  justify-content: center !important;
  align-items: center !important; /* căn giữa theo cả chiều dọc */
}
/* Footer */
.footer {background: #263238; color: #ccc; text-align: center; padding: 15px; font-size: 14px; border-top: 2px solid rgba(255,255,255,0.1)}
.custom-title {
  color: #E16516 !important;
  font-family: 'Roboto', sans-serif !important;
  font-weight: 700 !important;
}
"""
with gr.Blocks(css=material_css) as demo:
    gr.HTML("""
      <div class='header'>
        DỰ BÁO RỦI RO PHÁ SẢN DOANH NGHIỆP
        <div style="font-size:16px; font-weight:400; margin-top:5px; color:#f0f0f0;">
          Đây là công cụ dự báo dành cho doanh nghiệp
        </div>
      </div>
      """)

    gr.HTML("<h2 class='custom-title'>NHẬP THÔNG TIN</h2>")

    # HÀNG 1: INPUT (3 cột)
    with gr.Row():
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### Năm dự báo")
            year_input = gr.Number(label="Năm",placeholder="Nhập năm...")        
            gr.Markdown("### Kết quả kinh doanh")
            doanh_thu = gr.Textbox(label="Tổng doanh thu (Đồng)", placeholder="Nhập số...")
            doanh_thu.blur(fn=format_vnd, inputs=doanh_thu, outputs=doanh_thu)
            loi_nhuan_sau_thue = gr.Textbox(label="Lợi nhuận sau thuế TNDN (Đồng)",placeholder="Nhập số...")
            loi_nhuan_sau_thue.blur(fn=format_vnd, inputs=loi_nhuan_sau_thue, outputs=loi_nhuan_sau_thue)

            
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### Bảng cân đối kế toán")       
            ts_ngan_han = gr.Textbox(label="Tài sản ngắn hạn (Đồng)",placeholder="Nhập số...")           
            ts_ngan_han.blur(fn=format_vnd, inputs=ts_ngan_han, outputs=ts_ngan_han)
    
            tong_ts_t = gr.Textbox(label="Tổng cộng tài sản năm dự báo (Đồng)",placeholder="Nhập số...")
            tong_ts_t.blur(fn=format_vnd, inputs=tong_ts_t, outputs=tong_ts_t)
    
            tong_ts_t1 = gr.Textbox(label="Tổng cộng tài sản năm trước (Đồng)",placeholder="Nhập số...")
            tong_ts_t1.blur(fn=format_vnd, inputs=tong_ts_t1, outputs=tong_ts_t1)
    
            no_phai_tra = gr.Textbox(label="Nợ phải trả (Đồng)",placeholder="Nhập số...")    
            no_phai_tra.blur(fn=format_vnd, inputs=no_phai_tra, outputs=no_phai_tra)
        
        with gr.Column(scale=1, elem_classes="card"):
            gr.Markdown("### Bảng cân đối kế toán (tiếp)")          
     
            no_ngan_han = gr.Textbox(label="Nợ ngắn hạn (Đồng)",placeholder="Nhập số...") 
            no_ngan_han.blur(fn=format_vnd, inputs=no_ngan_han, outputs=no_ngan_han)
    
            von_chu_so_huu = gr.Textbox(label="Vốn chủ sở hữu (Đồng)",placeholder="Nhập số...")
            von_chu_so_huu.blur(fn=format_vnd, inputs=von_chu_so_huu, outputs=von_chu_so_huu)
    
            loi_nhuan_giu_lai = gr.Textbox(label="LNST chưa phân phối kỳ này (Đồng)",placeholder="Nhập số...")   
            loi_nhuan_giu_lai.blur(fn=format_vnd, inputs=loi_nhuan_giu_lai, outputs=loi_nhuan_giu_lai)
            gr.Markdown("""
            **Chú thích:**
            Các chỉ số được xem xét lấy theo năm dự báo
            """)
    
    # Nút phân tích căn giữa
    with gr.Row():
        with gr.Column(scale=1):
            pass
        with gr.Column(scale=1, elem_classes="center-btn"):  # <-- áp dụng class ở đây
            btn = gr.Button("Phân tích", elem_classes="btn-primary")
        with gr.Column(scale=1):
            pass

    # HÀNG 2: OUTPUT (3 cột)
    gr.HTML("<h2 class='custom-title'>KẾT QUẢ PHÂN TÍCH</h2>")
    with gr.Row():
        with gr.Column(scale=1, elem_classes="card-alt"):
            zscore_out = gr.Textbox(label="Z-score", interactive=False)
            risk_out = gr.Textbox(label="Phân loại rủi ro", interactive=False)
            gauge_out = gr.Plot(label="Gauge Z-score")
            gr.Markdown("""
            **Chú thích:**
            - Màu đỏ (0 - 1.8): Nguy cơ phá sản cao
            - Màu vàng (1.8 - 3.0): Vùng xám, cần theo dõi thêm
            - Màu xanh (3.0 - 5.0): Doanh nghiệp an toàn
            """)
            
        with gr.Column(scale=1, elem_classes="card-alt"):
            feat_out = gr.Plot(label="Feature Importance")
            gr.Markdown(""" **Chú thích:**
            - Trục dọc: Tên các biến tài chính và kinh tế vĩ mô được sử dụng trong mô hình.
            - Trục ngang: Mức độ quan trọng (Importance) của từng biến trong việc dự báo rủi ro phá sản.
            - Giá trị càng cao → Biến đó càng ảnh hưởng mạnh đến kết quả dự báo.
            """)
        
        with gr.Column(scale=1, elem_classes="card-alt"):
            gr.Markdown("### Kết luận và Khuyến nghị")
            rec_out = gr.Markdown(label="Báo cáo & Khuyến nghị")

    gr.HTML("""
    <div class='footer'>
      GVHD: PGS.TS. Trương Đức Toàn<br>
      Nhóm sinh viên:<br>
      Hoàng Hiền Giang<br>
      Tạ Đình Đồng<br>
      Trịnh Quang Hưng
    </div>
    """)

    btn.click(
        fn=predict_and_plot,
        inputs=[doanh_thu,no_phai_tra,von_chu_so_huu,loi_nhuan_sau_thue,
                tong_ts_t,tong_ts_t1,loi_nhuan_giu_lai,ts_ngan_han,no_ngan_han,
                year_input],
        outputs=[zscore_out,risk_out,gauge_out,feat_out,rec_out]
    )

demo.launch(share=True)
