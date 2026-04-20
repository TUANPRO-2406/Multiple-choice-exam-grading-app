# import cv2
# import numpy as np
# from ultralytics import YOLO
# import base64
# import json
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse

# # ==========================================
# # 1. CẤU HÌNH HỆ THỐNG & TẢI MODEL
# # ==========================================
# app = FastAPI(title="Hệ thống chấm thi trắc nghiệm AI")

# MODEL_PATH = 'best.pt'
# W_TARGET, H_TARGET = 1000, 1400  
# CHOICES = ['A', 'B', 'C', 'D']

# try:
#     model = YOLO(MODEL_PATH)
#     print(f"Đã tải thành công model từ {MODEL_PATH}")
# except Exception as e:
#     print(f"Lỗi tải model YOLO: {e}")

# # ==========================================
# # 2. HÀM LÕI XỬ LÝ ẢNH 
# # ==========================================
# def process_grading_core(img, all_ans_keys):
#     # --- A. TIỀN XỬ LÝ & DUỖI ẢNH BẰNG ĐIỂM BIÊN (Ý TƯỞNG CỦA BẠN) ---
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thres = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     img_h, img_w = thres.shape

#     frame_w = img_w * 0.95
#     frame_h = frame_w * 1.4142
    
#     if frame_h > img_h * 0.95:
#         frame_h = img_h * 0.95
#         frame_w = frame_h / 1.4142

#     left = int((img_w - frame_w) / 2)
#     top = int((img_h - frame_h) / 2)
    
#     box_w = int(frame_w * 0.25)
#     box_h = int(frame_h * 0.15) 

#     roi_tl = thres[top : top+box_h, left : left+box_w]
#     roi_tr = thres[top : top+box_h, left + int(frame_w) - box_w : left + int(frame_w)]
#     roi_bl = thres[top + int(frame_h) - box_h : top + int(frame_h), left : left+box_w]
#     roi_br = thres[top + int(frame_h) - box_h : top + int(frame_h), left + int(frame_w) - box_w : left + int(frame_w)]

#     # HÀM TÌM VẠCH ĐEN (Trả về x, y, w, h nguyên bản)
#     def find_anchor(roi_img, offset_x, offset_y, corner_type):
#         cnts, _ = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         valid_cnts = []
#         roi_area = box_w * box_h
        
#         for c in cnts:
#             area = cv2.contourArea(c)
#             x, y, w, h = cv2.boundingRect(c)
#             if h == 0 or w == 0: continue
            
#             ratio = w / float(h)
#             extent = area / float(w * h)

#             if (40 < area < (roi_area * 0.15)) and (1.0 <= ratio <= 3.8) and (extent > 0.65):
#                 valid_cnts.append(c)

#         if not valid_cnts:
#             return None
            
#         # ÁP DỤNG TRỌNG SỐ CHO CỰC TRỊ GÓC 
#         if corner_type == 'TL':   
#             best_c = min(valid_cnts, key=lambda c: cv2.boundingRect(c)[0] + 3 * cv2.boundingRect(c)[1])
#         elif corner_type == 'TR': 
#             best_c = max(valid_cnts, key=lambda c: cv2.boundingRect(c)[0] - 3 * cv2.boundingRect(c)[1])
#         elif corner_type == 'BL': 
#             best_c = max(valid_cnts, key=lambda c: 3 * cv2.boundingRect(c)[1] - cv2.boundingRect(c)[0])
#         elif corner_type == 'BR': 
#             best_c = max(valid_cnts, key=lambda c: cv2.boundingRect(c)[0] + 3 * cv2.boundingRect(c)[1])

#         x, y, w, h = cv2.boundingRect(best_c)
#         return [x + offset_x, y + offset_y, w, h]

#     # TRUY TÌM 4 MỐC
#     box_tl = find_anchor(roi_tl, left, top, 'TL')
#     box_tr = find_anchor(roi_tr, left + int(frame_w) - box_w, top, 'TR')
#     box_bl = find_anchor(roi_bl, left, top + int(frame_h) - box_h, 'BL')
#     box_br = find_anchor(roi_br, left + int(frame_w) - box_w, top + int(frame_h) - box_h, 'BR')

#     if not box_tl or not box_tr or not box_bl or not box_br:
#         raise ValueError("Không tìm thấy đủ 4 vạch biên hợp lệ. Xin hãy chỉnh lại góc máy.")

#     # LẤY CÁC TỌA ĐỘ GÓC NGOÀI CÙNG NHẤT
#     pt_tl = [box_tl[0], box_tl[1]]                               
#     pt_tr = [box_tr[0] + box_tr[2], box_tr[1]]                   
#     pt_bl = [box_bl[0], box_bl[1] + box_bl[3]]                   
#     pt_br = [box_br[0] + box_br[2], box_br[1] + box_br[3]]       

#     pts = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype="float32")
    
#     M_mat = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [W_TARGET-1, 0], [W_TARGET-1, H_TARGET-1], [0, H_TARGET-1]], dtype="float32"))
#     warped = cv2.warpPerspective(img, M_mat, (W_TARGET, H_TARGET))

#     gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     _, thres_w = cv2.threshold(gray_w, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     # --- B. TÌM VẠCH NHỊP ---
#     y_marks_raw = []
#     cnts_v, _ = cv2.findContours(thres_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for c in cnts_v:
#         x, y, w, h = cv2.boundingRect(c)
#         if x > 850 and w > h and 5 < w < 60 and 4 < h < 30 and y > 350:
#             y_marks_raw.append(y + h//2)

#     cleaned_y_marks = []
#     for y_val in sorted(y_marks_raw):
#         if not cleaned_y_marks or abs(y_val - cleaned_y_marks[-1]) > 15:
#             cleaned_y_marks.append(y_val)
#     y_marks = cleaned_y_marks[-25:] if len(cleaned_y_marks) >= 25 else cleaned_y_marks

#     # --- C. TÌM 4 KHỐI ĐÁP ÁN (LAI GIỮA DYNAMIC VÀ FALLBACK - Ý TƯỞNG CỦA BẠN) ---
#     col_blocks = []
    
#     # B1: Dùng phép Nở ảnh (Dilation) để nối liền các nét in bị đứt khúc
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     dilated_w = cv2.dilate(thres_w, kernel, iterations=2)

#     cnts_c, _ = cv2.findContours(dilated_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     dynamic_blocks = []
#     for c in cnts_c:
#         x, y, w, h = cv2.boundingRect(c)
#         # Nới lỏng điều kiện bắt thô: Cột cao 600-900, rộng 150-280, nửa dưới ảnh
#         if 600 < h < 900 and 150 < w < 280 and y > 450:
#             dynamic_blocks.append((x, y, w, h))
            
#     dynamic_blocks = sorted(dynamic_blocks, key=lambda b: b[0])

#     # B2: KIỂM TRA ĐIỀU KIỆN KHẮT KHE (y, w, h phải gần bằng nhau)
#     if len(dynamic_blocks) >= 4:
#         b1, b2, b3, b4 = dynamic_blocks[:4]
        
#         y_vals = [b[1] for b in [b1, b2, b3, b4]]
#         w_vals = [b[2] for b in [b1, b2, b3, b4]]
#         h_vals = [b[3] for b in [b1, b2, b3, b4]]
        
#         # Sai số cho phép: y lệch max 30px, w lệch max 40px, h lệch max 50px
#         if (max(y_vals) - min(y_vals) < 30) and (max(w_vals) - min(w_vals) < 40) and (max(h_vals) - min(h_vals) < 50):
#             col_blocks = [b1, b2, b3, b4]
#             print("=> Đã bắt được Khung đáp án DYNAMIC (Đẹp hoàn hảo!)")

#     # B3: FALLBACK (Phương án dự phòng nếu in quá mờ hoặc lỗi)
#     if len(col_blocks) != 4:
#         print("=> DYNAMIC thất bại. Đang kích hoạt FALLBACK Tọa độ cứng.")
#         col_blocks = [
#             (40, 560, 210, 750),   
#             (280, 560, 210, 750),  
#             (515, 560, 210, 750),  
#             (745, 560, 210, 750)   
#         ]

#     # --- D. TÌM KHUNG SBD & MÃ ĐỀ ---
#     info_blocks = []
#     cnts_info, _ = cv2.findContours(thres_w[0:500, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for c in cnts_info:
#         x, y, w, h = cv2.boundingRect(c)
#         if x > 400 and h > 200 and h > w:
#             info_blocks.append((x, y, w, h))

#     info_blocks = sorted(info_blocks, key=lambda b: b[0])

#     rect_sbd = (490, 110, 140, 305)   
#     rect_made = (645, 110, 80, 305)
    
#     if len(info_blocks) >= 2:
#         rect_sbd = info_blocks[0]
#         rect_made = info_blocks[1]

#     # --- E. GIẢI MÃ SBD & MÃ ĐỀ (CHUẨN 5 CỘT, 3 CỘT) ---
#     def decode_info_block(rect, num_cols):
#         x_start, y_start, w, h = rect
#         roi = warped[y_start:y_start+h, x_start:x_start+w]
#         results = model.predict(roi, conf=0.45, verbose=False) 
        
#         digits = {}
#         for box in results[0].boxes:
#             x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#             cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
#             col = int(cx / (w / num_cols))
#             row = int(cy / (h / 10))
#             if 0 <= col < num_cols and 0 <= row < 10:
#                 if col not in digits:
#                     digits[col] = str(row)
                
#         return "".join([digits.get(i, "X") for i in range(num_cols)])

#     sbd = decode_info_block(rect_sbd, 5)
#     made = decode_info_block(rect_made, 3)

#     current_ans_key = {}
#     needs_review = False
#     review_reason = []

#     if "X" in sbd: review_reason.append("Thieu SBD")
#     if "X" in made: review_reason.append("Thieu Ma de")

#     if made in all_ans_keys:
#         current_ans_key = {int(k): v for k, v in all_ans_keys[made].items()}
#     else:
#         if "X" not in made: 
#             review_reason.append(f"Chua co dap an ma de {made}")

#     if len(review_reason) > 0: needs_review = True

#     # --- F. ĐỌC ĐÁP ÁN BÀI LÀM ---
#     final_answers = {} 
#     for i, (bx, by, bw, bh) in enumerate(col_blocks):
#         roi = warped[by:by+bh, bx:bx+bw]
#         results = model.predict(roi, conf=0.45, verbose=False)
#         start_q = i * 25 + 1
        
#         for res in results:
#             for box in res.boxes.xyxy:
#                 x1, y1, x2, y2 = box.cpu().numpy()
#                 y_abs = ((y1 + y2) / 2) + by
#                 x_center_roi = (x1 + x2) / 2
                
#                 if len(y_marks) == 25:
#                     row_idx = min(range(25), key=lambda idx: abs(y_marks[idx] - y_abs))
#                     q_num = start_q + row_idx
                    
#                     if x_center_roi > (bw * 0.2):
#                         choice_idx = int((x_center_roi - (bw * 0.2)) / ((bw * 0.8) / 4))
#                         if 0 <= choice_idx < 4:
#                             ans = CHOICES[choice_idx]
#                             if q_num not in final_answers:
#                                 final_answers[q_num] = []
#                             final_answers[q_num].append((ans, int(x_center_roi + bx), int(y_abs)))

#     # --- G. CHẤM ĐIỂM ---
#     correct_count = 0
#     for q_num, detected_list in final_answers.items():
#         if len(detected_list) == 1:
#             if detected_list[0][0] == current_ans_key.get(q_num):
#                 correct_count += 1

#     score = 0.0 if needs_review else (correct_count / 100) * 10

#     # --- H. VẼ ĐỒ HỌA TRỰC QUAN ---
#     debug_final = warped.copy()
    
#     cv2.rectangle(debug_final, (rect_sbd[0], rect_sbd[1]), (rect_sbd[0]+rect_sbd[2], rect_sbd[1]+rect_sbd[3]), (255, 255, 0), 2)
#     cv2.rectangle(debug_final, (rect_made[0], rect_made[1]), (rect_made[0]+rect_made[2], rect_made[1]+rect_made[3]), (255, 255, 0), 2)

#     # VẼ KHUNG 4 CỘT ĐÁP ÁN (Màu tím)
#     for (bx, by, bw, bh) in col_blocks:
#         cv2.rectangle(debug_final, (bx, by), (bx + bw, by + bh), (255, 0, 255), 3)

#     for q_num, detected_list in final_answers.items():
#         is_correct = (len(detected_list) == 1 and detected_list[0][0] == current_ans_key.get(q_num))
#         for (ans, cx, cy) in detected_list:
#             if is_correct:
#                 cv2.circle(debug_final, (cx, cy), 12, (0, 255, 0), -1)
#             else:
#                 cv2.circle(debug_final, (cx, cy), 12, (0, 0, 255), 3) 
#                 cv2.line(debug_final, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 3)
#                 cv2.line(debug_final, (cx + 10, cy - 10), (cx - 10, cy + 10), (0, 0, 255), 3)

#     box_x, box_y, box_w, box_h = 15, 15, 420, 250
#     cv2.rectangle(debug_final, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1)
#     cv2.rectangle(debug_final, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), 2)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(debug_final, f"SBD  : {sbd}", (30, 60), font, 1.2, (0, 0, 0), 3)
#     cv2.putText(debug_final, f"MA DE: {made}", (30, 110), font, 1.2, (0, 0, 0), 3)

#     error_text = ""
#     if needs_review:
#         error_text = " + ".join(review_reason)
#         cv2.putText(debug_final, f"LOI: {error_text}", (30, 160), font, 1.0, (0, 165, 255), 3)
#         cv2.putText(debug_final, f"DIEM : 0.00 (CHO DUYET)", (30, 220), font, 1.0, (0, 0, 255), 3)
#     else:
#         cv2.putText(debug_final, f"DUNG : {correct_count}/100", (30, 160), font, 1.2, (0, 150, 0), 3)
#         cv2.putText(debug_final, f"DIEM : {score:.2f}", (30, 220), font, 1.3, (0, 0, 255), 3)

#     # --- I. MÃ HÓA ẢNH ĐỂ TRẢ VỀ API ---
#     _, buffer = cv2.imencode('.jpg', debug_final)
#     img_base64 = base64.b64encode(buffer).decode('utf-8')

#     return {
#         "sbd": sbd,
#         "made": made,
#         "score": score,
#         "correct_count": correct_count,
#         "needs_review": needs_review,
#         "error_reason": error_text,
#         "result_image_base64": img_base64
#     }

# @app.post("/api/grade")
# async def grade_exam(
#     image_file: UploadFile = File(...), 
#     answers_json: str = Form(...)
# ):
#     try:
#         contents = await image_file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             return JSONResponse(status_code=400, content={"status": "error", "message": "File ảnh bị hỏng hoặc không đúng định dạng."})

#         all_ans_keys = json.loads(answers_json)
#         result = process_grading_core(img, all_ans_keys)

#         return JSONResponse(status_code=200, content={"status": "success", "data": result})

#     except ValueError as ve:
#         return JSONResponse(status_code=400, content={"status": "error", "message": str(ve)})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"status": "error", "message": f"Lỗi server: {str(e)}"})

# --------------------------------------------------------------------------

import cv2
import numpy as np
from ultralytics import YOLO
import base64
import json
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ==========================================
# 1. CẤU HÌNH HỆ THỐNG & TẢI MODEL
# ==========================================
if not os.path.exists('debug_output'):
    os.makedirs('debug_output')

app = FastAPI(title="Hệ thống chấm thi trắc nghiệm AI")
app.mount("/debug", StaticFiles(directory="debug_output"), name="debug")

MODEL_PATH = 'best.pt'
W_TARGET, H_TARGET = 1000, 1400  
CHOICES = ['A', 'B', 'C', 'D']

try:
    model = YOLO(MODEL_PATH)
    print(f"Đã tải thành công model từ {MODEL_PATH}")
except Exception as e:
    print(f"Lỗi tải model YOLO: {e}")

# ==========================================
# 2. HÀM LÕI XỬ LÝ ẢNH 
# ==========================================
def process_grading_core(img, all_ans_keys):
    # --- A. TIỀN XỬ LÝ & DUỖI ẢNH BẰNG ĐIỂM BIÊN NGOÀI CÙNG ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thres = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_h, img_w = thres.shape

    frame_w = img_w * 0.95
    frame_h = frame_w * 1.4142
    
    if frame_h > img_h * 0.95:
        frame_h = img_h * 0.95
        frame_w = frame_h / 1.4142

    left = int((img_w - frame_w) / 2)
    top = int((img_h - frame_h) / 2)
    
    box_w = int(frame_w * 0.25)
    box_h = int(frame_h * 0.15) 

    roi_tl = thres[top : top+box_h, left : left+box_w]
    roi_tr = thres[top : top+box_h, left + int(frame_w) - box_w : left + int(frame_w)]
    roi_bl = thres[top + int(frame_h) - box_h : top + int(frame_h), left : left+box_w]
    roi_br = thres[top + int(frame_h) - box_h : top + int(frame_h), left + int(frame_w) - box_w : left + int(frame_w)]

    def find_anchor(roi_img, offset_x, offset_y, corner_type):
        cnts, _ = cv2.findContours(roi_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_cnts = []
        roi_area = box_w * box_h
        
        for c in cnts:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or w == 0: continue
            
            ratio = w / float(h)
            extent = area / float(w * h)

            if (40 < area < (roi_area * 0.15)) and (1.0 <= ratio <= 3.8) and (extent > 0.65):
                valid_cnts.append(c)

        if not valid_cnts:
            return None
            
        if corner_type == 'TL':   
            best_c = min(valid_cnts, key=lambda c: cv2.boundingRect(c)[0] + 3 * cv2.boundingRect(c)[1])
        elif corner_type == 'TR': 
            best_c = max(valid_cnts, key=lambda c: cv2.boundingRect(c)[0] - 3 * cv2.boundingRect(c)[1])
        elif corner_type == 'BL': 
            best_c = max(valid_cnts, key=lambda c: 3 * cv2.boundingRect(c)[1] - cv2.boundingRect(c)[0])
        elif corner_type == 'BR': 
            best_c = max(valid_cnts, key=lambda c: cv2.boundingRect(c)[0] + 3 * cv2.boundingRect(c)[1])

        x, y, w, h = cv2.boundingRect(best_c)
        return [x + offset_x, y + offset_y, w, h]

    box_tl = find_anchor(roi_tl, left, top, 'TL')
    box_tr = find_anchor(roi_tr, left + int(frame_w) - box_w, top, 'TR')
    box_bl = find_anchor(roi_bl, left, top + int(frame_h) - box_h, 'BL')
    box_br = find_anchor(roi_br, left + int(frame_w) - box_w, top + int(frame_h) - box_h, 'BR')

    if not box_tl or not box_tr or not box_bl or not box_br:
        raise ValueError("Không tìm thấy đủ 4 vạch biên hợp lệ. Xin hãy chỉnh lại góc máy.")

    pt_tl = [box_tl[0], box_tl[1]]                               
    pt_tr = [box_tr[0] + box_tr[2], box_tr[1]]                   
    pt_bl = [box_bl[0], box_bl[1] + box_bl[3]]                   
    pt_br = [box_br[0] + box_br[2], box_br[1] + box_br[3]]       

    pts = np.array([pt_tl, pt_tr, pt_br, pt_bl], dtype="float32")
    M_mat = cv2.getPerspectiveTransform(pts, np.array([[0, 0], [W_TARGET-1, 0], [W_TARGET-1, H_TARGET-1], [0, H_TARGET-1]], dtype="float32"))
    warped = cv2.warpPerspective(img, M_mat, (W_TARGET, H_TARGET))

    # --- B. ĐÃ XÓA TÌM VẠCH NHỊP BẰNG OPENCV (Vì đã dùng toán học tuyệt đối) ---

    # --- C. TỌA ĐỘ CỨNG CHO 4 KHỐI ĐÁP ÁN ---
    col_blocks = [
        (10,  635, 210, 740),   
        (250, 635, 210, 740),   
        (490, 635, 210, 740),   
        (720, 635, 210, 740)    
    ]

    # --- D. TÌM KHUNG SBD & MÃ ĐỀ ---
    gray_w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, thres_w = cv2.threshold(gray_w, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    info_blocks = []
    cnts_info, _ = cv2.findContours(thres_w[0:500, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts_info:
        x, y, w, h = cv2.boundingRect(c)
        if x > 400 and h > 200 and h > w:
            info_blocks.append((x, y, w, h))

    info_blocks = sorted(info_blocks, key=lambda b: b[0])

    rect_sbd = (490, 110, 140, 305)   
    rect_made = (645, 110, 80, 305)
    
    if len(info_blocks) >= 2:
        rect_sbd = info_blocks[0]
        rect_made = info_blocks[1]

    # --- E. GIẢI MÃ SBD & MÃ ĐỀ BẰNG YOLO ---
    def decode_info_block(rect, num_cols):
        x_start, y_start, w, h = rect
        roi = warped[y_start:y_start+h, x_start:x_start+w]
        results = model.predict(roi, conf=0.45, verbose=False) 
        
        digits = {}
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            col = int(cx / (w / num_cols))
            row = int(cy / (h / 10))
            if 0 <= col < num_cols and 0 <= row < 10:
                if col not in digits:
                    digits[col] = str(row)
                
        return "".join([digits.get(i, "X") for i in range(num_cols)])

    sbd = decode_info_block(rect_sbd, 5)
    made = decode_info_block(rect_made, 3)

    current_ans_key = {}
    needs_review = False
    review_reason = []

    if "X" in sbd: review_reason.append("Thieu SBD")
    if "X" in made: review_reason.append("Thieu Ma de")

    if made in all_ans_keys:
        current_ans_key = {int(k): v for k, v in all_ans_keys[made].items()}
    else:
        if "X" not in made: 
            review_reason.append(f"Chua co dap an ma de {made}")

    if len(review_reason) > 0: needs_review = True

    # --- F. ĐỌC ĐÁP ÁN BÀI LÀM & LƯU ẢNH DEBUG ---
    final_answers = {} 
    for i, (bx, by, bw, bh) in enumerate(col_blocks):
        roi = warped[by:by+bh, bx:bx+bw]
        results = model.predict(roi, conf=0.45, verbose=False) 
        start_q = i * 25 + 1
        
        for res in results:
            debug_yolo_img = roi.copy()
            
            for box, conf in zip(res.boxes.xyxy, res.boxes.conf):
                x1, y1, x2, y2 = map(int, box.cpu().numpy())
                conf_val = float(conf.cpu().numpy())
                cv2.rectangle(debug_yolo_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_yolo_img, f"{conf_val:.2f}", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # CẬP NHẬT LỖI X-AXIS: Căn lề 15% để ôm trọn đáp án A
            start_x = int(bw * 0.15) 
            step_w = (bw * 0.85) / 4
            
            for j in range(5):
                line_x = int(start_x + j * step_w)
                cv2.line(debug_yolo_img, (line_x, 0), (line_x, bh), (255, 0, 0), 2)
            
            cv2.putText(debug_yolo_img, "A", (int(start_x + step_w*0.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(debug_yolo_img, "B", (int(start_x + step_w*1.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(debug_yolo_img, "C", (int(start_x + step_w*2.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.putText(debug_yolo_img, "D", (int(start_x + step_w*3.3), 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            cv2.imwrite(f'debug_output/cot_dap_an_{i+1}.jpg', debug_yolo_img)

            # LOGIC MAP TỌA ĐỘ VÀO ĐÁP ÁN
            for box in res.boxes.xyxy:
                x1, y1, x2, y2 = box.cpu().numpy()
                y_center_roi = (y1 + y2) / 2
                x_center_roi = (x1 + x2) / 2
                
                # CẬP NHẬT LỖI Y-AXIS: Chia đều 25 hàng bằng Toán học (29.6px/hàng)
                row_idx = int(y_center_roi / (bh / 25.0))
                
                # Giới hạn an toàn (Tránh lỗi out-of-bounds)
                if row_idx >= 25: row_idx = 24
                if row_idx < 0: row_idx = 0
                
                q_num = start_q + row_idx
                
                if x_center_roi > start_x:
                    choice_idx = int((x_center_roi - start_x) / step_w)
                    if 0 <= choice_idx < 4:
                        ans = CHOICES[choice_idx]
                        if q_num not in final_answers:
                            final_answers[q_num] = []
                        final_answers[q_num].append((ans, int(x_center_roi + bx), int((y1+y2)/2 + by)))

    # --- G. CHẤM ĐIỂM ---
    correct_count = 0
    for q_num, detected_list in final_answers.items():
        if len(detected_list) == 1:
            if detected_list[0][0] == current_ans_key.get(q_num):
                correct_count += 1

    score = 0.0 if needs_review else (correct_count / 100) * 10

    # --- H. VẼ ĐỒ HỌA TRỰC QUAN ---
    debug_final = warped.copy()
    
    cv2.rectangle(debug_final, (rect_sbd[0], rect_sbd[1]), (rect_sbd[0]+rect_sbd[2], rect_sbd[1]+rect_sbd[3]), (255, 255, 0), 2)
    cv2.rectangle(debug_final, (rect_made[0], rect_made[1]), (rect_made[0]+rect_made[2], rect_made[1]+rect_made[3]), (255, 255, 0), 2)

    for (bx, by, bw, bh) in col_blocks:
        cv2.rectangle(debug_final, (bx, by), (bx + bw, by + bh), (255, 0, 255), 3)

    for q_num, detected_list in final_answers.items():
        is_correct = (len(detected_list) == 1 and detected_list[0][0] == current_ans_key.get(q_num))
        for (ans, cx, cy) in detected_list:
            if is_correct:
                cv2.circle(debug_final, (cx, cy), 12, (0, 255, 0), -1)
            else:
                cv2.circle(debug_final, (cx, cy), 12, (0, 0, 255), 3) 
                cv2.line(debug_final, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 3)
                cv2.line(debug_final, (cx + 10, cy - 10), (cx - 10, cy + 10), (0, 0, 255), 3)

    box_x, box_y, box_w, box_h = 15, 15, 420, 250
    cv2.rectangle(debug_final, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1)
    cv2.rectangle(debug_final, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_final, f"SBD  : {sbd}", (30, 60), font, 1.2, (0, 0, 0), 3)
    cv2.putText(debug_final, f"MA DE: {made}", (30, 110), font, 1.2, (0, 0, 0), 3)

    error_text = ""
    if needs_review:
        error_text = " + ".join(review_reason)
        cv2.putText(debug_final, f"LOI: {error_text}", (30, 160), font, 1.0, (0, 165, 255), 3)
        cv2.putText(debug_final, f"DIEM : 0.00 (CHO DUYET)", (30, 220), font, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(debug_final, f"DUNG : {correct_count}/100", (30, 160), font, 1.2, (0, 150, 0), 3)
        cv2.putText(debug_final, f"DIEM : {score:.2f}", (30, 220), font, 1.3, (0, 0, 255), 3)

    # THÊM ĐOẠN NÀY ĐỂ TRÍCH XUẤT VẾT BÚT CHÌ CỦA HỌC SINH
    student_choices = {}
    for q_num, detected_list in final_answers.items():
        if len(detected_list) == 1:
            student_choices[str(q_num)] = detected_list[0][0]
        elif len(detected_list) > 1:
            student_choices[str(q_num)] = "MULTIPLE" # Lỗi tô đúp

    _, buffer = cv2.imencode('.jpg', debug_final)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    # CẬP NHẬT LẠI KHỐI RETURN NÀY
    return {
        "sbd": sbd,
        "made": made,
        "score": score,
        "correct_count": correct_count,
        "needs_review": needs_review,
        "error_reason": error_text,
        "result_image_base64": img_base64,
        "student_choices": student_choices
    }

# ==========================================
# 3. ENDPOINT API
# ==========================================
@app.post("/api/grade")
async def grade_exam(
    image_file: UploadFile = File(...), 
    answers_json: str = Form(...)
):
    try:
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(status_code=400, content={"status": "error", "message": "File ảnh bị hỏng hoặc không đúng định dạng."})

        all_ans_keys = json.loads(answers_json)
        result = process_grading_core(img, all_ans_keys)

        return JSONResponse(status_code=200, content={"status": "success", "data": result})

    except ValueError as ve:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(ve)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Lỗi server: {str(e)}"})