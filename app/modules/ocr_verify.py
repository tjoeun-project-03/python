# 파일명: python/app/modules/ocr_verify.py
import re
import easyocr
from .object_recognition import get_license_crop

reader = easyocr.Reader(['ko', 'en'])

def group_text_by_rows(ocr_results, threshold=15):
    items = []
    for (bbox, text, prob) in ocr_results:
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        x_start = bbox[0][0]
        items.append({'y': y_center, 'x': x_start, 'text': text})

    items.sort(key=lambda item: item['y'])

    rows = []
    if not items: return rows

    current_row = [items[0]]
    for i in range(1, len(items)):
        if abs(items[i]['y'] - current_row[-1]['y']) < threshold:
            current_row.append(items[i])
        else:
            current_row.sort(key=lambda item: item['x'])
            rows.append("".join([item['text'] for item in current_row]))
            current_row = [items[i]]
    
    current_row.sort(key=lambda item: item['x'])
    rows.append("".join([item['text'] for item in current_row]))
    return rows

def extract_info_from_license(image_input): # <- 이름만 변경
    cropped_img = get_license_crop(image_input)
    if cropped_img is None: return None

    print("OCR 글자 추출 시작...")
    raw_results = reader.readtext(cropped_img, detail=1)
    rows = group_text_by_rows(raw_results)
    full_text = "".join([res[1] for res in raw_results]).replace(" ", "")
    
    extracted_data = {"이름": None, "생년월일": None, "자격증번호": None, "자격취득일": None}

    for row in rows:
        clean_row = row.replace(" ", "")
        if not extracted_data["이름"]:
            name_match = re.search(r'명[:\s]*([가-힣]{2,4})', clean_row)
            if name_match: extracted_data["이름"] = name_match.group(1)

    no_match = re.search(r'(\d{1,2}[-\.~_]\d{2}[-\.~_]\d{6})', full_text)
    if no_match: extracted_data["자격증번호"] = no_match.group(1)

    date_pattern = r'((19|20)\d{2})[\.년\s]?(\d{2})[\.월\s]?(\d{2})'
    all_dates = re.findall(date_pattern, full_text)
    
    if all_dates:
        first, last = all_dates[0], all_dates[-1]
        extracted_data["생년월일"] = f"{first[0]}.{first[2]}.{first[3]}"
        if len(all_dates) > 1:
            extracted_data["자격취득일"] = f"{last[0]}.{last[2]}.{last[3]}"

    return extracted_data