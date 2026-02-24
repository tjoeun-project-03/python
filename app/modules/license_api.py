# 파일명: python/app/modules/license_api.py
import os
from dotenv import load_dotenv
import requests
from .ocr_verify import extract_info_from_license

load_dotenv()
SERVICE_KEY = os.getenv("KOTSA_SERVICE_KEY")
BASE_URL = "http://apis.data.go.kr/B553881/lcnsCheckService/lcnsCheck"

def call_check_api(name, birth, lcns_no, transport_type):
    clean_birth = "".join(filter(str.isdigit, birth))
    if len(clean_birth) == 8: clean_birth = clean_birth[2:] 
    clean_lcns_no = "".join(filter(str.isdigit, lcns_no))
    
    params = {
        "serviceKey": SERVICE_KEY,
        "returnType": "json",
        "name": name,
        "birth": clean_birth,
        "lcnsNo": clean_lcns_no,
        "type": transport_type
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        if response.status_code == 200: return response.json()
        return None
    except Exception as e:
        print(f"API 요청 오류: {e}")
        return None

def verify_driver_license(image_input, transport_type, fixed_type=None): # <- 이름만 변경
    if fixed_type is not None and str(transport_type) != str(fixed_type):
        return {"success": False, "message": f"요청하신 운송 타입({transport_type})은 허용된 타입이 아닙니다."}

    extracted = extract_info_from_license(image_input)
    if not extracted or not all([extracted['이름'], extracted['생년월일'], extracted['자격증번호']]):
        return {"success": False, "message": "OCR 정보 추출 실패"}

    api_response = call_check_api(extracted["이름"], extracted["생년월일"], extracted["자격증번호"], transport_type)

    if api_response:
        res_root = api_response.get("response", {})
        header = res_root.get("header", {})
        body = res_root.get("body", {})
        
        if header.get("resultCode") == "00":
            item_data = body.get("items", {}).get("item", {})
            item = item_data[0] if isinstance(item_data, list) and len(item_data) > 0 else item_data
            
            status_code = item.get("status")
            status_map = {"O": "정상(유효)", "X": "불일치 혹은 자격취소", "△": "자격재취득대상"}
            
            return {
                "success": True,
                "info": extracted,
                "status": status_map.get(status_code, f"알 수 없는 상태({status_code})"),
                "time": item.get("time")
            }
        else:
            return {"success": False, "message": header.get("resultMsg")}
    
    return {"success": False, "message": "API 통신 실패"}