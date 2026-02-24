# 파일명: python/app/modules/object_recognition.py
import os
from dotenv import load_dotenv
import base64
import requests
import cv2
import numpy as np

load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "cju-ymfwh"
WORKFLOW_ID = "find-cdls"

def get_license_crop(image_input):
    # 1. 메모리 데이터(Bytes)인지, 로컬 파일 경로(String)인지 구분하여 처리
    if isinstance(image_input, bytes):
        encoded_string = base64.b64encode(image_input).decode("utf-8")
        img_array = np.frombuffer(image_input, np.uint8)
    else:
        with open(image_input, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        img_array = np.fromfile(image_input, np.uint8)

    url = f"https://serverless.roboflow.com/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    payload = {
        "api_key": API_KEY,
        "inputs": {"image": {"type": "base64", "value": encoded_string}}
    }

    print("Roboflow 서버에 분석 요청 중...")
    try:
        response = requests.post(url, json=payload, timeout=20)
        if response.status_code != 200:
            print(f"API 에러: {response.text}")
            return None

        result = response.json()
        outputs = result.get("outputs", [])
        if not outputs: return None
        
        predictions = outputs[0].get("predictions", {}).get("predictions", [])
        if not predictions:
            print("자격증을 찾지 못했습니다.")
            return None

        target = predictions[0]
        x_center, y_center = target["x"], target["y"]
        width, height = target["width"], target["height"]

        # 2. 하드디스크 저장 없이 메모리 위에서 바로 자르기
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        x1 = max(0, int(x_center - width/2))
        y1 = max(0, int(y_center - height/2))
        x2 = int(x_center + width/2)
        y2 = int(y_center + height/2)

        cropped_img = img[y1:y2, x1:x2]
        
        return cropped_img

    except Exception as e:
        print(f"이미지 분석 중 오류 발생: {e}")
        return None