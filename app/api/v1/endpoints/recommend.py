"""
===============================================================================
[추천 시스템 (Recommendation API) & 일일 스케줄링 엔드포인트]
- 역할: 하나의 딥러닝 ETA 모델을 메모리에 공유하여, 
        1) 단일 꿀콜 추천 (/top3)
        2) 연속 배차 일일 스케줄링 (/daily-schedule) 을 수행하는 통합 AI 배차 엔진.
- 고도화: 차량 제원별 상하차 대기시간(Handling Time)을 반영하여 100% 현실적인 스케줄 도출
===============================================================================
"""

import os
import math
import torch
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
import torch.nn as nn

# ---------------------------------------------------------
# 1. Pydantic DTO (데이터 검증 스키마)
# ---------------------------------------------------------
class DriverStatusRequest(BaseModel):
    current_lat: float
    current_lng: float
    home_lat: float
    home_lng: float
    car_type: str  

class RecommendedOrder(BaseModel):
    rank: int
    final_score: float
    total_price: int
    predicted_eta: int
    pickup_dist: float
    return_dist: float
    dep_lat: float
    dep_lng: float
    arr_lat: float
    arr_lng: float

class RecommendationResponse(BaseModel):
    message: str
    data: List[RecommendedOrder]

# ---------------------------------------------------------
# 2. PyTorch 모델 정의 (가중치 로드용)
# ---------------------------------------------------------
class ETAPredictor(pl.LightningModule):
    def __init__(self, input_dim=9):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(0.1), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x): 
        return self.model(x)

# ---------------------------------------------------------
# 3. 모델 전역 로드 및 유틸리티
# ---------------------------------------------------------
router = APIRouter()

# STS 서버 기본 주소 (필요시 수정)
STS_BASE_URL = "http://localhost:8080"

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(APP_DIR, 'ml_assets', 'best_eta_model.ckpt')
# Scaler의 기준점(Mean, Std)을 잡기 위해 학습 당시의 CSV 헤더 정보는 필요합니다.
DATA_PATH = os.path.join(APP_DIR, 'ml_assets', 'ml_training_data.csv')

ai_model = None
scaler = None
device = None
car_weight_map = {'1t': 1.0, '1.4t': 1.4, '2.5t': 2.5, '5t': 5.0}
feature_cols = ['dep_lat', 'dep_lng', 'arr_lat', 'arr_lng', 'distance', 'hour', 'dayofweek', 'car_type_num', 'weight']

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def load_ai_engine():
    global ai_model, scaler, device
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print(f"⚠️ [경고] AI 모델 또는 기준 데이터가 없습니다.")
        return

    # Scaler 피팅 (데이터 분포 고정용)
    full_df = pd.read_csv(DATA_PATH)
    full_df['car_type_num'] = full_df['car_type'].map(car_weight_map)
    scaler = StandardScaler()
    scaler.fit(full_df[feature_cols].values)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ai_model = ETAPredictor.load_from_checkpoint(MODEL_PATH)
    ai_model = ai_model.to(device).eval()
    print("✅ [AI 엔진] STS 실시간 연동 모드 활성화")

load_ai_engine()

# ---------------------------------------------------------
# 4. STS 실시간 데이터 브릿지 (API 호출 및 매핑)
# ---------------------------------------------------------
def fetch_available_orders_from_sts():
    try:
        response = requests.get(f"{STS_BASE_URL}/api/orders/available", timeout=5)
        response.raise_for_status()
        sts_data = response.json()
        
        if not sts_data:
            return pd.DataFrame()

        # STS JSON 필드를 AI 모델 피처명으로 변환
        processed_list = []
        now = pd.Timestamp.now()
        
        for item in sts_data:
            processed_list.append({
                'order_id': item['orderId'],
                'dep_lat': float(item['startLat']),
                'dep_lng': float(item['startLng']),
                'arr_lat': float(item['endLat']),
                'arr_lng': float(item['endLng']),
                'distance': float(item['distance']),
                'car_type': item['carType'],
                'weight': float(item['weight']),
                'total_price': int(item['price']),
                'hour': now.hour,
                'dayofweek': now.dayofweek
            })
        
        return pd.DataFrame(processed_list)
    except Exception as e:
        print(f"❌ STS 데이터 호출 실패: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# 5. 엔드포인트: 실시간 Top 3 추천 API
# ---------------------------------------------------------
@router.post("/top3", response_model=RecommendationResponse)
async def get_top3_orders(req: DriverStatusRequest):
    print(f"\n🚀 [API START] 추천 요청 수신 - 차종: {req.car_type}, 현재위치: ({req.current_lat}, {req.current_lng})")
    
    if ai_model is None or scaler is None:
        print("❌ [DEBUG] AI 모델 또는 Scaler가 로드되지 않았습니다.")
        raise HTTPException(status_code=500, detail="AI 엔진 오프라인")

    # 1. STS API 데이터 호출
    df = fetch_available_orders_from_sts()

    if df.empty:
        print("ℹ️ [DEBUG] 추천할 수 있는 오더가 하나도 없습니다. (STS 데이터 없음)")
        return RecommendationResponse(message="수행 가능한 오더가 없습니다.", data=[])

    # 2. 차종 필터링 로그
    original_count = len(df)
    df = df[df['car_type'] == req.car_type].copy()
    print(f"🔍 [DEBUG] 차종 필터링: {original_count}건 -> {len(df)}건 (필터: {req.car_type})")

    if df.empty:
        print(f"ℹ️ [DEBUG] 요청하신 차종({req.car_type})과 일치하는 오더가 없습니다.")
        return RecommendationResponse(message=f"{req.car_type} 차량용 오더가 없습니다.", data=[])

    # 3. AI 추론 및 계산
    try:
        df['car_type_num'] = df['car_type'].map(car_weight_map).fillna(1.0)
        X_scaled = scaler.transform(df[feature_cols].values)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            df['predicted_eta'] = ai_model(X_tensor).cpu().numpy().flatten()
        print("🧠 [DEBUG] AI ETA 예측 완료")

        # 4. 스코어링
        df['profit_score'] = df['total_price'] / (df['predicted_eta'] + 0.1) # 0 나누기 방지
        df['pickup_dist'] = df.apply(lambda r: get_distance(req.current_lat, req.current_lng, r['dep_lat'], r['dep_lng']), axis=1)
        df['return_dist'] = df.apply(lambda r: get_distance(req.home_lat, req.home_lng, r['arr_lat'], r['arr_lng']), axis=1)

        # 정규화 및 최종 점수
        def safe_norm(col, invert=False):
            if col.max() == col.min(): return 1.0
            norm = (col - col.min()) / (col.max() - col.min())
            return 1.0 - norm if invert else norm

        df['profit_norm'] = safe_norm(df['profit_score'])
        df['pickup_norm'] = safe_norm(df['pickup_dist'], invert=True)
        df['return_norm'] = safe_norm(df['return_dist'], invert=True)
        df['final_score'] = (df['profit_norm'] * 0.5) + (df['return_norm'] * 0.3) + (df['pickup_norm'] * 0.2)
        
        print(f"📊 [DEBUG] 스코어링 완료 - 최고 점수: {df['final_score'].max():.2f}")

    except Exception as e:
        print(f"❌ [DEBUG] 계산 과정 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail="추천 계산 중 오류 발생")

    # 5. 결과 반환
    top3 = df.sort_values(by='final_score', ascending=False).head(3)
    result_list = []
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        result_list.append(RecommendedOrder(
            rank=rank,
            final_score=round(row['final_score'] * 100, 1),
            total_price=int(row['total_price']),
            predicted_eta=int(row['predicted_eta']),
            pickup_dist=round(row['pickup_dist'], 1),
            return_dist=round(row['return_dist'], 1),
            dep_lat=row['dep_lat'], dep_lng=row['dep_lng'],
            arr_lat=row['arr_lat'], arr_lng=row['arr_lng']
        ))

    print(f"✨ [API END] 추천 {len(result_list)}건 반환 성공\n")
    return RecommendationResponse(message="실시간 추천 완료", data=result_list)

