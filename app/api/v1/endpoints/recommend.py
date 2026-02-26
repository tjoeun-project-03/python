"""
===============================================================================
[추천 시스템 (Recommendation API) 엔드포인트]
- 역할: 기사님의 현재 위치와 차종 정보를 받아, 가장 수익성이 높고 동선이 좋은 
        상위 3개(Top 3)의 화물 오더를 AI를 이용해 추천해 주는 핵심 모듈.
- 핵심 로직:
  1. DB에서 해당 차종이 수행 가능한 '대기 중' 오더 목록 조회
  2. 사전에 학습된 PyTorch ML 모델을 통해 소요 시간 추론
  3. (분당 수익성 50% + 퇴근 거리 30% + 상차 거리 20%) 가중치로 최종 점수 산출
===============================================================================
"""

import os
import math
import torch
import pandas as pd
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
# 2. PyTorch 모델 뼈대 정의 (가중치 로드용)
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
# 3. 모델 전역 로드 및 유틸리티 (서버 최적화)
# ---------------------------------------------------------
router = APIRouter()

# 💡 현재 파일(recommend.py) 위치에서 app/ 폴더까지 4단계 역추적
APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 💡 app/ml_assets/ 폴더 안의 파일들을 바라보도록 명시적 설정
MODEL_PATH = os.path.join(APP_DIR, 'ml_assets', 'best_eta_model.ckpt')
DATA_PATH = os.path.join(APP_DIR, 'ml_assets', 'ml_training_data.csv')

ai_model = None
scaler = None
device = None
car_weight_map = {'1t': 1.0, '1.4t': 1.4, '2.5t': 2.5, '5t': 5.0}
feature_cols = ['dep_lat', 'dep_lng', 'arr_lat', 'arr_lng', 'distance', 'hour', 'dayofweek', 'car_type_num', 'weight']

def load_ai_engine():
    global ai_model, scaler, device
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print(f"⚠️ [경고] AI 모델 또는 데이터 파일이 없어 추천 엔진을 초기화할 수 없습니다.\n(경로 확인: {MODEL_PATH})")
        return

    full_df = pd.read_csv(DATA_PATH)
    full_df['car_type_num'] = full_df['car_type'].map(car_weight_map)
    scaler = StandardScaler()
    scaler.fit(full_df[feature_cols].values)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ai_model = ETAPredictor.load_from_checkpoint(MODEL_PATH)
    ai_model = ai_model.to(device)
    ai_model.eval()
    print("✅ [AI 엔진 준비 완료] ML 모델과 스케일러가 메모리에 정상 로드되었습니다.")

load_ai_engine()

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# ---------------------------------------------------------
# 4. 핵심 API 엔드포인트
# ---------------------------------------------------------
@router.post("/top3", response_model=RecommendationResponse)
async def get_top3_orders(req: DriverStatusRequest):
    if ai_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="AI 추천 엔진이 오프라인 상태입니다.")

    # =========================================================================
    # [TODO: 🚨 오라클(Oracle) DB 연동 시 수정해야 할 블록]
    # 프론트엔드 연동이 끝나고 실제 DB가 구축되면, 아래 CSV 로드 코드를 지우고 
    # Oracle DB에서 데이터를 가져오도록 수정해야 합니다.
    # 
    # [추천 구현 방식 (oracledb + pandas)]
    # import oracledb
    # connection = oracledb.connect(user="계정", password="비번", dsn="호스트:포트/서비스명")
    # query = f"SELECT * FROM orders WHERE status = '대기중' AND car_type = '{req.car_type}'"
    # df = pd.read_sql(query, con=connection)
    # connection.close()
    # =========================================================================
    
    # (현재) 프론트엔드 테스트를 위한 임시 CSV 데이터 로드
    df = pd.read_csv(DATA_PATH)
    df = df[df['car_type'] == req.car_type].copy()

    # DB에 조건에 맞는 오더가 하나도 없을 경우의 예외 처리
    if df.empty:
        return RecommendationResponse(message="현재 수행 가능한 대기 오더가 없습니다.", data=[])

    # -------------------------------------------------------------------------
    # 이후 로직은 Oracle DB에서 가져온 df(DataFrame) 형식이 동일하다면 수정할 필요 없음!
    # -------------------------------------------------------------------------

    df['car_type_num'] = df['car_type'].map(car_weight_map)
    X_scaled = scaler.transform(df[feature_cols].values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        df['predicted_eta'] = ai_model(X_tensor).cpu().numpy().flatten()

    df['total_price'] = df['profit_per_min'] * df['duration']
    df['profit_score'] = df['total_price'] / df['predicted_eta']
    df['pickup_dist'] = df.apply(lambda r: get_distance(req.current_lat, req.current_lng, r['dep_lat'], r['dep_lng']), axis=1)
    df['return_dist'] = df.apply(lambda r: get_distance(req.home_lat, req.home_lng, r['arr_lat'], r['arr_lng']), axis=1)

    safe_norm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 1.0
    df['profit_norm'] = safe_norm(df['profit_score'])
    df['pickup_norm'] = 1 - safe_norm(df['pickup_dist'])
    df['return_norm'] = 1 - safe_norm(df['return_dist'])
    
    df['final_score'] = (df['profit_norm'] * 0.5) + (df['return_norm'] * 0.3) + (df['pickup_norm'] * 0.2)

    top3 = df.sort_values(by='final_score', ascending=False).head(3)
    
    result_list = []
    for rank, (_, row) in enumerate(top3.iterrows(), 1):
        result_list.append(
            RecommendedOrder(
                rank=rank,
                final_score=round(row['final_score'] * 100, 1),
                total_price=int(row['total_price']),
                predicted_eta=int(row['predicted_eta']),
                pickup_dist=round(row['pickup_dist'], 1),
                return_dist=round(row['return_dist'], 1),
                dep_lat=row['dep_lat'],
                dep_lng=row['dep_lng'],
                arr_lat=row['arr_lat'],
                arr_lng=row['arr_lng']
            )
        )

    return RecommendationResponse(message="성공적으로 추천 오더를 불러왔습니다.", data=result_list)