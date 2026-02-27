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

# --- 스케줄링 전용 DTO ---
class ScheduleRequest(BaseModel):
    current_lat: float
    current_lng: float
    home_lat: float
    home_lng: float
    car_type: str
    max_work_hours: int = 8  # 기사님의 하루 희망 근무 시간 (기본 8시간)

class RouteStep(BaseModel):
    step_type: str  # "EMPTY_RETURN"(공차이동) 또는 "DELIVERY"(화물운송 - 상하차시간 포함)
    order_id: int = 0
    duration_min: int
    distance_km: float
    profit: int = 0
    start_lat: float
    start_lng: float
    end_lat: float
    end_lng: float

class DailyScheduleOption(BaseModel):
    rank: int
    total_profit: int
    total_work_time_min: int
    efficiency_score: float # 시간당 수익률
    route_details: List[RouteStep]

class ScheduleResponse(BaseModel):
    message: str
    data: List[DailyScheduleOption]

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

APP_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_PATH = os.path.join(APP_DIR, 'ml_assets', 'best_eta_model.ckpt')
DATA_PATH = os.path.join(APP_DIR, 'ml_assets', 'ml_training_data.csv')

ai_model = None
scaler = None
device = None
car_weight_map = {'1t': 1.0, '1.4t': 1.4, '2.5t': 2.5, '5t': 5.0}
feature_cols = ['dep_lat', 'dep_lng', 'arr_lat', 'arr_lng', 'distance', 'hour', 'dayofweek', 'car_type_num', 'weight']

# 💡 [핵심] 차종별 평균 상하차 대기시간 (단위: 분)
handling_time_map = {
    '1t': 40,   # 수작업(까대기) 및 소형 화물 상하차
    '1.4t': 50,
    '2.5t': 70, # 파렛트 및 중형 화물 지게차 대기
    '5t': 90    # 대형 화물, 호루/윙바디 세팅 및 지게차 대기
}

def load_ai_engine():
    global ai_model, scaler, device
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        print(f"⚠️ [경고] AI 모델/데이터가 없어 추천 엔진을 초기화할 수 없습니다.")
        return

    full_df = pd.read_csv(DATA_PATH)
    full_df['car_type_num'] = full_df['car_type'].map(car_weight_map)
    scaler = StandardScaler()
    scaler.fit(full_df[feature_cols].values)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ai_model = ETAPredictor.load_from_checkpoint(MODEL_PATH)
    ai_model = ai_model.to(device)
    ai_model.eval()
    print("✅ [AI 엔진 준비 완료] ML 모델이 메모리에 정상 로드되었습니다.")

load_ai_engine()

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# ---------------------------------------------------------
# 4. 엔드포인트 1: 단일 오더 Top 3 추천 API
# ---------------------------------------------------------
@router.post("/top3", response_model=RecommendationResponse)
async def get_top3_orders(req: DriverStatusRequest):
    if ai_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="AI 엔진 오프라인")

    df = pd.read_csv(DATA_PATH)
    df = df[df['car_type'] == req.car_type].copy()

    if df.empty:
        return RecommendationResponse(message="수행 가능한 오더가 없습니다.", data=[])

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
        result_list.append(RecommendedOrder(
            rank=rank, final_score=round(row['final_score'] * 100, 1),
            total_price=int(row['total_price']), predicted_eta=int(row['predicted_eta']),
            pickup_dist=round(row['pickup_dist'], 1), return_dist=round(row['return_dist'], 1),
            dep_lat=row['dep_lat'], dep_lng=row['dep_lng'],
            arr_lat=row['arr_lat'], arr_lng=row['arr_lng']
        ))

    return RecommendationResponse(message="성공적으로 추천 오더를 불러왔습니다.", data=result_list)

# ---------------------------------------------------------
# 5. 엔드포인트 2: 일일 전체 코스 스케줄링 API (완전 현실 반영 버전)
# ---------------------------------------------------------
@router.post("/daily-schedule", response_model=ScheduleResponse)
async def get_daily_schedule(req: ScheduleRequest):
    if ai_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="AI 엔진 오프라인")

    df = pd.read_csv(DATA_PATH)
    df = df[df['car_type'] == req.car_type].copy()

    if df.empty:
        return ScheduleResponse(message="오늘 수행 가능한 오더가 없습니다.", data=[])

    df['car_type_num'] = df['car_type'].map(car_weight_map)
    max_time_min = req.max_work_hours * 60
    
    # 💡 해당 차종의 상하차 대기시간 세팅 (기본 60분)
    handling_time = handling_time_map.get(req.car_type, 60)

    # AI 추론 1회 일괄 수행 (운전 소요 시간 계산)
    X_scaled = scaler.transform(df[feature_cols].values)
    with torch.no_grad():
        df['base_eta'] = ai_model(torch.tensor(X_scaled, dtype=torch.float32).to(device)).cpu().numpy().flatten()
    df['total_price'] = df['profit_per_min'] * df['duration']

    # 1. 1차 콜 선정: '공차이동 + 상하차시간 + 운전시간' 대비 수익성(효율성) 탐색
    df['dist_from_current'] = df.apply(lambda r: get_distance(req.current_lat, req.current_lng, r['dep_lat'], r['dep_lng']), axis=1)
    df['empty_time_to_start'] = df['dist_from_current'] / 1.5
    
    # 💡 상하차 시간(handling_time)을 포함하여 현실적인 시간당 수익 계산
    df['start_efficiency'] = df['total_price'] / (df['empty_time_to_start'] + handling_time + df['base_eta'])
    
    first_candidates = df.sort_values(by='start_efficiency', ascending=False).head(5)
    best_schedules = []

    # 2. 탑 5개의 시작점 각각에 대해 하루 일정을 무한 루프로 시뮬레이션
    for _, start_order in first_candidates.iterrows():
        current_time_spent = 0
        current_profit = 0
        current_lat, current_lng = req.current_lat, req.current_lng
        
        route_steps = []
        available_orders = df.copy()
        
        # 첫 번째 오더 수행 처리
        o1_empty_time = start_order['dist_from_current'] / 1.5
        o1_time = start_order['base_eta']
        
        # 💡 누적 시간에 상하차 시간 포함
        current_time_spent += (o1_empty_time + handling_time + o1_time)
        current_profit += start_order['total_price']
        current_lat, current_lng = start_order['arr_lat'], start_order['arr_lng']
        
        route_steps.append(RouteStep(step_type="EMPTY_RETURN", duration_min=int(o1_empty_time), distance_km=round(start_order['dist_from_current'],1), start_lat=req.current_lat, start_lng=req.current_lng, end_lat=start_order['dep_lat'], end_lng=start_order['dep_lng']))
        
        # 💡 DELIVERY 스텝의 소요 시간에 상하차 시간(handling_time) 포함하여 앱에 노출
        route_steps.append(RouteStep(step_type="DELIVERY", order_id=int(start_order.get('order_id', 0)), duration_min=int(o1_time + handling_time), distance_km=round(start_order['distance'],1), profit=int(start_order['total_price']), start_lat=start_order['dep_lat'], start_lng=start_order['dep_lng'], end_lat=start_order['arr_lat'], end_lng=start_order['arr_lng']))
        
        available_orders = available_orders.drop(start_order.name)

        # 3. 무한 루프(While): 8시간이 차거나 남은 오더가 없을 때까지 꼬리물기
        while not available_orders.empty:
            dist_to_home = get_distance(current_lat, current_lng, req.home_lat, req.home_lng)
            home_empty_time = dist_to_home / 1.5
            
            # 최종 퇴근 시간을 고려하여 8시간 초과 시 루프 탈출
            if current_time_spent + home_empty_time >= max_time_min:
                break

            available_orders['dist_to_pickup'] = available_orders.apply(lambda r: get_distance(current_lat, current_lng, r['dep_lat'], r['dep_lng']), axis=1)
            available_orders['dist_to_home_after'] = available_orders.apply(lambda r: get_distance(r['arr_lat'], r['arr_lng'], req.home_lat, req.home_lng), axis=1)
            
            safe_norm = lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 1.0
            
            # 💡 다중 스코어링: 현실적인 소요 시간(base_eta + handling_time)으로 점수 재계산
            profit_score = safe_norm(available_orders['total_price'] / (available_orders['base_eta'] + handling_time))
            pickup_score = 1 - safe_norm(available_orders['dist_to_pickup'])
            return_score = 1 - safe_norm(available_orders['dist_to_home_after'])
            
            available_orders['next_score'] = (profit_score * 0.5) + (pickup_score * 0.2) + (return_score * 0.3)
            
            best_next = available_orders.sort_values(by='next_score', ascending=False).iloc[0]
            
            next_empty_time = best_next['dist_to_pickup'] / 1.5
            next_time = best_next['base_eta']
            next_home_time = best_next['dist_to_home_after'] / 1.5
            
            # 💡 다음 오더를 수행하고 퇴근할 때 8시간을 넘기면 포기하고 집으로 이동
            if current_time_spent + next_empty_time + handling_time + next_time + next_home_time > max_time_min:
                break
                
            # 누적 시간 및 수익 업데이트
            current_time_spent += (next_empty_time + handling_time + next_time)
            current_profit += best_next['total_price']
            
            route_steps.append(RouteStep(step_type="EMPTY_RETURN", duration_min=int(next_empty_time), distance_km=round(best_next['dist_to_pickup'],1), start_lat=current_lat, start_lng=current_lng, end_lat=best_next['dep_lat'], end_lng=best_next['dep_lng']))
            route_steps.append(RouteStep(step_type="DELIVERY", order_id=int(best_next.get('order_id', 0)), duration_min=int(next_time + handling_time), distance_km=round(best_next['distance'],1), profit=int(best_next['total_price']), start_lat=best_next['dep_lat'], start_lng=best_next['dep_lng'], end_lat=best_next['arr_lat'], end_lng=best_next['arr_lng']))
            
            current_lat, current_lng = best_next['arr_lat'], best_next['arr_lng']
            available_orders = available_orders.drop(best_next.name)

        # 4. 하루 일과 종료: 남은 시간 여유롭게 집으로 최종 퇴근
        final_dist_to_home = get_distance(current_lat, current_lng, req.home_lat, req.home_lng)
        final_home_time = final_dist_to_home / 1.5
        current_time_spent += final_home_time
        route_steps.append(RouteStep(step_type="EMPTY_RETURN", duration_min=int(final_home_time), distance_km=round(final_dist_to_home,1), start_lat=current_lat, start_lng=current_lng, end_lat=req.home_lat, end_lng=req.home_lng))

        efficiency = current_profit / current_time_spent if current_time_spent > 0 else 0
        best_schedules.append({
            "total_profit": int(current_profit),
            "total_work_time_min": int(current_time_spent),
            "efficiency_score": efficiency,
            "route_details": route_steps
        })

    best_schedules = sorted(best_schedules, key=lambda x: x['efficiency_score'], reverse=True)[:3]

    results = []
    for rank, sched in enumerate(best_schedules, 1):
        results.append(DailyScheduleOption(
            rank=rank,
            total_profit=sched['total_profit'],
            total_work_time_min=sched['total_work_time_min'],
            efficiency_score=round(sched['efficiency_score'], 1),
            route_details=sched['route_details']
        ))

    return ScheduleResponse(message="다중 스코어링 및 상하차 대기시간 반영 최적 코스 구성 완료", data=results)