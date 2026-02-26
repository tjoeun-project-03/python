# 파일명: app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # [추가] 보안 설정 모듈
from app.api.v1.endpoints import orders, tracking, license, recommend
from app.core.config import TMAP_API_KEY

app = FastAPI()

# [추가] 브라우저 접속 허용 (CORS) 설정
# 이 코드가 없으면 웹사이트에서 접속할 때 'Connection closed'가 뜰 수 있습니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 사이트에서 접속 허용 (보안상 *는 개발용)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(orders.router, prefix="/api/v1/orders", tags=["Orders"])
app.include_router(tracking.router, prefix="/api/v1/tracking", tags=["Tracking"])
app.include_router(license.router, prefix="/api/v1/license", tags=["License Verification"])
app.include_router(recommend.router, prefix="/api/v1/recommendations", tags=["Recommendations"])

if not TMAP_API_KEY:
    raise ValueError(".env 파일에 TMAP_API_KEY가 없습니다.")

@app.get("/")
def read_root():
    return {"message": "화물 중개 플랫폼 API 서버 (Orders + Tracking)"}


