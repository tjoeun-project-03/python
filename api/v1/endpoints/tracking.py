# 파일명: app/api/v1/endpoints/tracking.py
# 역할: 실시간 위치 공유를 위한 WebSocket 엔드포인트

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.modules.connection_manager import manager

router = APIRouter()

# 주소 예시: ws://127.0.0.1:8000/api/v1/tracking/ws/{order_id}
@router.websocket("/ws/{order_id}")
async def websocket_endpoint(websocket: WebSocket, order_id: str):
    """
    [실시간 위치 공유 소켓]
    - 차주: 이 소켓으로 본인의 위도/경도를 계속 보냄 (Send)
    - 화주: 이 소켓을 켜두고 차주의 위치를 실시간으로 받음 (Receive)
    """
    # 1. 연결 수락 (방 입장)
    await manager.connect(websocket, order_id)
    
    try:
        while True:
            # 2. 데이터 수신 (Client -> Server)
            # 예: {"lat": 37.5, "lng": 127.0, "role": "driver"}
            data = await websocket.receive_json()
            
            # 3. 같은 방에 있는 사람들에게 그대로 뿌림 (Server -> Clients)
            # "방금 들어온 위치 정보를 이 방의 모두에게 전파하라"
            await manager.broadcast(data, order_id)
            
    except WebSocketDisconnect:
        # 4. 연결 끊김 처리
        manager.disconnect(websocket, order_id)
        # (선택) "상대방이 나갔습니다" 메시지 전송 가능