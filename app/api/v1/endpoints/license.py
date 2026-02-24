# 파일명: python/app/api/v1/endpoints/license.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.modules.license_api import verify_driver_license

router = APIRouter()

@router.post("/verify")
async def verify_license(
    file: UploadFile = File(...),
    transport_type: str = Form("2")
):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")

    try:
        # 파일 내용을 바이트(Bytes) 형태로 메모리에 바로 읽어옵니다.
        image_bytes = await file.read()

        # 무거운 OCR 연산이 서버를 멈추게 하지 않도록 백그라운드 스레드에서 실행합니다.
        result = await run_in_threadpool(verify_driver_license, image_bytes, transport_type)

        if result and result.get("success"):
            return {"status": "success", "message": "자격증 검증 완료", "data": result}
        else:
            return {"status": "fail", "message": result.get("message", "검증 실패"), "data": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 처리 중 오류 발생: {str(e)}")