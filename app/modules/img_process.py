import cv2
import numpy as np
import utils
import os

########################################################################
webCamFeed = False  # True: 웹캠 사용, False: 이미지 파일 사용
pathImage = "/Users/ljw/Desktop/tjoeun-project-03/python/app/SE-674332f8-1eed-4d6b-ae2a-c3d69f5dfacc.jpg" # 테스트할 이미지 파일명 (같은 폴더에 있어야 함)
cap = cv2.VideoCapture(0)
cap.set(10, 160)

# 감지(Detection)용 저해상도 크기 (속도 최적화용)
heightImg = 640
widthImg = 480
########################################################################

utils.initializeTrackbars()
count = 0

# 디버그 이미지 저장 경로 생성
save_dir = "Debug_Images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

while True:
    if webCamFeed:
        success, img = cap.read()
    else:
        img = cv2.imread(pathImage)
        if img is None:
            print(f"오류: '{pathImage}' 파일을 찾을 수 없습니다.")
            break

    # 1. 원본 이미지 백업 (나중에 여기서 고화질로 잘라내기 위함)
    imgOriginal = img.copy()

    # 2. 감지 속도를 위해 이미지 리사이즈 (480x640)
    img = cv2.resize(img, (widthImg, heightImg))
    
    # 3. 전처리 (Gray -> Blur -> Canny -> Dilate/Erode)
    imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    
    # 트랙바 값 가져오기 (필요 시 조절)
    thres = utils.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
    
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    ## 윤곽선 그리기용 복사본
    imgContours = img.copy()
    imgBigContour = img.copy()
    
    # 4. 윤곽선 찾기
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)

    # 5. 가장 큰 사각형(자격증) 찾기
    biggest, maxArea = utils.biggestContour(contours)
    
    if biggest.size != 0:
        biggest = utils.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20)
        imgBigContour = utils.drawRectangle(imgBigContour, biggest, 2)
        
        # =================================================================
        # [핵심 로직] 원본 좌표로 변환하여 고화질 크롭 (비율 보정 포함)
        # =================================================================
        
        # 리사이즈 비율 계산
        scaleX = imgOriginal.shape[1] / widthImg
        scaleY = imgOriginal.shape[0] / heightImg
        
        # 감지된 좌표를 원본 이미지 스케일로 변환
        biggestOriginal = biggest.copy()
        biggestOriginal[:, :, 0] = biggest[:, :, 0] * scaleX
        biggestOriginal[:, :, 1] = biggest[:, :, 1] * scaleY

        # 자격증(신용카드) 비율 설정 (가로형: 약 860x540)
        widthCard = 860
        heightCard = 540
        
        pts1 = np.float32(biggestOriginal)
        pts2 = np.float32([[0, 0], [widthCard, 0], [0, heightCard], [widthCard, heightCard]])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        
        # 원본 이미지(imgOriginal)에서 변환 수행
        imgWarpColored = cv2.warpPerspective(imgOriginal, matrix, (widthCard, heightCard))

        # 가장자리 노이즈 제거 (상하좌우 15픽셀씩 잘라냄)
        crop_val = 15
        imgWarpColored = imgWarpColored[crop_val:heightCard-crop_val, crop_val:widthCard-crop_val]
        
        # [저장 로직] 처리된 결과물을 파일로 저장
        cv2.imwrite(f"{save_dir}/1_threshold.jpg", imgThreshold)
        cv2.imwrite(f"{save_dir}/2_final_result.jpg", imgWarpColored)
        
        # =================================================================
        
        # 화면 표시를 위해 다시 줄이기 (결과 확인용)
        imgWarpDisplay = cv2.resize(imgWarpColored, (widthImg, heightImg))

        # Adaptive Threshold (확인용)
        imgWarpGray = cv2.cvtColor(imgWarpDisplay, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBigContour, imgWarpDisplay, imgWarpGray, imgAdaptiveThre])
        
        print("✅ 자격증 감지 성공! Debug_Images 폴더에 저장되었습니다.")
        
        # 이미지 파일 테스트 시 1번만 실행하고 멈춤 (무한 저장 방지)
        if not webCamFeed:
             # 결과창을 띄우고 키 입력을 기다림
            lables = [["Original", "Gray", "Threshold", "Contours"],
                      ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]
            stackedImage = utils.stackImages(imageArray, 0.75, lables)
            cv2.imshow("Result", stackedImage)
            cv2.waitKey(0) 
            break

    else:
        # 감지 실패 시 빈 화면 표시
        imageArray = ([img, imgGray, imgThreshold, imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
        print("❌ 자격증을 찾을 수 없습니다.")

    # 결과 화면 표시 (웹캠 모드일 때 계속 갱신)
    lables = [["Original", "Gray", "Threshold", "Contours"],
              ["Biggest Contour", "Warp Prespective", "Warp Gray", "Adaptive Threshold"]]

    stackedImage = utils.stackImages(imageArray, 0.75, lables)
    cv2.imshow("Result", stackedImage)

    # 's' 키를 누르면 저장 (웹캠 모드용)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv2.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv2.FILLED)
        cv2.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        cv2.imshow('Result', stackedImage)
        cv2.waitKey(300)
        count += 1