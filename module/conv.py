import cv2
import numpy as np

def filter(img, mask):
    rows, cols = img.shape[:2]
    dst = np.zeros((rows, cols), np.float32) # 회선 결과 저장 행렬
    xcenter, ycenter = mask.shape[1]//2, mask.shape[0]//2 # 마스크 중심 좌표
    
    for i in range(ycenter, rows - ycenter): # 입력 행렬 반복 순회
        for j in range(xcenter, cols - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1 # 관심영역 높이 범위
            x1, x2 = j - xcenter, j + xcenter + 1 # 관심영역 너비 볌위
            roi = img[y1:y2, x1:x2].astype('float32') # 관심영역 형변환
            tmp = cv2.multiply(roi, mask) # 회선 적용 - OpenCV 곱셈
            dst[i, j] = cv2.sumElems(tmp)[0] # 출력화소 저장
    return dst

def bluring(img):
    data = [1/9, 1/9, 1/9,
            1/9, 1/9, 1/9,
            1/9, 1/9, 1/9]
    
    mask = np.array(data, np.float32).reshape(3,3) 
    dst = filter(img, mask) #회선 수행
    dst = dst.astype('uint8')
    return dst

def sharpen(img):
    data = [-1, -1, -1,
            -1, 9, -1,
            -1, -1, -1]
    mask = np.array(data, np.float32).reshape(3,3)
    dst = filter(img, mask) # 회선 수행
    dst = cv2.convertScaleAbs(dst) # 윈도우 표기 위해 OpenCV 함수로 형변환 및 saturation 수행
    return dst

def prewitt(img):
    data1 = [-1, 0, 1,
            -1, 0, 1,
            -1, 0, 1]
    
    data2 = [-1, -1, -1,
            0, 0, 0,
            1, 1, 1]
    
    mask1 = np.array(data1, np.float32).reshape(3,3)
    mask2 = np.array(data2, np.float32).reshape(3,3)
    
    dst1 = filter(img, mask1) # mask1 회선 수행
    dst2 = filter(img, mask2) # mask2 회선 수행
    dst = cv2.magnitude(dst1, dst2) # 회선 결과 두 행렬의 크기 계산
    dst = cv2.convertScaleAbs(dst) # 윈도우 표기 위해 OpenCV 함수로 형변환 및 saturation 수행
    return dst
    
def sobel(img):
    data1 = [-1, 0, 1,
             -2, 0, 2,
             -1, 0, 1]
    
    data2 = [-1, -2, -1,
             0, 0, 0,
             1, 2, 1]
    
    mask1 = np.array(data1, np.float32).reshape(3,3)
    mask2 = np.array(data2, np.float32).reshape(3,3)
    
    dst1 = filter(img, mask1) # mask1 회선 수행
    dst2 = filter(img, mask2) # mask2 회선 수행
    dst = cv2.magnitude(dst1, dst2) # 회선 결과 두 행렬의 크기 계산
    dst = cv2.convertScaleAbs(dst) # 윈도우 표기 위해 OpenCV 함수로 형변환 및 saturation 수행
    return dst
 
def laplacian(img):
    data = [[-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]] # 8 방향 필터
    
    mask8 = np.array(data, np.int16) # 음수가 있으므로 자료형이 int16행렬 선언
    dst = cv2.filter2D(img, cv2.CV_16S, mask8) # filter2D()를 통한 라플라시안 수행
    dst = cv2.convertScaleAbs(dst) # 윈도우 표기 위해 OpenCV 함수로 형변환 및 saturation 수행
    return dst

