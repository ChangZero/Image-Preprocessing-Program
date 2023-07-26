import cv2
import numpy as np
from module.conv import bluring, sharpen, prewitt, sobel, laplacian

flag = False # 마우스 좌클릭(LBUTTONDOWN) 체크 변수
oldx, oldy = -1, -1 # 과거 마우스 포인터의 x,y 위치값/초기화 진행
def onMouse(event, x, y, flags, param):
    global img              # 원 영상
    global pointer_img      # 마우스 포인터를 표기한 영상
    global drag_img         # 마우스 드래그를 표기한 영상
    global mask_img         # 터치 효과 부분 마스크 처리한 영상
    global done_img         # 터치 효과가 한번 적용된 부분을 마스크 처리한 영상
    global flag             # 마우스 좌클릭(LBUTTONDOWN) 체크 변수
    global oldx, oldy       # 과거 마우스 포인터의 x,y 위치값
    
    not_done_img = cv2.bitwise_not(done_img)    # NOT연산으로 터치 적용안된 부분 마스크 처리 -> 추후 터치효과가 적용되어지는 부분과 AND연산으로 터치효과가 한번 진행된 부분 제거함 
    if (x > 250) | (y > 50): # 텍스트 영역 보호
        if event == cv2.EVENT_LBUTTONDOWN:
            mask_img= np.zeros_like(img) # 전체 화소값 0인 마스크 생성
            oldx, oldy = x, y # 과거 마우스 포인터 업데이터
            drag_img = img.copy() # 마우스 드래그 영역 이미지 생성
            flag = True
            
        elif (flag == True) & (event == cv2.EVENT_MOUSEMOVE): # 마우스 좌클릭 & 마우스 포인터 이동시
            # 마우스 드래그 영역 그리기
            cv2.line(drag_img, (oldx, oldy), (x, y), (0,0,255), 60)
            cv2.circle(drag_img, (x,y), 30, (0,0,255), -1)
            
            # 터치 효과 영역 마스크 그리기
            cv2.line(mask_img, (oldx, oldy), (x, y), (255,255,255), 60)
            cv2.circle(mask_img, (x,y), 30, (255,255,255), -1)
            
            mask_img = cv2.bitwise_and(not_done_img, mask_img) # 터치효과 적용된 부분과 NOT연산으로 터치 적용안된 마스크부분을 AND연산 처리하여 한번 터치효과가 적용된 부분은 터치효과 적용되지 않도록 마스크 처리
            cv2.imshow('img', drag_img)
        
            oldx, oldy = x,y # 과거 마우스 포인터 위치 없데이트
            
        elif event == cv2.EVENT_LBUTTONUP: # 마우스 드래그 종료시
            filter_img = img.copy() # 회선(터치효과)를 부여한 이미지를 생성
            filter_img = cv2.split(filter_img)[2] # R채널만 분리 -> 이유: 마우스 포인터의 색깔이 빨간색이기 때문에 추후 마스크 처리로 영상을 합성할때 용이함
            
            if param == 1:
                fg_img = bluring(filter_img)    # Bluring 진행
                fg_img = cv2.cvtColor(fg_img, cv2.COLOR_GRAY2BGR) # GRAY -> BGR
                cv2.copyTo(fg_img, mask_img, img) # 회선처리된 이미지에서 마스크영역(mask_img)의 부분을 본영상(img)에 합성
                
            elif param == 2:
                fg_img = sharpen(filter_img)    # Sharpen 진행
                fg_img = cv2.cvtColor(fg_img, cv2.COLOR_GRAY2BGR) # GRAY -> BGR
                cv2.copyTo(fg_img, mask_img, img) # 회선처리된 이미지에서 마스크영역(mask_img)의 부분을 본영상(img)에 합성
                
            elif param == 3:
                fg_img = prewitt(filter_img) # Prewitt 진행
                fg_img = cv2.cvtColor(fg_img, cv2.COLOR_GRAY2BGR) # GRAY -> BGR
                cv2.copyTo(fg_img, mask_img, img) # 회선처리된 이미지에서 마스크영역(mask_img)의 부분을 본영상(img)에 합성
                
            elif param == 4:
                fg_img = sobel(filter_img) # Sobel 진행
                fg_img = cv2.cvtColor(fg_img, cv2.COLOR_GRAY2BGR) # GRAY -> BGR
                cv2.copyTo(fg_img, mask_img, img) # 회선처리된 이미지에서 마스크영역(mask_img)의 부분을 본영상(img)에 합성
                
            elif param == 5:
                fg_img = laplacian(filter_img) # Laplacian 진행
                fg_img = cv2.cvtColor(fg_img, cv2.COLOR_GRAY2BGR) # GRAY -> BGR
                cv2.copyTo(fg_img, mask_img, img) # 회선처리된 이미지에서 마스크영역(mask_img)의 부분을 본영상(img)에 합성
                
            else:
                print("No Convolution")
            
            flag = False
            done_img = cv2.bitwise_or(done_img, mask_img) # 새로 터치효과 부여된 부분과 기존 터치효과 부여된 부분 OR 연산
        else:
            # 마우스 포인터 출력 진행
            pointer_img = img.copy()
            cv2.circle(pointer_img, (x,y), 30, (0,0,255), -1)
            cv2.imshow('img', pointer_img)
        

def main():
    global img # 본 영상
    global mask_img # 터치효과 부분 마스크 처리한 영상
    global done_img # 터치효과 한번이라도 적용된 부분 마스크 처리한 영상
    
    # 영상 읽어오기
    img =cv2.imread('input.jpg')
    img = cv2.resize(img, dsize=(1920,1080), interpolation=cv2.INTER_LINEAR)
    
    mask_img= np.zeros_like(img) # 영상 초기화
    done_img= np.zeros_like(img) # 영상 초기화
    
    # 텍스트 표기
    cv2.rectangle(img, (0,0), (250, 50), (0,0,0),cv2.FILLED)
    cv2.putText(img, 'mask manu', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(150,150,150))
    
    # 영상 출력
    cv2.imshow('img', img)
    
    masking = False # 키보드(1~5)가 입력 여부 확인 변수
    while True:
        key = cv2.waitKey(1) # 키 입력
        if key == ord('1'):
            cv2.rectangle(img, (0,0), (250, 50), (0,0,0),cv2.FILLED)
            cv2.putText(img, 'Bluring', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(150,150,150))
            masking = 1
            cv2.imshow('img', img)
        elif key == ord('2'):
            cv2.rectangle(img, (0,0), (250, 50), (0,0,0),cv2.FILLED)
            cv2.putText(img, 'Sharpening', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(150,150,150))
            masking = 2
            cv2.imshow('img', img)
        elif key == ord('3'):
            cv2.rectangle(img, (0,0), (250, 50), (0,0,0),cv2.FILLED)
            cv2.putText(img, 'Prewitt', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(150,150,150))
            masking = 3
            cv2.imshow('img', img)
        elif key == ord('4'):
            cv2.rectangle(img, (0,0), (250, 50), (0,0,0),cv2.FILLED)
            cv2.putText(img, 'Sobel', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(150,150,150))
            masking = 4
            cv2.imshow('img', img)
        elif key == ord('5'):
            cv2.rectangle(img, (0,0), (250, 50), (0,0,0),cv2.FILLED)
            cv2.putText(img, 'Laplacian', (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(150,150,150))
            masking = 5
            cv2.imshow('img', img)
        elif key == ord('q'):
            cv2.imwrite('./save_img/20181794_1.jpg',img) # 터치 효과 적용된 영상 저장
            
            n_done_img = cv2.bitwise_not(done_img) # NOT연산으로 터치 적용안된 부분 마스크 처리 -> 추후 터치효과가 적용되어지는 부분과 AND연산으로 터치효과가 한번 진행된 부분 제거함
            img2 = cv2.copyTo(img, n_done_img) # 원 영상에서 마스크영역(터치효과가 적용되지 않은 부분)의 부분 새로운 영상으로 합성
            cv2.imwrite('./save_img/20181794_2.jpg',img2) # 터치 효과 적용 영역의 화소 값 0 변환 영상 저장
            
            break

        if masking: # 회선처리시 
            cv2.setMouseCallback('img', onMouse, param=masking) # 마스킹 번호 파라미터 넘겨줌
        else:
            cv2.setMouseCallback('img', onMouse)      
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()