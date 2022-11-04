import cv2
import numpy as np
import deci2bit

def make_noise(std, img):
    height, width = img.shape
    img_noise = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for a in range(width):
            make_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * make_noise
            img_noise[i][a] = img[i][a] + set_noise
    return img_noise


def nothing(x):
    pass

cv2.namedWindow('with noise')
cv2.createTrackbar('standard', 'with noise', 0, 250, nothing)
cv2.setTrackbarPos('standard', 'with noise', 127)

img_color = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', img_gray)

std = 0
img_noise = make_noise(std, img_gray)
while(True):
    std = cv2.getTrackbarPos('standard', 'with noise')/10
    img_noise = make_noise(std, img_gray)

    cv2.imshow('with noise', img_noise.astype(np.uint8))

#    img_result = cv2.bitwise_and(img_color, img_color, mask = img_binary)
 #   cv2.imshow('Result', img_result)


    if cv2.waitKey(100)&0xFF == 27:
        break


cv2.destroyAllWindows()