'''
-----------------------------------
Created time : 2018/1/25

Aythor : chen ye
-----------------------------------
'''

import cv2
path = r'C:\Users\cheny\PycharmProjects\crawl\Data'

img = cv2.imread(path + '\\test5.png', 0)
img2 = img
cv2.namedWindow('image')

def nothing(x):
    pass

cv2.createTrackbar('minVal ', 'image', 0, 100, nothing)
cv2.createTrackbar('maxVal', 'image', 200, 400, nothing)

while(1):
    '''
    利用trackbar对图像阈值进行设定，利用m键可以生成边缘提取后的图像，r键进行图像还原，Esc键退出
    '''
    cv2.imshow('image', img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('m'):
        img = cv2.Canny(img, cv2.getTrackbarPos('minVal', 'image'), cv2.getTrackbarPos('maxVal', 'image'))
    elif k == 27:
        break
    elif k == ord('r'):
        img = img2

cv2.destroyAllWindows()