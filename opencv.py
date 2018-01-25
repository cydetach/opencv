# coding=utf-8

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
path = r'C:\Users\cheny\PycharmProjects\crawl\Data'

'''人脸识别'''
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#
# img = cv2.imread('test.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, 1.3, 2)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# img = cv2.imread('test.jpg',-1)
# cv2.imshow('image',img)# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# img = cv2.imread('test.jpg',0)
# cv2.imshow('image',img)
# k = cv2.waitKey(0)
# if k == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()
# elif k == ord('s'): # wait for 's' key to save and exit
#     cv2.imwrite('messigray.png',img)
#     cv2.destroyAllWindows()



'''Create a black image'''
# img = np.zeros((512,512,3), np.uint8)
#
# # Draw a diagonal blue line with thickness of 5 px
# img = cv2.line(img,(0,0),(511,511),(255,0,0),5)
#
# img = cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
# img = cv2.circle(img,(447,63), 63, (0,0,255), 2)
#
# img = cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
#
# pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# pts = pts.reshape((-1,1,2))
# img = cv2.polylines(img,[pts],True,(0,255,255))
#
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

'''画圆函数'''
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#
# img = np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
#
# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20)& 0xff == 27:
#         break
# cv2.destroyAllWindows()

'''实现画圆活着画圈的转变：'''
# drawing = False
# mode = True
# ix,iy = -1,-1
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#
#     if event == cv2.EVENT_LBUTTONDOWN:
#         drawing = True
#         ix,iy = x,y
#
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if drawing == True:
#             if mode == True:
#                 cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),1)
#             else:
#                 cv2.circle(img,(x,y),5,(0,0,255),1)
#
#     elif event == cv2.EVENT_LBUTTONUP:
#         drawing =False
#         if mode == True:
#             cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 1)
#         else:
#             cv2.circle(img, (x, y), 5, (0, 0, 255), 1)
# #
# img = np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
# while(1):
#     cv2.imshow('image',img)
#     k = cv2.waitKey(20)& 0xff
#     if k == ord('m'):
#         mode = not mode
#     elif k==27:
#         break
# cv2.destroyAllWindows()



# def nothing(x):
#     pass
# img = np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image')
#
# cv2.createTrackbar('R', 'image', 0, 255, nothing)
# cv2.createTrackbar('G', 'image', 0, 255, nothing)
# cv2.createTrackbar('B', 'image', 0, 255, nothing)
#
# switch = '0 : OFF \n1 : ON'
# cv2.createTrackbar(switch,'image',0,1,nothing)
#
# while(1):
#     cv2.imshow('image', img)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#     r = cv2.getTrackbarPos('R','image')
#     g = cv2.getTrackbarPos('G','image')
#     b = cv2.getTrackbarPos('B','image')
#     s = cv2.getTrackbarPos(switch,'image')
#     if s == 0:
#         img[:] = 0
#     else:
#         img[:] = [b,g,r]
# cv2.destroyAllWindows()

# from matplotlib import pyplot as plt
# BLUE = [0,0,255]
#
# img1 = cv2.imread('test.jpg')
#
# replicate = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
# reflect = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101 = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
# constant= cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)
#
# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
#
# plt.show()


# img1 = cv2.imread('Track1.bmp')
# img2 = cv2.imread('Track2.bmp')
# print(cv2.useOptimized())
# dst = cv2.addWeighted(img1,0.5,img2,0.5,0)
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # I want to put logo on top-left corner, So I create a ROI
# rows,cols,channels = img2.shape
# roi = img1[0:rows, 0:cols]
#
# # Now create a mask of logo and create its inverse mask also
# img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
# mask_inv = cv2.bitwise_not(mask)
#
# # Now black-out the area of logo in ROI
# img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
#
# # Take only region of logo from logo image.
# img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
#
# # Put logo in ROI and modify the main image
# dst = cv2.add(img1_bg,img2_fg)
# img1[0:rows, 0:cols ] = dst
#
# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# green =np.uint8([[[0, 255, 0]]])
# hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
# print(hsv_green)

''' 提取某一种颜色'''
# img = cv2.imread('test3.jpg')
# img = cv2.resize(img, (540, 480))
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
# lower_green = np.array([50, 100, 100])
# upper_green = np.array([70, 255, 255])
#
# mask = cv2.inRange(hsv,lower_green,upper_green)
# res = cv2.bitwise_and(img, img, mask = mask)
# cv2.imshow('img',   img)
# cv2.imshow('mask',  mask)
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

''' simple thresholding '''
# img = cv2.imread('gradient.png', 0)
# ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret,thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret,thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# ret,thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret,thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
#
# for i in range(6):
#     plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# '''Adaptive thresholding'''
# img = cv2.imread(path + '\\test5.jpg', 0)
# img = cv2.medianBlur(img, 5)
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#
# titles = ['Original Image', 'Global Thresholding (v = 127)',
#             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# images = [img, th1, th2, th3]
#
# for i in range(4):
#     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])
# plt.show()


# '''Otsu’s Binarization'''
# img = cv2.imread(path + '\\test5.jpg', 0)
#
# # global thresholding
# ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
#
# # Otsu's thresholding
# ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # Otsu's thresholding after Gaussian filtering
# blur = cv2.GaussianBlur(img, (5, 5), 0)
# ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # plot all the images and their histograms
# images = [img, 0, th1,
#           img, 0, th2,
#           blur, 0, th3]
# titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
#           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
#           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
#
# for i in range(3):
#     plt.subplot(3, 3, i*3+1), plt.imshow(images[i*3], 'gray')
#     plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i*3+2), plt.hist(images[i*3].ravel(), 256)
#     plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
#     plt.subplot(3, 3, i*3+3), plt.imshow(images[i*3+2], 'gray')
#     plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
# plt.show()


# img = cv2.imread(path + '\\test5.jpg')
#
#
#
# cv2.imshow('res', res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread(path + '\\test5.jpg', 0)
# img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
# rows, cols = img.shape
#
# M = np.float32([[1, 0, -300], [0, 1, -50]])
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# cv2.imshow('img', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.imread(path + '\\test5.jpg')
# img = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_CUBIC)
# rows, cols, ch = img.shape
#
# pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
# pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
#
# M = cv2.getAffineTransform(pts1, pts2)
#
# dst = cv2.warpAffine(img, M, (cols, rows))
#
# plt.subplot(121), plt.imshow(img), plt.title('Input')
# plt.subplot(122), plt.imshow(dst), plt.title('Output')
# plt.show()

# img = cv2.imread(path + '\\j.png', 0)
# kernel = np.ones((9, 9), np.uint8)
# erosion = cv2.erode(img, kernel, iterations=1)
# dilation = cv2.dilate(img, kernel, iterations=2)
# opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
#
# cv2.imshow('tophat', tophat)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # cv2.imwrite(path + '\\j.png', opening)


# cv2.imshow('image', img)
# # k = cv2.waitKey(1) & 0xFF
# # if k == 27:
# #     cv2.destroyAllWindows()
# minVal = cv2.getTrackbarPos('minVal', 'image')
# maxVal = cv2.getTrackbarPos('maxVal', 'image')
# img = cv2.Canny(img, minVal, maxVal)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# plt.subplot(121), plt.imshow(img, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(edges, cmap='gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# plt.show()