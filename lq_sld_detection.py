import numpy as np
import cv2 as cv
from mmedit.apis import init_model
from mmedit.apis import restoration_inference
import torch
# torch.cuda.current_device()
# torch.cuda_initialized = True
import os
import shutil
import GLCM
from skimage import io, color, img_as_ubyte
import math
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import time

def cj_img(img):
    h,w = img.shape
    top = 0
    bottom = 0
    left = 0
    right = 0
    for i in range(h):
        flag = False
        for j in range(w):
            if img[i, j] != 255:
                top = i
                flag = True
                break
        if flag == True:
            break
    for i in range(h):
        flag = False
        for j in range(w):
            if img[h - 1 - i, j] != 255:
                bottom = i
                flag = True
                break
        if flag == True:
            break
    for i in range(w):
        flag = False
        for j in range(h):
            if img[j, i] != 255:
                left = i
                flag = True
                break
        if flag == True:
            break
    for i in range(w):
        flag = False
        for j in range(h):
            if img[j, w - i - 1] != 255:
                right = i
                flag = True
                break
        if flag == True:
            break
    new_img = img[ top -5 if top - 5 >=0 else 0 : h - bottom + 5 if h - bottom + 5 <= h else h ,left -5 if left -5 >= 0 else 0 : w - right + 5 if w - right +5 <= w else w]
    return new_img,(left,top)
# 获取滑片大小图片
def slice_img(img,dist =800,newLen = 1200):
    imgs = []
    h,w = img.shape
    if h<newLen and w >= newLen:
        v_count = 1
        h_count = math.ceil((w - newLen) / dist) + 1
        for j in range(h_count):
            left = dist * j if newLen + dist * j <= w else w - newLen
            right = newLen + dist * j if newLen + dist * j <= w else w
            top = 0
            bottom = h
            slice = img[top:bottom,left:right]
            new_slice = 255 * np.ones(shape=(newLen,newLen),dtype='uint8')
            new_slice[0:h,0:newLen] = slice
            info = [(left, top), new_slice]
            imgs.append(info)
    if h>=newLen and w <newLen:
        h_count = 1
        v_count = math.ceil((h - newLen) / dist) + 1
        for i in range(v_count):
            left = 0
            right = w
            top = dist * i if newLen + dist * i <= h else h - newLen
            bottom = newLen + dist * i if newLen + dist * i <= h else h
            slice = img[top: bottom, left: right]
            new_slice = 255 * np.ones(shape=(newLen, newLen),dtype='uint8')
            new_slice[0:newLen, 0:w] = slice
            info = [(left, top), new_slice]
            imgs.append(info)
    elif h>=newLen and w>= newLen:
        h_count = math.ceil((w - newLen) / dist) + 1
        v_count = math.ceil((h - newLen) / dist) + 1
        for i in range(v_count):
            for j in range(h_count):
                left = dist * j if newLen + dist * j <= w else w - newLen
                right = newLen + dist * j if newLen + dist * j <= w else w
                top = dist * i if newLen + dist * i <= h else h - newLen
                bottom = newLen + dist * i if newLen + dist * i <= h else h
                slice = img[top : bottom,left : right]
                info = [(left , top), slice]
                imgs.append(info)
    elif h<=newLen and w <=newLen:
        top = 0
        left = 0
        new_slice = 255 * np.ones(shape=(newLen, newLen), dtype='uint8')
        new_slice[0:h, 0:w] = img
        info = [(left, top), new_slice]
        imgs.append(info)

    return imgs
def get_detect_img_info(grayimg_dir,nottextimg_dir,*args):
    img = cv.imread(grayimg_dir, cv.IMREAD_GRAYSCALE)
    no_text_img = cv.imread(nottextimg_dir, cv.IMREAD_GRAYSCALE)
    h, w = img.shape
    ret, binary_img = cv.threshold(no_text_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
    white = np.ones(shape=img.shape, dtype='uint8') * 255
    # 获取竖线_img
    min_len = int(h / 40) if h >= w else int(w / 40)
    structure1 = cv.getStructuringElement(0, (1, min_len))
    eroded = cv.erode(binary_img, structure1, iterations=1)
    dilatedrow = cv.dilate(eroded, structure1, iterations=1)
    # 获取横线
    structure2 = cv.getStructuringElement(0, (min_len, 1))
    eroded = cv.erode(binary_img, structure2, iterations=1)
    dilatedcol = cv.dilate(eroded, structure2, iterations=1)
    binary_img = binary_img - dilatedcol - dilatedrow
    contours, hierarchy = cv.findContours(binary_img, mode=0, method=1)
    for contour in contours:
        left_top_x = cv.boundingRect(contour)[0]
        left_top_y = cv.boundingRect(contour)[1]
        Width = cv.boundingRect(contour)[2]
        Height = cv.boundingRect(contour)[3]
        white[left_top_y:left_top_y + Height, left_top_x:left_top_x + Width] = img[left_top_y:left_top_y + Height,
                                                                               left_top_x:left_top_x + Width]

    # white1 = gama_transfer(white, 2)
    if args:

        r = args[0]
        white = cv.resize(white,(0,0),fx=r,fy=r,interpolation=cv.INTER_AREA)
        # white1 = cv.resize(white1,(0,0),fx=r,fy=r,interpolation=cv.INTER_AREA)

    # white3 = gama_transfer(white1,2)
    # white2 = gama_transfer(white,1.8)

    cv.imwrite('output15.png',white)
    new_img, (left, top) = cj_img(white)
    # img_ori = cv.imread(ori_img,cv.IMREAD_GRAYSCALE)
    # if args:
    #     r = args[0]
    #     img_ori = cv.resize(img_ori, (0, 0), fx=r, fy=r, interpolation=cv.INTER_AREA)
    # newh,neww = new_img.shape
    # img_ori = img_ori[top:top+newh,left:left+neww]
    cv.imwrite('output16.png',new_img)
    imgs = slice_img(new_img)
    # new_img_1, (left, top) = cj_img(white1)
    # imgs_1 = slice_img(new_img)
    # new_img_2, (left, top) = cj_img(white2)
    # imgs_2 = slice_img(new_img)
    # new_img_3, (left, top) = cj_img(white3)
    # imgs_3 = slice_img(new_img)
    # imgs.extend(imgs_1)
    # imgs.extend(imgs_2)
    # imgs.extend(imgs_3)
    return imgs, left, top

def gama_trans(img_dir,gama):
    img = cv.imread(img_dir,cv.IMREAD_GRAYSCALE)
    invGama = 1.0/gama
    table=[]
    for i in range(256):
        table.append(((i/255.0) ** invGama) * 255)
    table = np.array(table).astype('uint8')
    return cv.LUT(img,table)


# 获取框以及置信2
def updata_every_result(result,x,y,left,top):
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j][0] = result[i][j][0] + left + x - 6
            result[i][j][1] = result[i][j][1] + top + y - 6
            result[i][j][2] = result[i][j][2] + left + x - 6
            result[i][j][3] = result[i][j][3] + top + y - 6
    return result
def get_all_result(model,classes_len,imgs,cj_left,cj_top):
    for i in range(len(imgs)):
        img = cv.cvtColor(imgs[i][-1],cv.COLOR_GRAY2RGB)
        result = inference_detector(model,img)
        x,y = imgs[i][0]
        result = updata_every_result(result,x,y,cj_left,cj_top)
        if i == 0:
            all_result = result
        else:
            for j in range(classes_len):
                all_result[j] = np.append(all_result[j],values=result[j],axis=0)
    return all_result
def cal_iou(x1, y1, x2, y2, x1_, y1_, x2_, y2_):
    w = max((min(x2, x2_) - max(x1, x1_)), 0)
    h = max((min(y2, y2_) - max(y1, y1_)), 0)
    area = w * h
    iou = area / ((x2 - x1) * (y2 - y1) + (x2_ - x1_) * (y2_ - y1_) - area)
    return iou
def updata_all_result(classes_len,result):
    new_result = []
    for i in range(classes_len):
        result_class = []
        if len(result[i]) == 0:
            result_class = result[i]
        else:
            for j in range(len(result[i])):
                x1,y1,x2,y2,c1 = result[i][j]
                flag = True
                if c1 < 0.99:
                    flag = False
                for k in range(len(result[i])):
                    if k != j:
                        x1_,y1_,x2_,y2_,c2 = result[i][k]
                        iou = cal_iou(x1,y1,x2,y2,x1_,y1_,x2_,y2_)
                        if iou >0.2:
                            if c1 < c2:
                                flag = False
                                break
                            if c1 == c2:
                                if j  > k:
                                    flag = False
                                    break
                if flag == True:
                    result_class.append(result[i][j])
        new_result.append(np.array(result_class))
    return new_result
def updata_all_result_2(classes_len,result):
    new_result = []
    for i in range(classes_len):
        result_class=[]
        if len(result[i]) == 0:
            result_class = result[i]
        else:
            for j in range(len(result[i])):
                x1,y1,x2,y2,c1 = result[i][j]
                flag = True
                for k in range(len(result)):
                    if k != i:
                        for h in range(len(result[k])):
                            x1_, y1_, x2_, y2_, c2 = result[k][h]
                            iou = cal_iou(x1, y1, x2, y2, x1_, y1_, x2_, y2_)
                            if iou >0.6:
                                if c1 < c2:
                                    flag = False
                                    break
                if flag == True:
                    result_class.append(result[i][j])
        new_result.append(np.array(result_class))
    for i in range(classes_len):
        if len(new_result[i]) == 0:
            new_result[i] = np.array([[0,0,0,0,0]])
    return new_result

def get_all_result_final(model,classes_len,imgs,cj_left,cj_top):
    result = get_all_result(model,classes_len,imgs,cj_left,cj_top)
    result = updata_all_result(classes_len,result)
    result = updata_all_result_2(classes_len,result)
    return result



def get_slices(img_dir,slice_len = 50):
    slices = []
    img = cv.imread(img_dir)
    h,w,c = img.shape
    if w % slice_len != 0 :
        new_w = w + slice_len - (w % slice_len)
    else:
        new_w = w
    if h % slice_len != 0:
        new_h = h + slice_len - (h % slice_len)
    else:
        new_h = h
    new_img = np.ones(shape=(new_h,new_w,3)) * 255
    new_img[0:h,0:w,:] = img
    w_n = int(new_img.shape[1] / slice_len)
    h_n = int(new_img.shape[0] / slice_len)
    for i in range(h_n):
        for j in range(w_n):
            slice = new_img[i * slice_len:(i+1)*slice_len, j*slice_len:(j+1)*slice_len, :]
            filename = './cache/_' + str(i) + '_' + str(j) + '.jpg'
            cv.imwrite(filename,slice)
            slice_info = (filename,i,j)
            slices.append(slice_info)
    return  slices,h_n,w_n,new_img

def Kmeans(slices):
    Z = []
    for i in range(len(slices)):
        img = io.imread(slices[i][0])
        gray = color.rgb2gray(img)
        t = GLCM.get_GLCM_texture(gray)
        Z.append(t)
    Z = np.array(Z)
    Z_normal = Z / Z.max(axis=0)
    Z = Z_normal
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    ret, label, center = cv.kmeans(Z, 2, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    slices = np.array(slices)
    new_slices = np.append(slices, label, axis=1)
    return new_slices

def construct_hr(slices,h_n,w_n,model,slice_len=50,scale=4):
    hr_slice_len = slice_len * scale
    img = np.ones(shape=(h_n*hr_slice_len,w_n*hr_slice_len,3))*255
    for i in range(h_n * w_n):
        filename, h , w, label = slices[i]
        h = int(h)
        w = int(w)
        if label == '1':
            result = restoration_inference(model, filename)
            result = torch.clamp(result, 0, 1)
            img_SR = result.squeeze(0).permute(1, 2, 0).numpy()
            img_SR = np.array(img_SR*255,dtype='uint8')
            img[h*hr_slice_len:(h+1)*hr_slice_len, w * hr_slice_len:(w+1) * hr_slice_len, :] = img_SR
        if label == '0':
            img = cv.imread(filename,cv.IMREAD_GRAYSCALE)
            ret = cv.resize(img,(0,0),fx = scale,fy=scale)
            img_SR = cv.cvtColor(ret, cv.COLOR_GRAY2RGB)
            img_SR = np.array(img_SR, dtype='uint8')
            img[h * hr_slice_len:(h + 1) * hr_slice_len, w * hr_slice_len:(w + 1) * hr_slice_len, :] = img_SR
            # result = restoration_inference(model, filename)
            # result = torch.clamp(result, 0, 1)
            # img_SR = result.squeeze(0).permute(1, 2, 0).numpy()
            # img_SR = np.array(img_SR*255,dtype='uint8')
            # img[h*hr_slice_len:(h+1)*hr_slice_len, w * hr_slice_len:(w+1) * hr_slice_len, :] = img_SR
    return img

def main(imgdir,model,slice_len=50,scale=4):
    shutil.rmtree('./cache')
    os.mkdir('./cache')
    slices ,h_n, w_n,lr_img= get_slices(img_dir=imgdir,slice_len=slice_len)
    slices = Kmeans(slices)
    img = construct_hr(slices,h_n,w_n,model,slice_len=slice_len,scale=scale)
    return img,lr_img

# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
start_time = time.time()

# 检测模型
config_file1 = '../mmdetection/checkpoints/fr_2000/elements.py'
check_point1 = '../mmdetection/checkpoints/fr_2000/epoch_12.pth'
model1 = init_detector(config_file1, check_point1)

# config_file1 = '../mmdetection/work_dirs/ssd-Lite/1.py'
# check_point1 = '../mmdetection/work_dirs/ssd-Lite/latest.pth'
# model1 = init_detector(config_file1, check_point1)

# config_file1 = '../mmdetection/checkpoints/swin_fasterrcnn_2000/elements.py'
# check_point1 = '../mmdetection/checkpoints/swin_fasterrcnn_2000/epoch_12.pth'
# model1 = init_detector(config_file1, check_point1)

# config_file1 = '../mmdetection/work_dirs/st_new/1.py'
# check_point1 = '../mmdetection/work_dirs/st_new/latest.pth'
# model1 = init_detector(config_file1, check_point1)

# 超分模型
config_file2 = './configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py'
check_point2 = './work_dirs/esrgan0.25/latest.pth'
model2 = init_model(config_file2,check_point2)

img,lr_img = main('../LR_test_imgs/sf=0.25/test16.png',model2,slice_len=50)
# img,lr_img = main('./img_4.png',model2,slice_len=100)
# img = cv .resize(img,(0,0),fx=0.5,fy=0.5)
cv.imwrite('output10.png',img)
cv.imwrite('output11.png',lr_img)

end_time = time.time()
execution_time = end_time - start_time
print('time:',execution_time)
# i=0
# input_dir = '../mmdetection/data/image_test0.25'
# output_dir = '../mmdetection/data/images_test0.25'
# for filename in os.listdir(input_dir):
#     img_path = os.path.join(input_dir, filename)
#     img,lrimg = main(img_path, model2,slice_len=50)
#     output_path = os.path.join(output_dir, filename)
#     cv.imwrite(output_path,img)
#     i = i+1
#     print(i)

import Laplacian
# HR_img = cv.imread('output10.png')
# LR_img = cv.imread('output11.png')
# blend_img = Laplacian.modified_laplacian_p(HR_img, LR_img)
#
# cv.imwrite('output12.png',blend_img *0.35+ HR_img * 0.65)
#
# imgs,left,top = get_detect_img_info('./output12.png','./output12.png')
# result = get_all_result_final(model1,8,imgs,cj_left=left,cj_top=top)
# img = cv.imread('./output10.png')
# show_result_pyplot(model1,img,result,out_file='./output_test2.png')
# print("num(disconncetor):",len(result[0]))
# print("num(dr):",len(result[1]))
# print("num(dg):",len(result[2]))
# print("num(byq):",len(result[3]))
# print("num(fdj):",len(result[4]),result[4])
# print("num(glkg):",len(result[5]))
# print("num(jd):",len(result[6]))
# print("num(jdkg):",len(result[7]))


imgs,left,top = get_detect_img_info('./output10.png','./output10.png')
result = get_all_result_final(model1,8,imgs,cj_left=left,cj_top=top)
img = cv.imread('./output10.png')
show_result_pyplot(model1,img,result,out_file='./output_test.png')
# print("num(disconncetor):",len(result[0]))
# print("num(dr):",len(result[1]))
# print("num(dg):",len(result[2]))
# print("num(byq):",len(result[3]))
# print("num(fdj):",len(result[4]),result[4])
# print("num(glkg):",len(result[5]))
# print("num(jd):",len(result[6]))
# print("num(jdkg):",len(result[7]))