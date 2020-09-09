import cv2
import glob
import random
import dlib
import numpy as np
import math
import itertools
from sklearn.svm import SVC
import PIL
import warnings
from PIL import Image
import os
from natsort import natsorted

path_1 = r"E:/video/111/"  # 视频目录
path_2 = r"E:/video/111/"  # 导出目录

def get_datasets(emotion):
    files = glob.glob("E:\\Mutilmodal expression recognition\\CK+\\%s\\*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction




# def get_landmarks(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     clahe_image = clahe.apply(gray)
#     detections = detector(clahe_image, 1)
#     for k,d in enumerate(detections): #For all detected face instances individually
#         shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
#         xlist = []
#         ylist = []
#         landmarks= []
#         for i in range(0,68): #Store X and Y coordinates in two lists
#             cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
#             #For each point, draw a red circle with thickness2 on the original frame
#             xlist.append(float(shape.part(i).x))
#             ylist.append(float(shape.part(i).y))
#
#         xmean = np.mean(xlist) #Find both coordinates of centre of gravity
#         ymean = np.mean(ylist)
#         x_max = np.max(xlist)
#         x_min = np.min(xlist)
#         y_max = np.max(ylist)
#         y_min = np.min(ylist)
#         cv2.rectangle(clahe_image,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,150,0),2)
#         # print ("centre of gravity",xmean, ymean)
#         # print ("range of the face",x_max, x_min, y_max, y_min)
#         cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2)
#         # for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
#         #     landmarks.append(x)
#         #     landmarks.append(y)
#         # print ("landmarks coords",landmarks)
#         # cv2.imshow("image", image)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()
#         xlist = np.array(xlist,dtype = np.float64)
#         ylist = np.array(ylist,dtype = np.float64)
#         # xlist = np.float32(xlist)
#         # ylist = np.float32(ylist)
#
#     if len(detections) > 0:
#         return xlist, ylist
#     else: #If no faces are detected, return error message to other function to handle
#         xlist = np.array([])
#         ylist = np.array([])
#         return xlist, ylist


# this is the get_landmarks algorithms with cropping and resizing
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        landmarks= []
        for i in range(0,68): #Store X and Y coordinates in two lists
            cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
            #For each point, draw a red circle with thickness2 on the original frame
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        x_max = np.max(xlist)
        x_min = np.min(xlist)
        y_max = np.max(ylist)
        y_min = np.min(ylist)
        cv2.rectangle(clahe_image,(int(x_min),int(y_min-((ymean - y_min)/3))),(int(x_max),int(y_max)),(255,150,0),2)
        # print ("centre of gravity",xmean, ymean)
        # print ("range of the face",x_max, x_min, y_max, y_min)
        cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2)

        x_start = int(x_min)
        y_start = int(y_min-((ymean - y_min)/3))
        w = int(x_max) - x_start
        h = int(y_max) - y_start

        crop_img = image[y_start:y_start+h, x_start:x_start+w] # Crop from {x, y, w, h } => {0, 0, 300, 400}

    if len(detections) > 0:
        mywidth = 255
        hsize = 255
        cv2.imwrite('crop_img.png',crop_img)
        img = Image.open('crop_img.png')
        # wpercent = (mywidth/float(img.size[0]))
        # hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
        img.save('resized.png')

        image_resized = cv2.imread('resized.png')
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1)
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []
            landmarks= []
            for i in range(0,68): #Store X and Y coordinates in two lists
                cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
                #For each point, draw a red circle with thickness2 on the original frame
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))

            xmean = np.mean(xlist) #Find both coordinates of centre of gravity
            ymean = np.mean(ylist)
            x_max = np.max(xlist)
            x_min = np.min(xlist)
            y_max = np.max(ylist)
            y_min = np.min(ylist)
            cv2.rectangle(clahe_image,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,150,0),2)
            # print ("centre of gravity",xmean, ymean)
            # print ("range of the face",x_max, x_min, y_max, y_min)
            cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2)
            # for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            #     landmarks.append(x)
            #     landmarks.append(y)
            # print ("landmarks coords",landmarks)
            # cv2.imshow("image", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            xlist = np.array(xlist,dtype = np.float64)
            ylist = np.array(ylist,dtype = np.float64)
            # xlist = np.float32(xlist)
            # ylist = np.float32(ylist)

        if len(detections) > 0:
            return xlist, ylist
        else: #If no faces are detected, return error message to other function to handle
            xlist = np.array([])
            ylist = np.array([])
            return xlist, ylist
    else:
        xlist = np.array([])
        ylist = np.array([])
        return xlist, ylist


def linear_interpolation(xlist,ylist):
    xlist = np.array(xlist,dtype = np.float64)
    ylist = np.array(ylist,dtype = np.float64)
    x_new = np.array([])
    y_new = np.array([])
    for i in range (len(xlist)-1):
        x_new = np.concatenate((x_new,[(xlist[i]+xlist[i+1])/2.0]))
        y_new = np.concatenate((y_new,[(ylist[i]+ylist[i+1])/2.0]))
    xlist = np.append(xlist, x_new)
    ylist = np.append(ylist, y_new)
    return xlist, ylist

def extract_AU(xlist,ylist):
    AU_feature = []
    Norm_AU_feature = []
    AU1_1_x = xlist[19:22]
    AU1_1_y = ylist[19:22]
    AU1_1_x,AU1_1_y = linear_interpolation(AU1_1_x,AU1_1_y)
    AU_feature = [get_average_curvature(AU1_1_x,AU1_1_y)]

    AU1_2_x = xlist[22:25]
    AU1_2_y = ylist[22:25]
    AU1_2_x,AU1_2_y = linear_interpolation(AU1_2_x,AU1_2_y)
    AU_feature = AU_feature + [get_average_curvature(AU1_2_x,AU1_2_y)]

    AU2_1_x = xlist[17:20]
    AU2_1_y = ylist[17:20]
    AU2_1_x,AU2_1_y = linear_interpolation(AU2_1_x,AU2_1_y)
    AU_feature = AU_feature + [get_average_curvature(AU2_1_x,AU2_1_y)]
    AU2_2_x = xlist[24:27]
    AU2_2_y = ylist[24:27]
    AU2_2_x,AU2_2_y = linear_interpolation(AU2_2_x,AU2_2_y)
    AU_feature = AU_feature + [get_average_curvature(AU2_2_x,AU2_2_y)]

    AU5_1_x = xlist[36:40]
    AU5_1_y = ylist[36:40]
    AU5_1_x,AU5_1_y = linear_interpolation(AU5_1_x,AU5_1_y)
    AU_feature = AU_feature + [get_average_curvature(AU5_1_x,AU5_1_y)]
    AU5_2_x = xlist[42:46]
    AU5_2_y = ylist[42:46]
    AU5_2_x,AU5_2_y = linear_interpolation(AU5_2_x,AU5_2_y)
    AU_feature = AU_feature + [get_average_curvature(AU5_2_x,AU5_2_y)]

    AU7_1_x = np.append(xlist[39:42],xlist[36])
    AU7_1_y = np.append(ylist[39:42],ylist[36])
    AU7_1_x,AU7_1_y = linear_interpolation(AU7_1_x,AU7_1_y)
    AU_feature = AU_feature + [get_average_curvature(AU7_1_x,AU7_1_y)]

    AU7_2_x = np.append(xlist[46:48],xlist[42])
    AU7_2_y = np.append(ylist[46:48],ylist[42])
    AU7_2_x,AU7_2_y = linear_interpolation(AU7_2_x,AU7_2_y)
    AU_feature = AU_feature + [get_average_curvature(AU7_2_x,AU7_2_y)]

    AU9_x = xlist[31:36]
    AU9_y = ylist[31:36]
    AU9_x,AU9_y = linear_interpolation(AU9_x,AU9_y)
    AU_feature = AU_feature + [get_average_curvature(AU9_x,AU9_y)]

    AU10_x = np.append(xlist[48:51],xlist[52:55])
    AU10_y = np.append(ylist[48:51],ylist[52:55])
    AU10_x,AU10_y = linear_interpolation(AU10_x,AU10_y)
    AU_feature = AU_feature + [get_average_curvature(AU10_x,AU10_y)]

    AU12_1_x = [xlist[48]] + [xlist[60]] + [xlist[67]]
    AU12_1_y = [ylist[48]] + [ylist[60]] + [ylist[67]]
    AU12_1_x,AU12_1_y = linear_interpolation(AU12_1_x,AU12_1_y)
    AU_feature = AU_feature + [get_average_curvature(AU12_1_x,AU12_1_y)]

    AU12_2_x = [xlist[54]] + [xlist[64]] + [xlist[65]]
    AU12_2_y = [ylist[54]] + [ylist[64]] + [ylist[65]]
    AU12_2_x,AU12_2_y = linear_interpolation(AU12_2_x,AU12_2_y)
    AU_feature = AU_feature + [get_average_curvature(AU12_2_x,AU12_2_y)]


    AU20_x = xlist[55:60]
    AU20_y = ylist[55:60]
    AU20_x,AU20_y = linear_interpolation(AU20_x,AU20_y)
    AU_feature = AU_feature + [get_average_curvature(AU20_x,AU20_y)]

    Norm_AU_feature = (AU_feature-np.min(AU_feature))/np.ptp(AU_feature)

    return Norm_AU_feature


warnings.simplefilter('ignore', np.RankWarning) # 关闭警告：RankWarning: Polyfit may be poorly conditioned


def get_average_curvature(AU_xlist,AU_ylist):
    K = []
    Z = np.polyfit(AU_xlist,AU_ylist,4)
    P = np.poly1d(Z)
    P_1 = np.poly1d.deriv(P)
    P_2 = np.poly1d.deriv(P_1)
    for i in range(len(AU_xlist)):
        # K[i] =  P_2[AU_xlist[i]]/math.pow((1+math.pow((P_1(AU_xlist[i])),2)),1.5)
        Y = 1+math.pow(P_1(AU_xlist[i]),2)
        Y = math.pow(Y,1.5)
        # print("Y",Y)
        # print("X",P_2(AU_xlist[i]))
        K.append(P_2(AU_xlist[i])/Y)
    m_K = np.mean(K)
    return m_K


# def get_vectorized_landmark(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     clahe_image = clahe.apply(gray)
#     detections = detector(clahe_image, 1)
#     for k,d in enumerate(detections): #For all detected face instances individually
#         shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
#         xlist = []
#         ylist = []
#         for i in range(0,68): #Store X and Y coordinates in two lists
#             xlist.append(float(shape.part(i).x))
#             ylist.append(float(shape.part(i).y))
#         xmean = np.mean(xlist)
#         ymean = np.mean(ylist)
#         xcentral = [(x-xmean) for x in xlist]
#         ycentral = [(y-ymean) for y in ylist]
#         landmarks_vectorized = []
#         for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
#             landmarks_vectorized.append(w)
#             landmarks_vectorized.append(z)
#             meannp = np.asarray((ymean,xmean))
#             coornp = np.asarray((z,w))
#             dist = np.linalg.norm(coornp-meannp)
#             landmarks_vectorized.append(dist)
#             landmarks_vectorized.append((math.atan2(y, x)*360)/(2*math.pi))
#         landmarks_vectorized = landmarks_vectorized[68:]
#         landmarks_vectorized = np.array(landmarks_vectorized,dtype = np.float64)
#         return landmarks_vectorized
#     if len(detections) < 1:
#         landmarks_vectorized = np.array([])
#     return landmarks_vectorized


# this is the get vectorized landmark algorithms with cropping and resizing
def get_vectorized_landmark(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        landmarks= []
        for i in range(0,68): #Store X and Y coordinates in two lists
            cv2.circle(clahe_image, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2)
            #For each point, draw a red circle with thickness2 on the original frame
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))

        xmean = np.mean(xlist) #Find both coordinates of centre of gravity
        ymean = np.mean(ylist)
        x_max = np.max(xlist)
        x_min = np.min(xlist)
        y_max = np.max(ylist)
        y_min = np.min(ylist)
        cv2.rectangle(clahe_image,(int(x_min),int(y_min-((ymean - y_min)/3))),(int(x_max),int(y_max)),(255,150,0),2)
        # print ("centre of gravity",xmean, ymean)
        # print ("range of the face",x_max, x_min, y_max, y_min)
        cv2.circle(clahe_image, (int(xmean), int(ymean) ), 1, (0,255,255), thickness=2)

        x_start = int(x_min)
        y_start = int(y_min-((ymean - y_min)/3))
        w = int(x_max) - x_start
        h = int(y_max) - y_start

        crop_img = image[y_start:y_start+h, x_start:x_start+w] # Crop from {x, y, w, h } => {0, 0, 300, 400}

    if len(detections) > 0:
        mywidth = 255
        hsize = 255
        cv2.imwrite('crop_img.png',crop_img)
        img = Image.open('crop_img.png')
        # wpercent = (mywidth/float(img.size[0]))
        # hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((mywidth,hsize), PIL.Image.ANTIALIAS)
        img.save('resized.png')

        image_resized = cv2.imread('resized.png')
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1)
        for k,d in enumerate(detections): #For all detected face instances individually
            shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
            xlist = []
            ylist = []
            for i in range(0,68): #Store X and Y coordinates in two lists
                xlist.append(float(shape.part(i).x))
                ylist.append(float(shape.part(i).y))
            xmean = np.mean(xlist)
            ymean = np.mean(ylist)
            xcentral = [(x-xmean) for x in xlist]
            ycentral = [(y-ymean) for y in ylist]
            landmarks_vectorized = []
            for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
                landmarks_vectorized.append(w)
                landmarks_vectorized.append(z)
                meannp = np.asarray((ymean,xmean))
                coornp = np.asarray((z,w))
                dist = np.linalg.norm(coornp-meannp)
                landmarks_vectorized.append(dist)
                landmarks_vectorized.append((math.atan2(y, x)*360)/(2*math.pi))
            landmarks_vectorized = landmarks_vectorized[68:]
            landmarks_vectorized = np.array(landmarks_vectorized,dtype = np.float64)
            return landmarks_vectorized
        if len(detections) < 1:
            landmarks_vectorized = np.array([])
        return landmarks_vectorized
    else:
        landmarks_vectorized = np.array([])
    return landmarks_vectorized




def make_training_sets():
    training_data = np.array([])
    training_labels = np.array([])
    prediction_data = np.array([])
    prediction_labels = np.array([])
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_datasets(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            # clahe_image = clahe.apply(gray)
            [xlist, ylist] = get_landmarks(image)
            vec_landmark = get_vectorized_landmark(image)
            if (xlist.size) and (vec_landmark.size):
                Norm_AU_feature = extract_AU(xlist,ylist)
                vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
                training_labels = np.concatenate((training_labels,[emotions.index(emotion)]))
                if training_data.size:
                    training_data = np.vstack((training_data,vec_AU))
                    # training_data.appe nd(data['landmarks_vectorised']) #append image array to training data list
                else:
                    training_data = np.concatenate((training_data,vec_AU))
            else:
                print("no face detected on this training one")


        for item in prediction:
            image = cv2.imread(item)
            [xlist, ylist] = get_landmarks(image)
            vec_landmark = get_vectorized_landmark(image)
            if (xlist.size) and (vec_landmark.size):
                Norm_AU_feature = extract_AU(xlist,ylist)
                vec_AU = np.concatenate((Norm_AU_feature,vec_landmark))
                prediction_labels = np.concatenate((prediction_labels,[emotions.index(emotion)]))
                if prediction_data.size:
                    prediction_data = np.vstack((prediction_data,vec_AU))
                    # training_data.append(data['landmarks_vectorised']) #append image array to training data list
                else:
                    prediction_data = np.concatenate((prediction_data,vec_AU))
            else:
                print("no face detected on this prediction one")

    return training_data, training_labels, prediction_data, prediction_labels


# # 接下来的几行是单个单个读取图片的代码
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("E:\\Mutilmodal expression recognition\\shape_predictor_68_face_landmarks\\shape_predictor_68_face_landmarks.dat")
#
# img = cv2.imread(r'E:\Mutilmodal expression recognition\video\harassment 01\harassment 01_20191115_115146.633.png')
# [xlist, ylist] = get_landmarks(img)
# print(xlist)
# print(ylist)
# Norm_AU_feature = extract_AU(xlist,ylist)
# print("Norm AU Feature",Norm_AU_feature)

# 接下来的几行是读取视频逐帧特征的代码
for root, dirs, files in os.walk(path_1):
    # print("Root = ", root, "dirs = ", dirs, "files = ", files)
    files = natsorted(files)
    for i in range(len(files)):
        # 接下来的是逐帧读取视频的代码
        #
        predictor_path = r"./shape_predictor_68_face_landmarks.dat"
        # 初始化
        predictor = dlib.shape_predictor(predictor_path)

        # 初始化dlib人脸检测器
        detector = dlib.get_frontal_face_detector()

        # 初始化窗口
        # video_path = r"E:/Mutilmodal expression recognition/VideoReviews/videos/10_books.mp4"
        video_path = path_1 + files[i]

        cap = cv2.VideoCapture(video_path)
        # cap = cv2.VideoCapture(0)

        point = 1

        # with open(
        #         r"E:\Mutilmodal expression recognition\shape_predictor_68_face_landmarks\video3_fea\happy_add_fea\illness-satisfy.txt",
        #         "a", encoding="utf-8")as f:
        with open(
                path_2 + files[i][:-4] + ".txt",
                "a", encoding="utf-8")as f:
            f.write("Frame number ")
            while point <= 68:
                f.write("p" + str(point) + "X " + "p" + str(point) + "Y ")
                point = point + 1
            f.write("AU1_1 AU1_2 AU2_1 AU2_2 AU5_1 AU5_2 AU7_1 AU7_2 AU9 AU10 AU12_1 AU12_2 AU20" + "\n")

        frame = 0

        while cap.isOpened():
            ok, cv_img = cap.read()
            if not ok:
                break

            # cascade = cv2.CascadeClassifier(r"E:\Mutilmodal expression recognition\shape_predictor_68_face_landmarks\haarcascade_frontalface_default.xml")  ## 读入分类器数据
            # # faces = cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
            # if len(faces) < 0:
            #     i = i + 1
            #     print("FrameNumber: " + str(i) + "No face!")
            #     continue

            try:
                frame = frame + 1
                pt = 0
                au_number = 0
                [xlist, ylist] = get_landmarks(cv_img)
                # print(xlist)
                # print(ylist)
                Norm_AU_feature = extract_AU(xlist, ylist)
                print("No."+ str(i+1) + " " + files[i]+" FrameNumber: " + str(frame))
                # print("Norm AU Feature", Norm_AU_feature)
                # with open(
                #         r"E:\Mutilmodal expression recognition\shape_predictor_68_face_landmarks\video3_fea\happy_add_fea\illness-satisfy.txt",
                #         "a", encoding="utf-8")as f:
                with open(
                        path_2 + files[i][:-4] + ".txt",
                        "a", encoding="utf-8")as f:
                    f.write(files[i]+ ":" + str(frame) + " ")

                    while pt < 68:
                        f.write(str(float(xlist[pt])) + " " + str(float(ylist[pt])) + " ")
                        pt = pt + 1
                    while au_number < 13:
                        f.write(str(format(Norm_AU_feature[au_number], ".6f")) + " ")  # 小数点后保留6位数
                        au_number = au_number + 1
                    f.write("\n")
            except BaseException:
                with open(
                        path_2 + files[i][:-4] + ".txt",
                        "a", encoding="utf-8")as f:
                    f.write(files[i]+ ":" + str(frame) + " ")
                    for index in range(149):
                        f.write("NaN ")
                    f.write("\n")
                print("No."+ str(i+1) + " " + files[i] + " FrameNumber: " + str(frame) + " Can not detect face!")

        cap.release()


# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]  #Emotion list
# # emotions = ["anger",  "disgust",  "happy", "neutral", "sadness", "surprise"] #Emotion list
# # emotions = ["anger",  "fear",  "happy", "sadness", "contempt", "surprise"] #Emotion list
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("E:\\Mutilmodal expression recognition\\shape_predictor_68_face_landmarks\\shape_predictor_68_face_landmarks.dat")
#
# training_data = np.array([])
# training_labels = np.array([])
# prediction_data = np.array([])
# prediction_labels = np.array([])
# [training_data, training_labels, prediction_data, prediction_labels] = make_training_sets()

# np.savetxt('training_data.txt', training_data, fmt='%1.4e')
# np.savetxt('training_labels.txt', training_labels, fmt='%1.4e')
# np.savetxt('prediction_data.txt', prediction_data, fmt='%1.4e')
# np.savetxt('prediction_labels.txt', prediction_labels, fmt='%1.4e')

# clf = SVC(kernel='linear', probability=True, tol=1e-3)
# accur_lin = []
# for i in range(0,10):
#     print("Making sets %s" %i) #Make sets by random sampling 80/20%
#·····················     [training_data, training_labels, prediction_data, prediction_labels] = make_training_sets()
#     print("training SVM linear %s" %i) #train SVM
#     clf.fit(training_data, training_labels)
#     print("getting accuracies %s" %i) #Use score() function to get accuracy
#     pred_lin = clf.score(prediction_data, prediction_labels)
#     print ("linear: ", pred_lin)
#     accur_lin.append(pred_lin) #Store accuracy in a list
#
# np.savetxt('accuracy.txt', accur_lin, fmt='%1.4e')
# print("Mean value lin svm: %s" %np.mean(accur_lin)) #FGet mean accuracy of the 10 runs


