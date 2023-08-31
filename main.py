import flask
from flask import Flask, request, render_template, Response, redirect, url_for, jsonify
import joblib
import numpy as np
from scipy import misc
from flask_restful import Resource, Api
import imageio

from datetime import datetime

import cv2
import matplotlib.pyplot as plt

from utils import *
from darknet import Darknet

import numpy as np
import os
from skimage.util import invert

from pathlib import Path
import tempfile
from werkzeug.utils import secure_filename
from PIL import Image
import re

# Load Model File
# Set the location and name of the cfg file
cfg_file = './cfg/detect.cfg'

# Set the location and name of the pre-trained weights file
weight_file = './weights/detect_last.weights'

# Set the location and name of the COCO object classes file
namesfile = 'data/object.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

def filewrite(label):
    with open('static/label/label.txt', 'w+') as file:
        file.write(' '.join(label))

def filewrite2(label):
    with open('static/label/label2.txt', 'w+') as file:
        file.write(' '.join(label))

# Flask
app = Flask(__name__)
api = Api(app)

# 메인 페이지 라우팅
@app.route("/")

@app.route("/index")
def index():
    return flask.render_template('index.html')

# 1. 사진 결과 조회
@app.route('/predict', methods=['POST'])
def prediction(): 
    if request.method == 'POST':

        # 업로드 파일 처리 분기
        file = request.files['image']
        file.save('static/uploads/' + secure_filename(file.filename))
        if not file: 
            return render_template('index.html', ml_label="No Files")
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'uploads'
            file.save(temp_filename)
            filename = str(file)
            pathname = 'uploads/'+filename.split(' ')[1][1:-1]
        img = imageio.imread(file)

        imgCropped = img[0:800,0:800] # height는 0-800, width는 0-1200으로 해서 자른다.
        imgResize = cv2.resize(imgCropped,(5000,5000))  # width, height 순서로 표현한다.
        img = imgResize
        # Convert the image to RGB
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # We resize the image to the input width and height of the first layer of the network.    
        resized_image = cv2.resize(original_image, (m.width, m.height))
        
        bgrLower = np.array([0, 0, 0])    # 추출할 색의 하한(BGR)
        bgrUpper = np.array([100, 100, 100])    # 추출할 색의 상한(BGR)
        img_mask = cv2.inRange(resized_image, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성, 범위에 들면 255할당, 아니면 0을 할당
        black = invert(img_mask) # invert를 통해 글자를 검은색으로 배경을 흰 색으로 전환
        resized_image[black>0]=(255,255,255)
        
        # 3. RUN OBJECT DETECTION MODEL
        
        # Set the IOU threshold. Default value is 0.4
        iou_thresh = 0.5

        # Set the NMS threshold. Default value is 0.6
        nms_thresh = 0.5

        # Detect objects in the image
        boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)

        # Print the objects found and the confidence level
        obj = objects_info(boxes, class_names)

        #Plot the image with bounding boxes and corresponding object class labels
        plot_boxes(original_image, boxes, class_names, plot_labels = True)
        
        return render_template('predict.html', ml_label=obj, print_image = pathname )


# 2. 동영상 결과 조회
@app.route('/video_post')
def video_load2():
    return render_template('video.html')

# gen_frame
def gen_frames(VideoSignal):
    YOLO_net = cv2.dnn.readNet(weight_file, cfg_file)
    classes = []
    with open(namesfile, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    while True:
        _, frame = VideoSignal.read()
        if not _:
            break
        else:
            i, frame2 = VideoSignal.read()

            bgrLower = np.array([0, 0, 0])    # 추출할 색의 하한(BGR)
            bgrUpper = np.array([100, 100, 100])    # 추출할 색의 상한(BGR)
            img_mask = cv2.inRange(frame, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성, 범위에 들면 255할당, 아니면 0을 할당
            black = invert(img_mask) # invert를 통해 글자를 검은색으로 배경을 흰 색으로 전환
            frame[black>0]=(255,255,255)
            h, w, c = frame.shape
            
            # YOLO 입력
            # 창 크기 설정
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            YOLO_net.setInput(blob)
            outs = YOLO_net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:

                for detection in out:

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.4:
                        # Object detected
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        dw = int(detection[2] * w)
                        dh = int(detection[3] * h)
                        # Rectangle coordinate
                        x = int(center_x - dw / 2)
                        y = int(center_y - dh / 2)
                        boxes.append([x, y, dw, dh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

            label_list = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    score = confidences[i]

                    # 경계상자와 클래스 정보 이미지에 입력
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    cv2.putText(frame2, label, (x, y - 20), cv2.FONT_ITALIC, 2, 
                    (255, 0, 0), 10)

                    label_list.append(label)
            
            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()
            # filewrite(label_list)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')        

@app.route('/video2', methods = ['POST'])
def video_posting():
    if request.method == 'POST':
        file3 = request.files['stream']
        file3.save('static/vid_uploads/' + secure_filename(file3.filename))
        filename = str(file3).split(' ')[1][1:-1]
        txt = open('static/video_name/video_name.txt', 'w+')
        txt.write(filename)
        txt.close()

        return render_template('index.html')

@app.route('/video3')
def video_show():
    file = open('static/video_name/video_name.txt', 'r')
    if file.mode == 'r':
        filename = file.read()
    VideoSignal = cv2.VideoCapture('static/vid_uploads/'+filename) #-- 웹캠 사용시 video_path를 0 으로 변경
    return Response(gen_frames(VideoSignal), mimetype='multipart/x-mixed-replace; boundary=frame')


# 3. 웹캠 결과 조회
@app.route('/cam_post')
def cam_load():
    return render_template('cam.html')

# gen_frame
def gen_frames(CamSignal):
    YOLO_net = cv2.dnn.readNet(weight_file, cfg_file)
    classes = []
    with open(namesfile, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    while True:
        _, frame = CamSignal.read()
        if not _:
            break
        else:
            i, frame2 = CamSignal.read()
            bgrLower = np.array([0, 0, 0])    # 추출할 색의 하한(BGR)
            bgrUpper = np.array([100, 100, 100])    # 추출할 색의 상한(BGR)
            img_mask = cv2.inRange(frame, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성, 범위에 들면 255할당, 아니면 0을 할당
            black = invert(img_mask) # invert를 통해 글자를 검은색으로 배경을 흰 색으로 전환
            frame[black>0]=(255,255,255)
            h, w, c = frame.shape
            
            # YOLO 입력
            # 창 크기 설정
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            YOLO_net.setInput(blob)
            outs = YOLO_net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:

                for detection in out:

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.4:
                        # Object detected
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        dw = int(detection[2] * w)
                        dh = int(detection[3] * h)
                        # Rectangle coordinate
                        x = int(center_x - dw / 2)
                        y = int(center_y - dh / 2)
                        boxes.append([x, y, dw, dh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)

            label_list = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    score = confidences[i]

                    # 경계상자와 클래스 정보 이미지에 입력
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    cv2.putText(frame2, label, (x, y - 20), cv2.FONT_ITALIC, 2, 
                    (255, 0, 0), 10)

                    label_list.append(label)
            
            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()
            filewrite(label_list)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

@app.route('/cam')
def cam_post():
    CamSignal = cv2.VideoCapture(0) #-- 웹캠 사용시 video_path를 0 으로 변경
    return Response(gen_frames(CamSignal), mimetype='multipart/x-mixed-replace; boundary=frame')


# 4. 문제 출제 
@app.route('/q_cam_post')
def q_cam_load(): 
    args_dict = request.args.to_dict()
    # print(args_dict)
    # db에서 문제 받아오기로 변경할 것
    quest = list(args_dict.keys())
    quest2 =  ' '.join(str(s) for s in quest)
    new_str = re.sub(r"[^a-zA-Z]", "", quest2)
    filewrite(new_str)
    return render_template('q_cam.html', ml_label = quest2)
    # return render_template('q_cam.html', ml_label = quest2 )

def q_gen_frames(CamSignal):
    YOLO_net = cv2.dnn.readNet(weight_file, cfg_file)
    classes = []
    with open(namesfile, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = YOLO_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]
    f = open('static/label/label.txt', 'r')
    q_label = f.read()
    q_label2 = q_label.split(' ')
    s_q_label3 = sorted(q_label2)
    filewrite2(' ')
    while True:
        _, frame = CamSignal.read()
        if not _:
            break
        else:
            i, frame2 = CamSignal.read()
            bgrLower = np.array([0, 0, 0])    # 추출할 색의 하한(BGR)
            bgrUpper = np.array([100, 100, 100])    # 추출할 색의 상한(BGR)
            img_mask = cv2.inRange(frame, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성, 범위에 들면 255할당, 아니면 0을 할당
            black = invert(img_mask) # invert를 통해 글자를 검은색으로 배경을 흰 색으로 전환
            frame[black>0]=(255,255,255)
            h, w, c = frame.shape
            
            # YOLO 입력
            # 창 크기 설정
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            YOLO_net.setInput(blob)
            outs = YOLO_net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:

                for detection in out:

                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.4:
                        # Object detected
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        dw = int(detection[2] * w)
                        dh = int(detection[3] * h)
                        # Rectangle coordinate
                        x = int(center_x - dw / 2)
                        y = int(center_y - dh / 2)
                        boxes.append([x, y, dw, dh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
            label_list = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    score = confidences[i]
                    # 경계상자와 클래스 정보 이미지에 입력
                    cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 0, 255), 5)
                    cv2.putText(frame2, label, (x, y - 20), cv2.FONT_ITALIC, 2, 
                    (255, 0, 0), 10)
                    label_list.append(label)
            cnt = 0
            s_label_list= sorted(label_list)
            if len(label_list) == len(q_label2):
                for i in range(len(label_list)):
                    if s_label_list[i] == s_q_label3[i]:
                        cnt += 1
                if cnt == len(q_label2):
                    cv2.imwrite('static/correct/frame.png', frame2)
                    filewrite2(label_list)

            cv2.imwrite('static/correct/frame_false.png', frame2)
            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')
    
@app.route('/q_cam')
def q_cam_post():
    CamSignal = cv2.VideoCapture(0) #-- 웹캠 사용시 video_path를 0 으로 변경
    #CamSignal = VideoCamera()
    return Response(q_gen_frames(CamSignal), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/q_cam_post_result')
def q_cam_load_result(): 
    with app.app_context(), app.test_request_context():
        f = open('static/label/label.txt', 'r')
        q_label = f.read()
        q_label2 = q_label.split(' ')
        s_q_label = sorted(q_label2)
        f2 = open('static/label/label2.txt', 'r')
        label = f2.read() 
        label2 = label.split(' ')
        s_label = sorted(label2)
        cnt = 0
        if len(s_q_label) == len(s_label):
            for i in range(len(s_q_label)):
                if s_label[i] == s_q_label[i] :
                    cnt += 1
                else : 
                    pass
        if cnt == len(s_q_label):
            TF = 1
        else : 
            TF = 0
        return render_template('q_cam_result.html', ml_label = q_label2, label = label2, TF = TF)

@app.route('/q_cam_post_correct')
def q_cam_load_correct(): 
    with app.app_context(), app.test_request_context():
        f = open('static/label/label.txt', 'r')
        q_label = f.read()
        q_label2 = q_label.split(' ')
        return render_template('q_cam_correct.html', ml_label = q_label2)

@app.route('/q_cam_post_false')
def q_cam_load_false(): 
    with app.app_context(), app.test_request_context():
        f = open('static/label/label.txt', 'r')
        q_label = f.read()
        q_label2 = q_label.split(' ')
        return render_template('q_cam_false.html', ml_label = q_label2)

@app.route('/question_list')
def question(): 
    question = open('question_list/question_list.txt', 'r')
    question_list = []
    
    while True :
        questions = question.readline()
        if not questions : break
        question_list.append(questions)
        
    question.close()
    # print(question_list)
    return render_template('question_list.html', question_list = question_list)

if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
