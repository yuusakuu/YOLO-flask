import flask
from flask import Flask, request, render_template, Response, redirect, url_for
import joblib
import numpy as np
from scipy import misc
from flask_restful import Resource, Api
import imageio
# ------------------------
from datetime import datetime

# from domain.question.question_schema import QuestionCreate
# from models import Question
# from sqlalchemy.orm import Session

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
# ----------------------


app = Flask(__name__)
api = Api(app)


# 메인 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

 
# 사진 결과 조회
@app.route('/predict', methods=['POST'])
def prediction(): #def prediction(file: UploadFile = File(...)):
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
            # filename2 = filename.split(' ')[1][1:-5]
            pathname = 'uploads/'+filename.split(' ')[1][1:-1]
            # print('filename:', filename2)
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
        # print('img_mask')
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
        # output_image = plot_boxes(original_image, boxes, class_names, plot_labels = True)
        

        # class_names
        # bbox, label, conf = cv.detect_common_objects(image, model=model)
        
        # Create image that includes bounding boxes and labels
        # output_image = draw_bbox(image, bbox, label, conf)
        
        # Save it in a folder within the server
        # cv2.imwrite(f'images_uploaded/{filename}', output_image)
        
        
        # 4. STREAM THE RESPONSE BACK TO THE CLIENT
        
        # Open the saved image for reading in binary mode
        # file_image = open(f'images_uploaded/{filename}', mode="rb")
        
        # Return the image as a stream specifying media type
        #return obj
        return render_template('predict.html', ml_label=obj, print_image = pathname )
        # return StreamingResponse(file_image, media_type="image/jpeg")


# 웹캠 결과 조회
@app.route('/cam_post')
def cam_load():
    #f = open('static/cam_label/label.txt', 'r')
    #label = f.read()
    #return render_template('cam.html', label = label)
    return render_template('cam.html')

def gen_frames(CamSignal):
    while True:
        _, frame = CamSignal.read()
        YOLO_net = cv2.dnn.readNet(weight_file, cfg_file)
        classes = []
        with open(namesfile, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = YOLO_net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

        if not _:
            break
        else:
            i, frame2 = CamSignal.read()
            bgrLower = np.array([0, 0, 0])    # 추출할 색의 하한(BGR)
            bgrUpper = np.array([100, 100, 100])    # 추출할 색의 상한(BGR)
            img_mask = cv2.inRange(frame, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성, 범위에 들면 255할당, 아니면 0을 할당
            # print('img_mask')
            black = invert(img_mask) # invert를 통해 글자를 검은색으로 배경을 흰 색으로 전환
            # print('black')
            frame[black>0]=(255,255,255)
            h, w, c = frame.shape
            # print(h, w, c)
            
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
            #plt.savefig('static/uploads/savefig_default.png')
            #results = Image.open('./static/uploads/savefig_default.png')
            
            # annotated_frame = results.render()
            
            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()
            filewrite(label_list)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')


@app.route('/cam')
def cam_post():
    CamSignal = cv2.VideoCapture(0) #-- 웹캠 사용시 video_path를 0 으로 변경
    # gen_frames2(CamSignal)
    '''return_value, frame = CamSignal.read()
    if return_value:
        pass
    else:
        raise ValueError("No image!")'''
    return Response(gen_frames(CamSignal), mimetype='multipart/x-mixed-replace; boundary=frame')


# 4번 메뉴
'''@app.route('/video', methods=['POST'])
def video_load():
    if request.method == 'POST':
        file3 = request.files['video2']
        #if not file2: 
        #    return render_template('index.html', ml_label2="No Files")
        file3.save('static/vid_uploads/' + secure_filename(file3.filename))
        with tempfile.TemporaryDirectory() as td:
            temp_filename = Path(td) / 'vid_uploads'
            file3.save(temp_filename)
            filename = str(file3)
        #with tempfile.TemporaryDirectory() as td:
        #    temp_filename = Path(td) / 'uploaded_video'
        #    file2.save(temp_filename)

        # file2.save(secure_filename(file2.filename))
        
        # VideoSignal = imageio.imread(file2)
        # 웹캠 신호 받기
        # filename = str(file3).split(' ')[1][1:-1]
            # VideoSignal = cv2.VideoCapture('static/uploads/'+filename)
            print(filename)
            pathname = 'vid_uploads/'+filename.split(' ')[1][1:-1]
            
        return render_template('video.html', videopath = pathname)'''

@app.route('/video')
def video_load():
    filename = 'video2.mp4'
    pathname = 'vid_uploads/'+filename
        
    return render_template('video.html', videopath = pathname)

# 동영상 결과 조회
@app.route('/video_post')
def video_load2():
    #f = open('static/label/label.txt', 'r')
    #label = f.read()
    #return render_template('video2.html', label = label)
    return render_template('video.html')

def gen_frames(VideoSignal):
    while True:
        _, frame = VideoSignal.read()
        YOLO_net = cv2.dnn.readNet(weight_file, cfg_file)
        classes = []
        with open(namesfile, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = YOLO_net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

        if not _:
            break
        else:
            i, frame2 = VideoSignal.read()
            bgrLower = np.array([0, 0, 0])    # 추출할 색의 하한(BGR)
            bgrUpper = np.array([100, 100, 100])    # 추출할 색의 상한(BGR)
            img_mask = cv2.inRange(frame, bgrLower, bgrUpper) # BGR로 부터 마스크를 작성, 범위에 들면 255할당, 아니면 0을 할당
            # print('img_mask')
            black = invert(img_mask) # invert를 통해 글자를 검은색으로 배경을 흰 색으로 전환
            # print('black')
            frame[black>0]=(255,255,255)
            h, w, c = frame.shape
            # print(h, w, c)
            
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
            #plt.savefig('static/uploads/savefig_default.png')
            #results = Image.open('./static/uploads/savefig_default.png')
            
            # annotated_frame = results.render()
            
            ret, buffer = cv2.imencode('.jpg', frame2)
            frame2 = buffer.tobytes()
            filewrite(label_list)
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')        


@app.route('/video2', methods = ['POST'])
def video_posting():
    if request.method == 'POST':
        file3 = request.files['stream']
        #if not file2: 
        #    return render_template('index.html', ml_label2="No Files")
        file3.save('static/vid_uploads/' + secure_filename(file3.filename))
        filename = str(file3).split(' ')[1][1:-1]
        txt = open('static/video_name/video_name.txt', 'w+')
        txt.write(filename)
        txt.close()

        return render_template('index.html')
        #return Response(gen_frames(VideoSignal), mimetype='multipart/x-mixed-replace; boundary=frame')


# video_posting에서 filename을 받아오고 video_show()에서 filename을 변수로 입력받으면 사용가능할듯
@app.route('/video3')
def video_show():
    file = open('static/video_name/video_name.txt', 'r')
    if file.mode == 'r':
        filename = file.read()
    # filename = 'video2.mp4'
    VideoSignal = cv2.VideoCapture('static/vid_uploads/'+filename) #-- 웹캠 사용시 video_path를 0 으로 변경
    return Response(gen_frames(VideoSignal), mimetype='multipart/x-mixed-replace; boundary=frame')


def filewrite(label):
    with open('static/label/label.txt', 'w+') as file:
        file.write('\n'.join(label))


# 2. 출력한 값 가져와서 화면에 띄우기 -- 해봐야 안다
# 3. 문제 출제하고 맞추면 정답 화면 띄우기
# 웹캠으로 사용하기에는 속도 문제가 있을 것 같은데.. 해결? 모델을 v7으로 바꿔야 하나 

# 모바일 카메라를 이용하여 실시간 이미지 인식하기 (videocapture 함수가 안먹는건가?)


if __name__ == '__main__':
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=8000, debug=True)
