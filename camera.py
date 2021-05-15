import cv2
import numpy as np
import dlib
from django.shortcuts import render
from math import atan
from keras.models import load_model
from django.http import StreamingHttpResponse



def calculate_scale(landmarks):
    return distance(landmarks.part(27),landmarks.part(31))
def distance(p1,p2):
#     print((p1.x - p2.x)**2 - (p1.y - p2.y)**2)
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

glob = {'angle':0}



class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        self.predictor = dlib.shape_predictor(predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.video = cv2.VideoCapture(0)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.yawn_frames = 0
        self.yawn_alert = 0
        self.eyes_closed_frames = 0
        self.eyes_closed_alert = 0
        self.checkpoint1 = 0
        self.checkpoint2 = 0
        self.classes = [' Safe driving',' Texting (right hand)','Talking on the phone (right hand).','Texting (left hand).',
                        'Talking on the phone (left hand).','Operating the radio.','Drinking.',
                        'Reaching behind.','Hair and makeup.','Talking to passenger(s)']
        self.model = load_model('Distracted_final.hdf5')
        # self.yawn_alert = False
        # self.eyes_closed_alert = False
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        image = cv2.flip(image,1)
        dets = self.detector(image)
        # print(self.yawn_frames)
        if self.checkpoint1 >= self.fps * 2:
            if self.yawn_frames >=int(self.fps/2):
                self.yawn_alert = int(self.fps / 2) 
                
            self.checkpoint1= 0
            self.yawn_frames = 0

        if self.checkpoint2 >= int(self.fps / 2):
            if self.eyes_closed_frames >=  int(self.fps / 4):
                self.eyes_closed_alert = int(self.fps / 4) 
                
            self.checkpoint2= 0
            self.eyes_closed_frames = 0
    

        self.checkpoint1 += 1
        self.checkpoint2 += 1

        


        if len(dets)>0:
            for k, d in enumerate(dets):
                    shape = self.predictor(image,d)
                    SCALE = calculate_scale(shape)
                    mouth_open_check = distance(shape.part(62),shape.part(67))
                    right_eye_open_check = distance(shape.part(43),shape.part(47))
                    left_eye_open_check = distance(shape.part(37),shape.part(41))
                    tlx = d.tl_corner().x
                    tly = d.tl_corner().y 
                    brx = d.br_corner().x 
                    bry = d.br_corner().y
                    trx = d.tr_corner().x
                    trcy = d.tr_corner().y 
                    midy = int((trcy + bry) /2)
            offset = 10
            cv2.rectangle(image,(tlx,tly),(brx,bry),(255,255,255),1)

            if mouth_open_check > (SCALE/4):
                self.yawn_frames += 1

            if self.yawn_alert >  0 :
                cv2.putText(image,"Yawn alert",(trx + offset,trcy),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2) 
                self.yawn_alert -= 1
                glob['yawn_alert' ]= 1
            else :
                glob['yawn_alert' ]= 0

                
            if self.eyes_closed_alert >  0 :
                cv2.putText(image,"Eyes Closed Alert",(trx + offset,midy),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2) 
                self.eyes_closed_alert -= 1
                glob['eyes_closed_alert'] = 1
            else: 
                glob['eyes_closed_alert'] = 0



            #Tilt :
            x1,y1 = (shape.part(39).x,shape.part(39).y)
            x2,y2 = (shape.part(42).x,shape.part(42).y)
            cv2.line(image,(shape.part(39).x,shape.part(39).y), (shape.part(42).x,shape.part(42).y),(255,25,2),1)
            head_angle = np.floor(180 * atan(((y2-y1) / (x2 - x1))))
            glob['angle']= head_angle
            cv2.putText(image,"Angle: " + str(head_angle),(brx + offset,bry),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            if right_eye_open_check < (SCALE/8) and left_eye_open_check < (SCALE/8) :
                self.eyes_closed_frames += 1

            #Predicting Distraction class type :
            pred_img = cv2.resize(image,(64,64))
            pred_img = np.expand_dims(pred_img,0)

            pred = (self.classes[self.model.predict_classes(pred_img)[0]])
            cv2.putText(image,"Type: " + str(pred),(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def webcam_feed(request):
    return (    StreamingHttpResponse(gen(VideoCamera()),
                                                     content_type='multipart/x-mixed-replace; boundary=frame'))



def index(request):
    return render(request, 'index.html')

