import json
import base64
import cv2
import dlib
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import logging as log
import datetime as dt

# Initialize dlib's face detector and facial landmarks predictor
class Detector_eyes(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize dlib's face detector and facial landmarks predictor as instance variables
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('IMED/dat/shape_predictor_68_face_landmarks.dat')

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processed_frame = self.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await self.send(text_data=json.dumps({'frame': jpg_as_text}))

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:
            print("No faces detected")

        for face in faces:
            landmarks = self.predictor(gray, face)
            for i in range(2):
                start, end = (36, 42) if i == 0 else (42, 48)
                eye_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(start, end)]
                eye_bounds = cv2.boundingRect(np.array(eye_points))
                x, y, w, h = eye_bounds
                eye_roi = frame[y:y+h, x:x+w]
                gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
                blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
                _, threshold_eye = cv2.threshold(blurred_eye, 70, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)
                    M = cv2.moments(max_contour)
                    if M['m00'] != 0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        cv2.circle(eye_roi, (cx, cy), 2, (0, 255, 0), -1)

        return frame


class SimpleEchoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        # Receive a message from the WebSocket
        try:
            text_data_json = json.loads(text_data)
            message = text_data_json['message']
            
            if message == "stop":
                await self.close()  # Close the WebSocket connection
            else:
                # Echo the received message back to the client
                await self.send(text_data=json.dumps({
                    'message': "helo my nighu"
                }))
        except json.JSONDecodeError:
            await self.send(text_data=json.dumps({
                    'message': "helo my nighu"
                }))
            # Handle invalid JSON gracefully (you can log or handle this as needed)
            pass



class Distance(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize Haar Cascade classifier
        self.faceCascade = cv2.CascadeClassifier("IMED/dat/haarcascade_frontalface_default.xml")
        # Logging setup
        log.basicConfig(filename='webcam.log', level=log.INFO)
        # Known distances and widths (for distance calculation)
        self.known_distance1 = 4.3
        self.known_width1 = 48
        self.focalLength = self.known_distance1 * self.known_width1
        self.anterior = 0

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processed_frame = self.process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed_frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await self.send(text_data=json.dumps({'frame': jpg_as_text}))

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if self.anterior != len(faces):
            self.anterior = len(faces)
            log.info("faces: " + str(len(faces)) + " at " + str(dt.datetime.now()))

        if len(faces) > 0:
            distance = self.focalLength / faces[0][2]
            cv2.putText(frame, "%.2fM" % distance,
                        (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (0, 255, 0), 2)

        return frame
    