import cv2
import mediapipe as mp
import uvicorn
from fastapi import FastAPI, WebSocket
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions('face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1)

fld_detector = vision.FaceLandmarker.create_from_options(options)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app = FastAPI(debug=True)


def frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        h, w, *_ = frame.shape
        h, w = h//2, w//2
        frame = cv2.resize(frame, (w, h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            face = frame[y:y+h, x:x+w, :].astype('uint8')

            img = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=face)

            res = fld_detector.detect(img)
            if res.face_landmarks:
                for landmarks in res.face_landmarks:
                    for landmark in landmarks:
                        fx, fy = landmark.x, landmark.y
                        fx, fy = int(fx*w), int(fy*h)
                        cv2.circle(frame, (x + fx, y + fy), 2, (0, 255, 0), -1)

        yield frame


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    for frame in frames():
        await websocket.send_bytes(cv2.imencode('.jpg', frame)[1].tobytes())

if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
