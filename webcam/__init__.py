import cv2
import mediapipe as mp
import uvicorn
from fastapi import FastAPI, WebSocket
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

face_detector = vision.FaceDetector.create_from_options(
    vision.FaceDetectorOptions(
        base_options=python.BaseOptions('detector.tflite'),
        min_detection_confidence=0.5))

fld_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            'face_landmarker_v2_with_blendshapes.task'),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1))

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

app = FastAPI(debug=True)


def frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        h, w, *_ = frame.shape
        h, w = h//2, w//2
        frame = cv2.resize(frame, (w, h))
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame)

        res = face_detector.detect(mp_frame)
        if res.detections:
            bbox = res.detections[0].bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
            face = frame[y:y+h, x:x+w, :].astype('uint8')

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            face = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=face)

            res = fld_detector.detect(face)
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
