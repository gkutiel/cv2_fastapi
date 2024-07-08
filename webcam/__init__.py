import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI(debug=True)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        yield frame


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    for frame in frames():
        await websocket.send_bytes(cv2.imencode('.jpg', frame)[1].tobytes())

if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
