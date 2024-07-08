import cv2
import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse

app = FastAPI(debug=True)


def frames():
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        yield frame


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    for frame in frames():
        await websocket.send_bytes(cv2.imencode('.jpg', frame)[1].tobytes())

if __name__ == '__main__':
    uvicorn.run(app,  host="127.0.0.1", port=8000)
