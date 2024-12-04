import base64
import time

import cv2
from fastapi import Response

from nicegui import app, ui


black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')


@app.get('/video/frame')
async def grab_video_frame() -> Response:
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("カメラが開けません")
        return placeholder
    ret, frame = video_capture.read()
    if not ret:
        return placeholder
    _, imencode_image = cv2.imencode('.jpg', frame)
    jpeg = imencode_image.tobytes()
    return Response(content=jpeg, media_type='image/jpeg')

video_image = ui.interactive_image().classes('w-full h-full')
ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))

ui.run()
