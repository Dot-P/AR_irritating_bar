import base64
import time
import cv2
from fastapi import Response
from nicegui import app, ui
from pydantic import BaseModel

# 黒い1px画像をBase64エンコードしたデータ（プレースホルダー画像として使用）
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')  # FastAPIのレスポンスとして返す

# ゲームオブジェクトの定義
class GameObject(BaseModel):
    start: bool = False  # ゲームの開始フラグ
    end: bool = False    # ゲームの終了フラグ
    score: int = 0       # ゲームスコア
    message: str = "Ready"  # 表示メッセージ

def update_game_state(start=None, end=None, score=None, message=None):
    """
    ゲームオブジェクトの状態を更新します。
    """
    if start is not None:
        game_object.start = start
    if end is not None:
        game_object.end = end
    if score is not None:
        game_object.score = score
    if message is not None:
        game_object.message = message

# `/video/frame`エンドポイントを定義
@app.get('/video/frame')
async def grab_video_frame() -> Response:
    """
    カメラから1フレームを取得し、ゲームロジックを適用後、JPEG画像として返す。
    カメラが利用できない場合、プレースホルダー画像を返す。
    """
    video_capture = cv2.VideoCapture(0)  # デフォルトのカメラ（ID: 0）を開く
    if not video_capture.isOpened():
        print("カメラが開けません")
        return placeholder
    ret, frame = video_capture.read()
    video_capture.release()
    if not ret:
        return placeholder
    
    # TODO: ARマーカーの表示

    # TODO: ゲームロジックの更新

    # ゲームロジックを適用
    if game_object.start:
        cv2.putText(frame, f"Score: {game_object.score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, game_object.message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Game Not Started", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    _, imencode_image = cv2.imencode('.jpg', frame)
    jpeg = imencode_image.tobytes()
    return Response(content=jpeg, media_type='image/jpeg')

# 初期状態のゲームオブジェクト
game_object = GameObject()

# NiceGUIのインタラクティブな画像コンポーネントを作成
video_image = ui.interactive_image().classes('w-full h-full')

# ゲーム制御用のUIを追加
with ui.row():
    ui.button('Start Game', on_click=lambda: update_game_state(start=True, end=False, score=0))
    ui.button('Stop Game', on_click=lambda: update_game_state(start=False, end=True))
    ui.button('Add Score', on_click=lambda: update_game_state(score=game_object.score + 1))

# タイマーで動画を更新
ui.timer(
    interval=0.1,  # 100msごとに更新
    callback=lambda: video_image.set_source(f'/video/frame?{time.time()}')  # `/video/frame`エンドポイントをポーリング
)

# NiceGUIアプリケーションを起動
ui.run()
