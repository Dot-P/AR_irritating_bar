
import base64
import time
import os
if os.name == 'nt':
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from fastapi import Response
from nicegui import app, ui
from pydantic import BaseModel
# ARマーカー
import cv2.aruco as aruco
import numpy as np

from game import Game, GameState

# 黒い1px画像をBase64エンコードしたデータ（プレースホルダー画像として使用）
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')  # FastAPIのレスポンスとして返す

ui.add_css('''
    .red {
        color: red;
    }
    .green {
        color: green;
    }
''')

# 投影するイライラ棒画像
overlay_image = cv2.imread("irritating_bar.jpg")

# ArUco辞書と検出器を初期化
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = aruco.ArucoDetector(aruco_dict)

# ビデオキャプチャオブジェクトの初期化 (on Windows)
if os.name == 'nt':
    global video_capture
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("カメラが開けません")
        video_capture = None  # カメラが開けない場合

global game
game = Game()

def format_time(seconds):
    """
    秒数を分:秒の形式に変換します。
    """
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{int(minutes)}:{int(seconds):02d}"

def update_ui():
    """
    UIを更新する関数。
    """
    global game
    # タイム表示の更新
    if game.state == GameState.PLAY:
        time_display = format_time(game.timemanager.time())
        time_label.set_text(f"Time: {time_display}")
    else:
        time_label.set_text("")

    # メッセージ表示の更新
    if game.state == GameState.READY:
        message_label.set_text("READY TO PLAY")
        message_label.classes('red')
    elif game.state == GameState.CLEAR:
        clear_time = format_time(game.timemanager.time())
        message_label.set_text(f"Game CLEAR! Clear Time: {clear_time}")
        message_label.classes('green')
    elif game.state == GameState.GAME_OVER:
        message_label.set_text("Game Over")
        message_label.classes('red')
    else:
        message_label.set_text("")

def reset_game():
    """
    ゲームをリセットする関数。
    """
    game.reset()
    time_label.set_text("")

# `/video/frame`エンドポイントを定義
@app.get('/video/frame')
async def grab_video_frame() -> Response:
    """
    カメラから1フレームを取得し、ゲームロジックを適用後、JPEG画像として返す。
    カメラが利用できない場合、プレースホルダー画像を返す。
    """
    global video_capture
    # ビデオキャプチャオブジェクトの初期化 (on Linux or Mac)
    if os.name == 'posix':
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("カメラが開けません")
            video_capture = None  # カメラが開けない場合
    if video_capture is None:
        return placeholder
    ret, frame = video_capture.read()
    if not ret:
        return placeholder

    # TODO: ARマーカーの表示

    # ARマーカーの検出
    corners, _, _ = aruco_detector.detectMarkers(frame)

    if len(corners) > 0:
        # カメラ行列とレンズ歪みの設定
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))  # レンズ歪みなしの設定

        for i, corner in enumerate(corners):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corner, 0.05, camera_matrix, dist_coeffs)

            # 四角形の頂点（ARマーカーを囲む形）
            square = np.float32([
                [0.1, 0.1, 0],
                [-0.1, 0.1, 0],
                [-0.1, -0.1, 0],
                [0.1, -0.1, 0]
            ])

            # 投影座標に変換
            imgpts, _ = cv2.projectPoints(square, rvecs, tvecs, camera_matrix, dist_coeffs)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # 画像の領域をARマーカーの領域に変換
            h, w, _ = overlay_image.shape
            src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
            dst_pts = np.array([imgpts[1],imgpts[0],imgpts[3],imgpts[2]], dtype="float32")

            # 変換行列を計算
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # 投影画像を変換して描画
            warped_image = cv2.warpPerspective(overlay_image, matrix, (frame.shape[1], frame.shape[0]))

            # 合成用のマスクを作成
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, imgpts, 255)
            mask_inv = cv2.bitwise_not(mask)

            # 元の画像に投影画像を合成
            img_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            img_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)
            frame = cv2.add(img_bg, img_fg)

    global game
    # TODO: ゲームロジックの更新
    frame = game.rogic(frame)

    # タイマーストップ判定
    if game.state == GameState.CLEAR or game.state == GameState.GAME_OVER:
        game.timemanager.finish_measure()

    # UIを更新
    update_ui()

    _, imencode_image = cv2.imencode('.jpg', frame)
    jpeg = imencode_image.tobytes()
    return Response(content=jpeg, media_type='image/jpeg')

# 親コンテナ
with ui.column().classes('items-center').style('width: 400px; margin: 0 auto;'):
    # タイム表示用ラベル
    time_label = ui.label().classes('text-xl font-bold').style('position: absolute; top: 10px; left: 10px; right: 10px;')
    # NiceGUIのインタラクティブな画像コンポーネントを作成
    video_image = ui.interactive_image().classes('w-full h-full').style('margin-top: 30px; margin-bottom: 20px; z-index: 5;')
    # メッセージ表示用ラベル
    message_label = ui.label().classes('text-xl font-bold').style('margin-top: -20px;')
    # リセットボタン
    reset_button = ui.button("リセット").classes('w-32 mt-2').style('color: white; font-weight: bold; border-radius: 5px; padding: 10px;').on('click', lambda: reset_game())

# タイマーで動画を更新
ui.timer(
    interval=0.1,  # 100msごとに更新
    callback=lambda: video_image.set_source(f'/video/frame?{time.time()}')  # `/video/frame`エンドポイントをポーリング
)

# アプリケーション終了時にリソースを解放
@app.on_event("shutdown")
async def shutdown_event():
    if video_capture is not None:
        video_capture.release()
        cv2.destroyAllWindows()

# NiceGUIアプリケーションを起動
ui.run(port=8100)
