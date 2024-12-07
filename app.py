
import base64
import time
import os
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

# 投影するイライラ棒画像
overlay_image = cv2.imread("irritating_bar.jpg")

# ArUco辞書と検出器を初期化
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = aruco.ArucoDetector(aruco_dict)

# ビデオキャプチャオブジェクトの初期化 (on Windows)
if os.name == 'nt':
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("カメラが開けません")
        video_capture = None  # カメラが開けない場合

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
    # タイム表示の更新
    if game.state == GameState.START:
        time_display = format_time(game.timemanager.past_time())
        time_label.set_text(f"Time: {time_display}")
    else:
        time_label.set_text("")

    # メッセージ表示の更新
    if game.state == GameState.READY:
        message_label.set_text("Game Not Started")
    elif game.state == GameState.END:
        message_label.set_text("Game END")
    else:
        message_label.set_text("")

# `/video/frame`エンドポイントを定義
@app.get('/video/frame')
async def grab_video_frame() -> Response:
    """
    カメラから1フレームを取得し、ゲームロジックを適用後、JPEG画像として返す。
    カメラが利用できない場合、プレースホルダー画像を返す。
    """
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

    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)

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

    # フレームの解像度を取得
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    # フレームの中心に円を描画
    marker_radius = 10  # 円の半径
    marker_color = (0, 255, 0)  # 緑 (BGR)
    marker_thickness = 2  # 線の太さ
    cv2.circle(frame, (center_x, center_y), marker_radius, marker_color, marker_thickness)

    # TODO: ゲームロジックの更新
    frame = game.rogic(frame)

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
    video_image = ui.interactive_image().classes('w-full h-full').style('margin-top: 30px; margin-bottom: 50px; z-index: 5;')

    # メッセージ表示用ラベル
    message_label = ui.label().classes('text-xl font-bold').style('margin-top: -50px;')

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