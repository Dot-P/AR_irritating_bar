import base64
import time
import cv2
from fastapi import Response
from nicegui import app, ui
from pydantic import BaseModel
# ARマーカー
import cv2.aruco as aruco
import numpy as np
import os

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
    video_capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # デフォルトのカメラ（ID: 0）を開く
    if not video_capture.isOpened():
        print("カメラが開けません")
        return placeholder
    ret, frame = video_capture.read()
    video_capture.release()
    if not ret:
        return placeholder
    
    # TODO: ARマーカーの表示

    # 投影する画像
    overlay_image = cv2.imread("iraira.jpg")

    # ArUco辞書と検出器を初期化
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_detector = aruco.ArucoDetector(aruco_dict)
        
    scale_percent = 50  # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)

    # ARマーカーの検出
    corners, ids, rejected = aruco_detector.detectMarkers(frame)

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
            dst_pts = imgpts.astype("float32")

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
    marker_radius = 5  # 円の半径
    marker_color = (0, 0, 255)  # 赤 (BGR)
    marker_thickness = 3  # 線の太さ
    cv2.circle(frame, (center_x, center_y), marker_radius, marker_color, marker_thickness)

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
