import cv2
import cv2.aruco as aruco
import numpy as np
import time
import os

### ここは各自で番号変えてね(0がデフォルト) ###
targetVideo = 0 # カメラデバイス
cap = cv2.VideoCapture( targetVideo )

# キャプチャ間隔 (秒)
capture_interval = 0.5
last_capture_time = time.time()

# キャプチャ画像保存先フォルダ
output_folder = "./captures"
os.makedirs(output_folder, exist_ok=True)
capture_count = 0

# 投影する画像
overlay_image = cv2.imread("iraira.jpg")

# ArUco辞書と検出器を初期化
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_detector = aruco.ArucoDetector(aruco_dict)


while cap.isOpened():
    ret, img = cap.read()
    if not ret or img is None:
        break

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # ARマーカーの検出
    corners, ids, rejected = aruco_detector.detectMarkers(img)

    if len(corners) > 0:
        # カメラ行列とレンズ歪みの設定
        center = (img.shape[1] / 2, img.shape[0] / 2)
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
            warped_image = cv2.warpPerspective(overlay_image, matrix, (img.shape[1], img.shape[0]))

            # 合成用のマスクを作成
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.fillConvexPoly(mask, imgpts, 255)
            mask_inv = cv2.bitwise_not(mask)

            # 元の画像に投影画像を合成
            img_bg = cv2.bitwise_and(img, img, mask=mask_inv)
            img_fg = cv2.bitwise_and(warped_image, warped_image, mask=mask)
            img = cv2.add(img_bg, img_fg)
            
    # フレームの解像度を取得
    height, width = img.shape[:2]
    center_x, center_y = width // 2, height // 2

    # フレームの中心に円を描画
    marker_radius = 5  # 円の半径
    marker_color = (0, 0, 255)  # 赤 (BGR)
    marker_thickness = 3  # 線の太さ
    cv2.circle(img, (center_x, center_y), marker_radius, marker_color, marker_thickness)
            
    # 一定間隔でキャプチャ画像を保存
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        capture_filename = f"{output_folder}/capture_{capture_count:04d}.png"
        cv2.imwrite(capture_filename, img)
        print(f"Captured: {capture_filename}")
        capture_count += 1
        last_capture_time = current_time

    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
