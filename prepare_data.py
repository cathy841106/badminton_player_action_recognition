import cv2
import os
import glob
import argparse
from ultralytics import YOLO
from PIL import Image
from player_half_court_classifier import classify_player


def predict_skeleton(ori_frame_folder, model_path):
    # 載入骨架偵測模型
    model = YOLO(model_path)

    # 預測骨架
    results = model.predict(source=ori_frame_folder)

    # 整理骨架預測結果
    grouped_results = {}
    for result in results:
        image_path = result.path
        image_name = result.path.split('/')[-1]  # 擷取圖片名稱
        if image_path not in grouped_results:
            grouped_results[image_path] = []  # 初始化特定圖片的陣列

        for item in result:
            detection = {
                "bbox": item.boxes.xyxy[0].tolist(),  # Bounding box座標
                "keypoints": item.keypoints.xy[0].tolist(),  # Keypoints座標
            }
            grouped_results[image_path].append(detection)
    return grouped_results


def process_frames(video_name, skeleton_predict_result):
    for image_path, results in skeleton_predict_result.items():
        for result in results:
            # player位置分類 (top half court/bottom half court/outside court)
            player_position_classify_result = classify_player(result['bbox'])
            # 忽略outside court的人
            if player_position_classify_result == 'Outside Court':
                continue

            if player_position_classify_result == 'Top Half':
                player_position = 'top_half'
            elif player_position_classify_result == 'Bottom Half':
                player_position = 'bottom_half'

            # 創建輸出資料夾
            output_dir = os.path.join('./processed_frame', video_name, player_position)
            os.makedirs(output_dir, exist_ok=True)

            # 獲取圖片名稱
            image_name = os.path.basename(image_path)

            # 開啟圖片並擷取區域
            with Image.open(image_path) as img:
                # 擷取指定的範圍
                # 為了使擷取的frame不要太對齊使用者，多增加一些範圍再擷取
                adjust_bbox = [result['bbox'][0]-40, result['bbox'][1]-40, result['bbox'][2]+40, result['bbox'][3]+40]
                cropped_image = img.crop(adjust_bbox)  

                # 儲存裁剪後的圖片
                output_path = os.path.join(output_dir, image_name)
                cropped_image.save(output_path)

            print(f"圖片已儲存至: {output_path}")            


def extract_frames(video_path, video_name, output_path, fps=10):   
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟影片: {video_path}")
        return

    # 獲取影片的幀率
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 結束時停止

        # 儲存需要的幀
        if frame_count % frame_interval == 0:
            frame_file = os.path.join(output_path, f"{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_file, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"完成，總共儲存了 {saved_frame_count} 幀到資料夾: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare action recognition data from video.")
    parser.add_argument("--video_path", type=str, help="Path to the input video file.")
    parser.add_argument("--video_folder_path", type=str, help="Path to the input video folder.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second to extract.")
    args = parser.parse_args()

    video_path_list = []
    if args.video_folder_path:
        video_path_list.extend(glob.glob(os.path.join(args.video_folder_path, '*')))
    elif args.video_path:
        video_path_list.append(args.video_path)

    for video_path in video_path_list:
        print (f'處理影片: {video_path}')
        # 擷取影片名稱（不含副檔名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 創建原始frame輸出資料夾
        ori_frame_output_dir = os.path.join('ori_frame', video_name)
        os.makedirs(ori_frame_output_dir, exist_ok=True)

        # 依照設定禎數擷取frame
        extract_frames(video_path, video_name, ori_frame_output_dir, args.fps)

        # 使用骨架偵測模型（Andy訓練好的best.pt）偵測frame資料夾裡的各個frame的人物骨架資訊
        skeleton_predict_result = predict_skeleton(ori_frame_output_dir, './yolo8n-pose_weights/best.pt')

        # 依照各個frame的人物骨架資訊擷取人物並儲存人物圖片
        process_frames(video_name, skeleton_predict_result)