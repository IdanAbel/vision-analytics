import cv2
import numpy as np
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict

model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)

def process_video(video_path, output_video_path, heatmap_output_path, insights_output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    heatmap = np.zeros((height, width), dtype=np.float32)

    people_tracks = defaultdict(list)  # {track_id: [(cx, cy, frame_index)]}
    people_first_seen = {}
    people_last_seen = {}
    people_per_minute = defaultdict(set)  # minute -> set of track_ids

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        detections = []
        for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
            if int(cls) == 0 and conf > 0.3:
                x1, y1, x2, y2 = map(int, box)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # heatmap
            if 0 <= cx < width and 0 <= cy < height:
                heatmap[cy, cx] += 1

            # track
            people_tracks[track_id].append((cx, cy, frame_index))
            if track_id not in people_first_seen:
                people_first_seen[track_id] = frame_index
            people_last_seen[track_id] = frame_index

            # per-minute
            minute = int((frame_index / fps) // 60)
            people_per_minute[minute].add(track_id)

            # draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()

    # Heatmap
    heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_img = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(heatmap_output_path, heatmap_img)

    # Insights
    total_people = len(people_tracks)
    avg_time_sec = np.mean([
        (people_last_seen[t] - people_first_seen[t]) / fps
        for t in people_tracks
    ]) if people_tracks else 0

    max_point = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    hotspots = np.argwhere(heatmap > np.percentile(heatmap, 99.5)).tolist()

    people_per_minute_simple = {
       f"{minute:02d}:00–{minute+1:02d}:00": len(track_ids)
       for minute, track_ids in people_per_minute.items()
    }

    insights = {
        "total_people": total_people,
        "avg_time_in_frame_sec": round(avg_time_sec, 2),
        "most_visited_point": [int(max_point[1]), int(max_point[0])],  # x,y
        "hotspots": [[int(x), int(y)] for y, x in hotspots[:20]],  # limit to 20
        "people_per_minute": people_per_minute_simple
    }

    with open(insights_output_path, 'w') as f:
        json.dump(insights, f, indent=2)

    print("✅ insights.json created.")
