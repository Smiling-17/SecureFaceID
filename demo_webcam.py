import cv2
import numpy as np
import yaml
from core.utils import cosine_similarity_db, align_crop_5pts, SimpleTracker, MovingAvgBuffer
from core.database import load_db
from core.engine import face_analysis
from core.liveness import liveness_predict


with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

similarity_threshold = config['similarity_threshold']
scale_factor = config['scale_factor']
input_size = config['liveness']['input_size']
liveness_threshold = config['liveness']['liveness_threshold']
need_frames = config['liveness']['need_frames']


def realtime_recognition():
    vectors_db, names_db = load_db()

    tracker = SimpleTracker(iou_thresh=0.3, max_miss=20)
    buffers = {}  # track_id -> MovingAvgBuffer

    if not vectors_db:
        print("The database is empty!")
        vectors_db = np.empty((0, 512))
    else:
        vectors_db = np.array(vectors_db, dtype=np.float32)

    vectors_db_norm = vectors_db / (np.linalg.norm(vectors_db, axis=1, keepdims=True) + 1e-10)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        frame_small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

        out_faces = face_analysis(frame_small)
        faces_bbox = out_faces['bbox']
        faces_emb = out_faces['vector_emb']
        faces_lm5 = out_faces['landmark5']

        bboxes_small = [tuple(map(float, b)) for b in faces_bbox]
        assign = tracker.update(bboxes_small)

        active_ids = set(tracker.tracks.keys())
        buffers = {k: v for k, v in buffers.items() if k in active_ids}

        # if len(vectors_db) > 0 and len(faces_emb) > 0:
        if len(faces_bbox) > 0:
            for i, bbox in enumerate(faces_bbox):
                xmin, ymin, xmax, ymax = (bbox / scale_factor).astype(int)

                tid = assign[i]
                if tid not in buffers:
                    buffers[tid] = MovingAvgBuffer(window=need_frames)

                face_crop = align_crop_5pts(frame_small, kps5=faces_lm5[i], out_size=input_size)
                _, real_score, _, _ = liveness_predict(face_crop)

                avg, ready, cnt = buffers[tid].update(real_score)
                if not ready:
                    name = f"Checking ... {cnt}/{need_frames}"
                    color_display = (255, 0, 0)  # Blue

                    cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), color=color_display, thickness=2)  # SandyBrown
                    cv2.putText(frame_display, text=name, org=(xmin+3, ymin - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=color_display, thickness=2)
                    continue
                
                live_confirmed = avg >= liveness_threshold

                if not live_confirmed:
                    name = "Fake / Spoof"
                    color_display = (0, 0, 255) # Red
                else:
                    if len(vectors_db_norm) > 0 and len(faces_emb) > 0:
                        scores = cosine_similarity_db(faces_emb[i], vectors_db_norm)

                        max_score = np.max(scores)
                        max_index = np.argmax(scores, 0)
                        
                        if max_score >= similarity_threshold:
                            name = names_db['name'][max_index]
                            # name = f"{name} ({max_score:.2f})"
                            color_display = (0, 255, 0)     # Green
                        else:
                            name = "Unknown"
                            color_display = (0, 255, 255)     # Yellow
                    else:
                        name = "Unknown"
                        color_display = (0, 255, 255)     # Yellow

                cv2.rectangle(frame_display, (xmin, ymin), (xmax, ymax), color=color_display, thickness=2)  # SandyBrown
                cv2.putText(frame_display, text=name, org=(xmin+3, ymin - 5), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=color_display, thickness=2)

        cv2.imshow("Secure Face Recognition", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    realtime_recognition()
