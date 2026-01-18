import numpy as np
import cv2
from collections import deque

def cosine_similarity(a, b):
    eps = 1e-10
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))

def cosine_similarity_db(emb, vectors_db_norm):
    # emb, vectors_db: np.array()

    eps = 1e-10
    emb = emb / (np.linalg.norm(emb) + eps)

    return np.dot(vectors_db_norm, emb)


DST_112 = np.array([
    [38.2946, 51.6963],     # left eye
    [73.5318, 51.5014],     # right eye
    [56.0252, 71.7366],     # nose
    [41.5493, 92.3655],     # left mouth
    [70.7299, 92.2041],     # right mouth
], dtype=np.float32)

def align_crop_5pts(img_bgr, kps5, out_size=80):
    scale = out_size / 112.0
    dst = DST_112 * scale

    M, _ = cv2.estimateAffinePartial2D(kps5, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        # fallback: use 3 points (2 eyes + nose)
        M = cv2.getAffineTransform(kps5[:3], dst[:3])

    aligned = cv2.warpAffine(
        img_bgr, M, (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )
    return aligned

def preprocess(img_bgr, input_size=80):
    img = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return img[None]    # batch_size = 1    (1, 3, 80, 80)

def softmax(scores: np.ndarray):
    s = scores.astype(np.float32)
    s = s - np.max(s)
    e = np.exp(s)
    return e / (np.sum(e) + 1e-12)


# IoU + Tracker
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return inter / union

class SimpleTracker:
    """
    Track by bbox IoU. Assign a stable track_id for each face across frames.
    """
    def __init__(self, iou_thresh=0.3, max_miss=10):
        self.iou_thresh = float(iou_thresh)
        self.max_miss = int(max_miss)
        self.next_id = 0
        self.tracks = {}  # id -> {"bbox": (x1,y1,x2,y2), "miss": int}

    def update(self, bboxes):
        """
        bboxes: list of (x1,y1,x2,y2) in the same coordinate system (frame_small or frame_full).
        returns: dict {bbox_index: track_id}
        """
        assigned = {}
        used_ids = set()

        # match each bbox to best existing track
        for j, bb in enumerate(bboxes):
            best_id, best_iou = None, 0.0
            for tid, t in self.tracks.items():
                if tid in used_ids:
                    continue
                score = iou_xyxy(bb, t["bbox"])
                if score > best_iou:
                    best_iou, best_id = score, tid

            if best_id is not None and best_iou >= self.iou_thresh:
                assigned[j] = best_id
                used_ids.add(best_id)
                self.tracks[best_id]["bbox"] = bb
                self.tracks[best_id]["miss"] = 0
            else:
                tid = self.next_id
                self.next_id += 1
                assigned[j] = tid
                used_ids.add(tid)
                self.tracks[tid] = {"bbox": bb, "miss": 0}

        # increase miss for un-used tracks
        for tid in list(self.tracks.keys()):
            if tid not in used_ids:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.max_miss:
                    del self.tracks[tid]

        return assigned

class MovingAvgBuffer:
    def __init__(self, window=20):
        self.buf = deque(maxlen=int(window))

    def update(self, score):
        self.buf.append(float(score))
        avg = float(np.mean(self.buf))
        ready = (len(self.buf) == self.buf.maxlen)
        return avg, ready, len(self.buf)

    def reset(self):
        self.buf.clear()
