import cv2
import yaml
from core.engine import face_analysis
from core.database import add_user

with open('./configs/config.yaml') as f:
    config = yaml.safe_load(f)

yaw_thresh_left = config['enroll_user']['yaw_thresh_left']
yaw_thresh_right = config['enroll_user']['yaw_thresh_right']

required_stable_frames = 50

def check_pose(kps):
    eye_l = kps[0]
    eye_r = kps[1]
    nose = kps[2]

    dist_l = abs(nose[0] - eye_l[0])
    dist_r = abs(nose[0] - eye_r[0])

    if dist_r <= 0:
        dist_r = 0.00001

    ratio = dist_l / dist_r

    if ratio < yaw_thresh_left:
        return "left_side"
    elif ratio > yaw_thresh_right:
        return "right_side"
    else:
        return "straight"
    
def run(user_name):
    step = 0
    captured_vectors = {}
    stability_counter = 0

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        out_faces = face_analysis(frame)
        faces_bbox = out_faces['bbox']
        faces_emb = out_faces['vector_emb']
        faces_lm5 = out_faces['landmark5']

        color_display = (0, 255, 255) # Yellow

        if len(faces_bbox) == 0:
            instruction = "Not found face"
            stability_counter = 0
        elif len(faces_bbox) > 1:
            instruction = "Only one person in front of camera"
            stability_counter = 0
        else:
            face_bbox = faces_bbox[0]
            face_emb = faces_emb[0]
            face_lm5 = faces_lm5[0]

            pose = check_pose(face_lm5)

            target_pose = ''
            if step == 0:
                target_pose = 'straight'
            elif step == 1:
                target_pose = 'left_side'
            elif step == 2:
                target_pose = 'right_side'

            if step < 3:
                if pose == target_pose:
                    stability_counter += 1
                    instruction = f"Keep still .... {stability_counter}/{required_stable_frames}"
                    color_display = (0, 255, 0)     # Green
                    if stability_counter >= required_stable_frames:
                        captured_vectors[pose] = face_emb
                        step += 1
                        stability_counter = 0
                else:
                    stability_counter = 0
                    if target_pose == 'straight':
                        instruction = "Please keep your head STRAIGHT"
                    elif target_pose == 'left_side':
                        instruction = f"Please turn to LEFT"
                    else:
                        instruction = f"Please turn to RIGHT"
                    color_display = (255, 0, 0)     # Blue

            else:
                instruction = "Completed! Please enter 'q' to exit"
                color_display = (255, 255, 0)

            for (x , y) in face_lm5.astype(int):
                cv2.circle(display_frame, center=(x,y), radius=2, color=(255,0,0))
        
        cv2.putText(display_frame, text=instruction, org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color_display, thickness=2)
        cv2.imshow("Auto Enrollment", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(captured_vectors) == 3:
        return captured_vectors
    return None
    
if __name__ == '__main__':
    user_name = input("Please enter your registration name: ").strip()
    vectors = run(user_name)

    if vectors:
        for side, vector in vectors.items():
            new_name = {'name': user_name, 'face_angle': side}
            add_user(new_name, vector)       
        