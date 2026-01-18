from insightface.app import FaceAnalysis
import yaml
import os

with open('./configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_root = config['paths']['model_face_root']
model_name = config['model']['model_name']
det_size = config['model']['det_size']

providers = [
    ("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC" 
    }),
    "CPUExecutionProvider"
]
app = FaceAnalysis(name=model_name,
                   root=model_root,
                   allowed_modules=['detection', 'recognition'],
                   providers=providers)
app.prepare(ctx_id=0, det_size=(det_size, det_size))

os.system('cls')

def face_analysis(img):
    if img is None:
        return {'bbox': [],
                'vector_emb': [],
                # 'landmark68': [],
                'landmark5': []}
    
    faces = app.get(img)

    if len(faces) == 0:
        return {'bbox': [],
                'vector_emb': [],
                # 'landmark68': [],
                'landmark5': []}
    
    bbox_list, emd_list, lm5_list, lm68_list = [], [], [], []

    for face in faces:
        bbox = face.bbox.astype(int)    # xmin, ymin, xmax, ymax
        bbox_list.append(bbox)

        # lm3d_68 = getattr(face, 'landmark_3d_68', None)
        # lm68_list.append(lm3d_68)

        kps = face.kps
        lm5_list.append(kps)

        emb = face.embedding
        emd_list.append(emb)

    return {'bbox': bbox_list,
            'vector_emb': emd_list,
            # 'landmark68': lm68_list,
            'landmark5': lm5_list}