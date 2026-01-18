import numpy as np
import onnxruntime as ort
import yaml
from core.utils import softmax, preprocess


with open('./configs/config.yaml') as f:
    config = yaml.safe_load(f)

model_path = config['liveness']['model_liveness_path']
threshold = config['liveness']['liveness_threshold']

providers = [
    ("CUDAExecutionProvider", {
        "cudnn_conv_algo_search": "HEURISTIC" 
    }),
    "CPUExecutionProvider"
]
session = ort.InferenceSession(model_path, providers=providers)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def liveness_predict(face_crop):
    x = preprocess(face_crop)

    logits = session.run([output_name], {input_name: x})[0][0]
    probs = softmax(np.asarray(logits))

    real_score = probs[1]
    fake_score = 1 - real_score
    is_real = True if (real_score >= threshold) else False

    return is_real, real_score, fake_score, probs
