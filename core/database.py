import numpy as np
import os
import json
import yaml
from core.utils import cosine_similarity

with open('./configs/config.yaml') as f:
    config = yaml.safe_load(f)

file_vectors = config['paths']['file_vectors']
file_names = config['paths']['file_names']
similiarity_threshold = 0.7

"""
Examples: 
names = {
    'name': ["Hung", "Hanh", "Han"],
    'face_angle': ['straight', 'left_side', 'right_side']
}
"""

def load_db():
    if not os.path.exists(file_vectors) or not os.path.exists(file_names):
        return [], {'name': [], 'face_angle': []}
    
    vectors = np.load(file_vectors)

    with open(file_names, 'r', encoding='utf-8') as g:
        names = json.load(g)

    return list(vectors), names

def save_db(vectors, names):
    np.save(file_vectors, np.array(vectors))

    with open(file_names, 'w', encoding='utf-8') as u:
        json.dump(names, u, ensure_ascii=False)

    print('Saved new data')

def add_user(new_name, new_vector):
    """
    Example:
    new_name = {
        'name': 'Thuy',
        'face_angle': 'straight'
    }
    """
    vectors, names = load_db()

    new_name['name'] = new_name['name'].strip()
    
    is_name_different = True
    is_angle_different = True

    all_names = [name.lower() for name in names['name']]
    
    if new_name['name'].lower() in all_names:
        is_name_different = False
        face_angle = [names['face_angle'][i] for i, v in enumerate(all_names) if v == new_name['name'].lower()]
        if new_name['face_angle'] in face_angle:
            is_angle_different = False
            print(f"The name '{new_name['name']}' already exists and the facial angle '{new_name['face_angle']}' is the same")
    
    if is_name_different or is_angle_different:
        is_duplicate_face = False
        for iter, vector_emb in enumerate(vectors):
            if (cosine_similarity(vector_emb, new_vector) >= similiarity_threshold) and (new_name['name'].lower() != all_names[iter]):
                print(f"The vector embedding of {new_name['name']} is already available in the database and is saved as {names['name'][iter]}")
                is_duplicate_face = True
                break

        if is_duplicate_face == False:
            names['name'].append(new_name['name'])
            names['face_angle'].append(new_name['face_angle'])

            vectors.append(new_vector)

            save_db(vectors, names)

def delete_user(name_delete):
    vectors, names = load_db()

    deleted_count = 0
    total_records = len(names['name'])

    for i in range(total_records-1, -1, -1):
        if names['name'][i].lower() == name_delete.lower():
            names['name'].pop(i)
            names['face_angle'].pop(i)

            vectors.pop(i)

            deleted_count += 1

    if deleted_count > 0:
        print(f"Deleted data of user '{name_delete}'")
        save_db(vectors, names)
    else:
        print(f"User '{name_delete}' not found")
        