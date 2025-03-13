import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from constants import FACE_ID_TRAIN_PATH, FACE_ID_TEST_PATH
from Facenet.face_id_dataset import load_faces_in_batch

import pandas as pd
from deepface.modules import modeling

def calculate_embeddings_from_dir(model, batch_size = 32, image_dir = FACE_ID_TRAIN_PATH):
    """ calculate embeddings for image_dir in batches """
    total_embeddings = pd.DataFrame(columns=['identity', 'embeddings'])

    for paths, images in load_faces_in_batch(batch_size, image_dir):
        embeddings = model.model(images).numpy()
        
        batch_df = pd.DataFrame({'identity': paths, 'embeddings': list(embeddings)})
        
        total_embeddings = pd.concat([total_embeddings, batch_df], ignore_index=True)
        
    return total_embeddings


def _calculate_train_embeddings():
    model = modeling.build_model(task="facial_recognition", model_name="Facenet512")
    
    embeddings = calculate_embeddings_from_dir(model)
    # save embeddings
    embeddings_dir = Path("src/Facenet/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings.to_csv(embeddings_dir / "train_embeddings.csv", index=None)
    
def _calculate_test_embeddings():
    model = modeling.build_model(task="facial_recognition", model_name="Facenet512")
    
    embeddings = calculate_embeddings_from_dir(model, image_dir=FACE_ID_TEST_PATH)
    # save embeddings
    embeddings_dir = Path("src/Facenet/embeddings")
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings.to_csv(embeddings_dir / "test_embeddings.csv", index=None)

if __name__ == "__main__":
    
    # _calculate_train_embeddings()
    _calculate_test_embeddings()
