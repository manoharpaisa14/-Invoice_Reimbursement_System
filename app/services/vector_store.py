import faiss
import os
import pickle
import shutil
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_PATH = os.path.join(BASE_DIR, "vector.index")
META_PATH = os.path.join(BASE_DIR, "metadata.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(text: str) -> List[float]:
    return model.encode([text])[0]

def save_to_vector_store(full_text: str, metadata: Dict):
    embedding = embed_text(full_text).astype("float32")

    if os.path.exists(VECTOR_PATH):
        index = faiss.read_index(VECTOR_PATH)
        with open(META_PATH, "rb") as f:
            metas = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(len(embedding))
        metas = []

    index.add(np.array([embedding]))
    metadata["full_text"] = full_text
    metas.append(metadata)

    faiss.write_index(index, VECTOR_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(metas, f)

    try:
        root_vector = os.path.abspath(os.path.join(BASE_DIR, "../../vector.index"))
        root_meta = os.path.abspath(os.path.join(BASE_DIR, "../../metadata.pkl"))
        shutil.copy(VECTOR_PATH, root_vector)
        shutil.copy(META_PATH, root_meta)
        print("📤 Copied index and metadata to root directory.")
    except Exception as e:
        print(f"❌ Failed to copy files to root: {e}")

    print(f"✅ Saved: {metadata.get('invoice_id')} | Total: {len(metas)}")

def search_similar(query: str, filters: Dict = {}, top_k: int = 10) -> List[Dict]:
    if not os.path.exists(VECTOR_PATH) or not os.path.exists(META_PATH):
        return []

    embedding = embed_text(query).astype("float32")
    index = faiss.read_index(VECTOR_PATH)
    with open(META_PATH, "rb") as f:
        metas = pickle.load(f)

    D, I = index.search(np.array([embedding]), top_k)
    results = []

    for idx in I[0]:
        if idx < len(metas):
            match = metas[idx]
            if not filters:
                results.append(match)
            else:
                all_match = True
                for k, v in filters.items():
                    meta_val = str(match.get(k, "")).strip().lower()
                    filter_val = str(v).strip().lower()
                    if meta_val != filter_val:
                        all_match = False
                        break
                if all_match:
                    results.append(match)
    return results

def ensure_vector_files_exist():
    if not os.path.exists(VECTOR_PATH):
        dim = 384  # embedding size
        index = faiss.IndexFlatL2(dim)
        faiss.write_index(index, VECTOR_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump([], f)

    try:
        root_vector = os.path.abspath(os.path.join(BASE_DIR, "../../vector.index"))
        root_meta = os.path.abspath(os.path.join(BASE_DIR, "../../metadata.pkl"))
        shutil.copy(VECTOR_PATH, root_vector)
        shutil.copy(META_PATH, root_meta)
        print("✅ Vector store created/copied to root.")
    except Exception as e:
        print(f"❌ Failed to copy vector files to root: {e}")
