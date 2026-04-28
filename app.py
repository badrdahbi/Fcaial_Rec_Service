import base64
import numpy as np
import torch
import requests
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps, ImageEnhance
from functools import lru_cache
import io

app = Flask(__name__)
CORS(app)

# Initialisation des modèles
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "="*50)
print("🚀 RECOGNITION SERVICE — INITIALISATION")
print("="*50)
print(f"🖥️  Périphérique : {device}")

# MTCNN pour la détection et ResNet pour les embeddings
mtcnn = MTCNN(image_size=160, margin=14, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

print("✅ Modèles IA chargés et prêts !")
print("="*50 + "\n")

# Statistiques de session
stats = {"total": 0, "accepted": 0, "rejected": 0, "no_face": 0}

@lru_cache(maxsize=100)
def get_cached_profile_embedding(profile_url):
    """Télécharge et extrait l'ID visage d'une URL avec cache."""
    try:
        resp = requests.get(profile_url, timeout=10)
        if resp.status_code != 200: return "ERR_HTTP", None
        
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        emb = extract_embedding(img)
        return (emb, None) if emb is not None else ("ERR_NO_FACE", None)
    except Exception as e:
        return "ERR_NETWORK", str(e)

def extract_embedding(img):
    """Extrait l'empreinte mathématique d'un visage."""
    try:
        img = ImageOps.exif_transpose(img)
        # Redimensionnement léger pour la performance
        if max(img.width, img.height) > 800:
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
        
        boxes, _ = mtcnn.detect(img)
        if boxes is None: return None
        
        # On prend le premier visage détecté
        face_tensor = mtcnn.extract(img, [boxes[0]], save_path=None)[0]
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device))
        return embedding.detach().cpu().numpy().flatten()
    except:
        return None

@app.route('/verify', methods=['POST'])
def verify():
    t_start = time.perf_counter()
    try:
        data = request.json
        if not data or 'face_image' not in data or 'profile_url' not in data:
            return jsonify({"error": "Données manquantes (face_image/profile_url)"}), 400

        profile_url = data['profile_url']
        face_b64    = data['face_image']
        card_id     = data.get('card_id', 'Inconnu')
        threshold   = data.get('threshold', 0.65)

        # Nettoyage Base64
        if "," in face_b64: face_b64 = face_b64.split(",")[1]
        
        # 1. Traitement Selfie
        captured_bytes = base64.b64decode(face_b64)
        captured_img   = Image.open(io.BytesIO(captured_bytes)).convert('RGB')
        captured_emb   = extract_embedding(captured_img)

        if captured_emb is None:
            stats["no_face"] += 1
            return jsonify({"success": False, "error": "Aucun visage détecté sur la caméra"}), 200

        # 2. Récupération Profil
        profile_res, err = get_cached_profile_embedding(profile_url)
        if isinstance(profile_res, str):
            return jsonify({"success": False, "error": f"Erreur profil: {profile_res}"}), 400

        # 3. Comparaison
        similarity = np.dot(profile_res, captured_emb) / (np.linalg.norm(profile_res) * np.linalg.norm(captured_emb))
        verified   = bool(similarity >= threshold)
        
        # Logs
        stats["total"] += 1
        if verified: stats["accepted"] += 1 
        else: stats["rejected"] += 1
        
        t_total = (time.perf_counter() - t_start) * 1000
        print(f"🔍 [VERIFY] {card_id} | Score: {similarity:.2f} | Result: {'✅ OK' if verified else '❌ KO'} ({t_total:.0f}ms)")

        return jsonify({
            "verified": verified,
            "confidence": round(float(similarity) * 100, 2),
            "processing_ms": round(t_total, 1),
            "card_id": card_id
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Erreur interne serveur"}), 500

@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    return jsonify({"status": "active", "api": "Facial Recognition", "workers_online": True}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
