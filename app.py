import os
import base64
import numpy as np
import torch
import requests  # ✅ NOUVEAU : Pour télécharger les images depuis le web
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Autorise Flutter à appeler l'API

# --- CONFIGURATION IA ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Chargement FaceNet sur {device}...")

# MTCNN pour détection/crop précis
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
# InceptionResnetV1 pour l'embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Base de données en mémoire (Nom -> Embedding)
FACE_DB = {}

# 🌐 L'URL du site web de ton camarade qui doit te renvoyer la liste des URLs des photos
# Exemple d'IP locale : 172.16.1.20:8000
URL_API_PHP = "http://172.16.1.20:8000/api/get-all-faces"


def get_embedding(image_bytes):
    """Convertit une image brute en embedding (signature 512 chiffres)"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_cropped = mtcnn(img)
        
        if img_cropped is None:
            return None
        
        img_embedding = resnet(img_cropped.unsqueeze(0).to(device))
        return img_embedding.detach().cpu().numpy().flatten()
    except Exception as e:
        print(f"Erreur embedding: {e}")
        return None


def load_db():
    """Télécharge les photos depuis le site web de ton camarade au démarrage"""
    print("🌐 Téléchargement des visages depuis le site web...")
    count = 0
    global FACE_DB
    FACE_DB = {}
    
    try:
        # 1. On demande la liste des URLs au site web PHP
        print(f"Connexion à {URL_API_PHP}...")
        response = requests.get(URL_API_PHP, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Erreur: Le site web a répondu avec le code {response.status_code}")
            return
            
        # 2. Le PHP doit nous renvoyer un JSON comme ça : {"V255430": "http://.../face.jpg"}
        faces_data = response.json()
        
        for user_id, img_url in faces_data.items():
            try:
                # 3. On télécharge l'image
                img_response = requests.get(img_url, timeout=10)
                if img_response.status_code == 200:
                    emb = get_embedding(img_response.content)
                    
                    if emb is not None:
                        if user_id not in FACE_DB:
                            FACE_DB[user_id] = []
                        FACE_DB[user_id].append(emb)
                        count += 1
                        print(f"  ✅ Chargé depuis le web: {user_id}")
                    else:
                        print(f"  ⚠️ Pas de visage détecté dans la photo de: {user_id}")
            except Exception as e:
                print(f"  ❌ Erreur de téléchargement pour {user_id}: {e}")

        # Moyenne des embeddings
        for name, embs in FACE_DB.items():
            FACE_DB[name] = np.mean(embs, axis=0)
            
        print(f"🎉 DB Web Prête : {list(FACE_DB.keys())} ({count} photos téléchargées)")
        
    except Exception as e:
        print(f"❌ Impossible de contacter le site PHP : {e}")


# --- ROUTES API ---

@app.route('/recognize', methods=['POST'])
def recognize():
    """Reçoit une image base64 de Flutter, renvoie l'identité"""
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "Pas d'image"}), 400
        
    try:
        image_bytes = base64.b64decode(data['image'])
        target_emb = get_embedding(image_bytes)
        
        if target_emb is None:
            return jsonify({"match": False, "reason": "No face detected"}), 200
            
        best_score = 0
        best_user = "inconnu"
        
        for name, db_emb in FACE_DB.items():
            score = np.dot(target_emb, db_emb) / (np.linalg.norm(target_emb) * np.linalg.norm(db_emb))
            if score > best_score:
                best_score = score
                best_user = name
                
        THRESHOLD = 0.65 
        
        if best_score > THRESHOLD:
            return jsonify({"match": True, "user": best_user, "confidence": round(float(best_score) * 100, 2)})
        else:
            return jsonify({"match": False, "best_guess": best_user, "confidence": round(float(best_score) * 100, 2)})
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/reload', methods=['GET'])
def reload():
    """Route pour tout recharger depuis le site web"""
    load_db()
    return jsonify({"message": "Base de données synchronisée depuis le web !", "total_users": len(FACE_DB)})


# ✅ NOUVELLE ROUTE TRÈS PERFORMANTE
@app.route('/add_face_url', methods=['POST'])
def add_face_url():
    """
    Au lieu de tout recharger, le PHP de ton camarade peut appeler cette route
    quand il ajoute UN SEUL employé, en envoyant : {"user_id": "V123", "url": "http://..."}
    """
    data = request.json
    user_id = data.get('user_id')
    img_url = data.get('url')
    
    if not user_id or not img_url:
        return jsonify({"error": "user_id et url requis"}), 400
        
    try:
        img_response = requests.get(img_url, timeout=10)
        if img_response.status_code == 200:
            emb = get_embedding(img_response.content)
            if emb is not None:
                global FACE_DB
                FACE_DB[user_id] = emb # Ajoute l'employé à la mémoire
                return jsonify({"message": f"Utilisateur {user_id} ajouté à l'IA avec succès !"})
            else:
                return jsonify({"error": "Aucun visage trouvé dans l'URL fournie"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
