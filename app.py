import base64
import os
import numpy as np
import torch
import time
import traceback
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageOps, ImageEnhance
import io
import sys

# Assurer que le dossier courant est dans sys.path pour importer embeddings.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from embeddings import extract_embedding_from_image

# ==================== LOGGER SETUP ====================
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import sys
from pythonjsonlogger import jsonlogger

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class DetailedFormatter(logging.Formatter):
    """Formateur avec couleurs pour console"""
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Vert
        'WARNING': '\033[33m',    # Jaune
        'ERROR': '\033[31m',      # Rouge
        'CRITICAL': '\033[41m',   # Fond rouge
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        record.timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        log_format = (
            f"{color}"
            f"[{record.timestamp}] "
            f"[{record.levelname:^8}] "
            f"[{record.name}:{record.funcName}:{record.lineno}] "
            f"{reset}"
            f"{record.getMessage()}"
        )
        return log_format

class JSONFormatter(jsonlogger.JsonFormatter):
    """Formateur JSON pour logs"""
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['module'] = record.name
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        log_record['level'] = record.levelname

def setup_logger(name='FaceRecognition'):
    """Setup logger avec console + fichiers"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    
    # Console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(DetailedFormatter())
    logger.addHandler(console)
    
    # Fichier texte
    file_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "face_recognition.log",
        maxBytes=10*1024*1024,
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(DetailedFormatter())
    logger.addHandler(file_handler)
    
    # Fichier JSON
    json_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "face_recognition.json",
        maxBytes=10*1024*1024,
        backupCount=10,
        encoding='utf-8'
    )
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(JSONFormatter())
    logger.addHandler(json_handler)
    
    # Fichier erreurs
    error_handler = logging.handlers.RotatingFileHandler(
        LOGS_DIR / "face_recognition_errors.log",
        maxBytes=5*1024*1024,
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(DetailedFormatter())
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logger()

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)

# Initialisation des modèles
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info("")
logger.info("=" * 80)
logger.info("🚀 RECONNAISSANCE FACIALE — INITIALISATION")
logger.info("=" * 80)
logger.info(f"🖥️  Périphérique : {device}")

try:
    logger.debug("📦 Chargement MTCNN...")
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=True, device=device)
    logger.info("✅ MTCNN chargé")
    
    logger.debug("📦 Chargement InceptionResnetV1...")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    logger.info("✅ ResNet chargé")
    
    logger.info("=" * 80)
    logger.info("✅ Modèles IA chargés et prêts !")
    logger.info("=" * 80 + "\n")
    
except Exception as e:
    logger.critical(f"❌ Erreur initialisation modèles: {str(e)}", exc_info=True)
    raise

# Statistiques de session
stats = {"total": 0, "accepted": 0, "rejected": 0, "no_face": 0}


# ======================================================
# SUPPRIMÉ : get_cached_profile_embedding(profile_url)
#   → Plus besoin de télécharger depuis une URL
#   → Le profil arrive directement en Base64 (profile_image)
# ======================================================


def decode_base64_image(b64_string, label, request_id):
    """Décode une image Base64 en objet PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
        logger.debug(f"[{request_id}] ✅ Data URI nettoyé ({label})")
    
    try:
        img_bytes = base64.b64decode(b64_string)
        logger.debug(f"[{request_id}] ✅ Décodage Base64 OK ({label}: {len(img_bytes)} bytes)")
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Erreur décodage Base64 ({label}): {str(e)}")
        return None, "Format Base64 invalide"
    
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        logger.debug(f"[{request_id}] ✅ Image chargée ({label}): {img.size}")
        return img, None
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Format image invalide ({label}): {str(e)}")
        return None, "Format image invalide"


def extract_embedding(img, request_id=""):
    """Extrait l'empreinte mathématique d'un visage."""
    try:
        logger.debug(f"[{request_id}] 🔄 Correction EXIF...")
        img = ImageOps.exif_transpose(img)
        
        original_size = img.size
        logger.debug(f"[{request_id}] 📐 Taille originale: {original_size}")
        
        # Redimensionnement pour performance
        if max(img.width, img.height) > 800:
            logger.debug(f"[{request_id}] 📉 Redimensionnement...")
            img.thumbnail((800, 800), Image.Resampling.LANCZOS)
            logger.debug(f"[{request_id}] ✅ Nouvelle taille: {img.size}")
        
        logger.debug(f"[{request_id}] 🎯 Détection MTCNN...")
        boxes, probs = mtcnn.detect(img)
        
        if boxes is None:
            logger.warning(f"[{request_id}] ⚠️ Aucun visage détecté")
            return None
        
        logger.info(f"[{request_id}] ✅ {len(boxes)} visage(s) détecté(s)")
        logger.debug(f"[{request_id}] 📊 Confiances: {[f'{p:.2%}' for p in probs]}")
        
        logger.debug(f"[{request_id}] 🎬 Extraction du premier visage...")
        face_tensor = mtcnn.extract(img, [boxes[0]], save_path=None)[0]
        logger.debug(f"[{request_id}] ✅ Visage extrait: {face_tensor.shape}")
        
        logger.debug(f"[{request_id}] 🧠 Calcul embedding ResNet...")
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0).to(device))
        
        emb_array = embedding.detach().cpu().numpy().flatten()
        logger.info(f"[{request_id}] ✅ Embedding généré ({len(emb_array)} dimensions)")
        
        return emb_array
        
    except Exception as e:
        logger.error(f"[{request_id}] ❌ Erreur extraction: {str(e)}", exc_info=True)
        return None

@app.route('/verify', methods=['POST'])
def verify():
    """Endpoint de vérification faciale"""
    request_id = str(uuid.uuid4())[:8]
    t_start = time.perf_counter()
    
    logger.info("=" * 80)
    logger.info(f"[{request_id}] 📨 Nouvelle requête /verify")
    logger.info("=" * 80)
    
    try:
        # Validation données
        logger.debug(f"[{request_id}] 🔍 Validation des données...")
        data = request.json
        
        if not data:
            logger.warning(f"[{request_id}] ⚠️ JSON vide")
            return jsonify({"error": "Données manquantes (JSON vide)"}), 400
        
        if 'face_image' not in data:
            logger.warning(f"[{request_id}] ⚠️ face_image manquant")
            return jsonify({"error": "face_image manquant"}), 400
        
        # ✅ MODIFIÉ : on attend profile_image (Base64) au lieu de profile_url
        if 'profile_image' not in data:
            logger.warning(f"[{request_id}] ⚠️ profile_image manquant")
            return jsonify({"error": "profile_image manquant"}), 400
        
        logger.info(f"[{request_id}] ✅ Données valides")

        profile_b64 = data['profile_image']
        face_b64    = data['face_image']
        card_id     = data.get('card_id', 'Inconnu')
        threshold   = data.get('threshold', 0.65)

        logger.debug(f"[{request_id}] 📋 Paramètres:")
        logger.debug(f"[{request_id}]   - card_id: {card_id}")
        logger.debug(f"[{request_id}]   - threshold: {threshold}")
        logger.debug(f"[{request_id}]   - profile_image: {len(profile_b64)} chars")
        logger.debug(f"[{request_id}]   - face_image: {len(face_b64)} chars")

        # 1. Traitement Selfie (caméra)
        logger.debug(f"[{request_id}] 📸 Décodage selfie...")
        captured_img, err = decode_base64_image(face_b64, "selfie", request_id)
        if captured_img is None:
            return jsonify({"error": err}), 400

        logger.debug(f"[{request_id}] 🧠 Extraction embedding selfie...")
        captured_emb = extract_embedding(captured_img, request_id)

        if captured_emb is None:
            stats["no_face"] += 1
            logger.warning(f"[{request_id}] ❌ Aucun visage détecté sur selfie")
            return jsonify({"success": False, "error": "Aucun visage détecté sur la caméra"}), 200

        logger.info(f"[{request_id}] ✅ Embedding selfie généré")

        # 2. ✅ MODIFIÉ : Traitement Profil depuis Base64 (plus de téléchargement HTTP)
        logger.debug(f"[{request_id}] 🖼️ Décodage image de profil (Base64)...")
        profile_img, err = decode_base64_image(profile_b64, "profil", request_id)
        if profile_img is None:
            return jsonify({"error": err}), 400

        logger.debug(f"[{request_id}] 🧠 Extraction embedding profil...")
        profile_emb = extract_embedding(profile_img, request_id)

        if profile_emb is None:
            logger.warning(f"[{request_id}] ⚠️ Aucun visage détecté dans l'image de profil")
            return jsonify({"success": False, "error": "Aucun visage détecté dans l'image de profil"}), 200

        logger.info(f"[{request_id}] ✅ Profil traité")

        # 3. Comparaison
        logger.debug(f"[{request_id}] 🔢 Calcul similarité (cosinus)...")
        try:
            similarity = np.dot(profile_emb, captured_emb) / (np.linalg.norm(profile_emb) * np.linalg.norm(captured_emb))
            logger.debug(f"[{request_id}] ✅ Similarité calculée: {similarity:.4f}")
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Erreur calcul similarité: {str(e)}", exc_info=True)
            return jsonify({"error": "Erreur comparaison"}), 500
        
        verified = bool(similarity >= threshold)
        logger.debug(f"[{request_id}] 📊 Threshold: {threshold} | Résultat: {similarity:.4f} >= {threshold} = {verified}")
        
        # Logs statistiques
        stats["total"] += 1
        if verified:
            stats["accepted"] += 1
            logger.info(f"[{request_id}] ✅ VÉRIFICATION ACCEPTÉE")
        else:
            stats["rejected"] += 1
            logger.warning(f"[{request_id}] ❌ VÉRIFICATION REJETÉE")
        
        t_total = (time.perf_counter() - t_start) * 1000
        logger.info(f"[{request_id}] ⏱️  Temps total: {t_total:.0f}ms")
        logger.info(f"[{request_id}] 📈 Stats session: {stats['accepted']}/{stats['total']} acceptées, {stats['rejected']} rejetées, {stats['no_face']} sans visage")
        logger.info("=" * 80 + "\n")

        return jsonify({
            "verified": verified,
            "confidence": round(float(similarity) * 100, 2),
            "processing_ms": round(t_total, 1),
            "card_id": card_id
        }), 200

    except Exception as e:
        t_total = (time.perf_counter() - t_start) * 1000
        logger.error(f"[{request_id}] 💥 ERREUR CRITIQUE: {str(e)}", exc_info=True)
        logger.error(f"[{request_id}] ⏱️  Temps avant erreur: {t_total:.0f}ms")
        logger.error("=" * 80 + "\n")
        traceback.print_exc()
        return jsonify({"success": False, "error": "Erreur interne serveur"}), 500

@app.route('/embed', methods=['POST'])
def extract_embedding_route():
    """Endpoint qui reçoit une image Base64 et retourne les embeddings."""
    request_id = str(uuid.uuid4())[:8]
    logger.info("=" * 80)
    logger.info(f"[{request_id}] 📨 Nouvelle requête /embed")
    logger.info("=" * 80)

    data = request.json
    if not data or 'image_base64' not in data:
        logger.warning(f"[{request_id}] ⚠️ image_base64 manquante")
        return jsonify({"error": "image_base64 manquante"}), 400

    image_base64 = data['image_base64']
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]

    try:
        embedding = extract_embedding_from_image(image_base64)
        logger.info(f"[{request_id}] ✅ Embeddings extraits ({len(embedding)} dimensions)")
        return jsonify({"embedding": embedding.tolist()}), 200

    except ValueError as e:
        logger.warning(f"[{request_id}] ⚠️ Erreur Base64 ou visage non détecté: {str(e)}")
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        logger.error(f"[{request_id}] ❌ Erreur interne /embed: {str(e)}", exc_info=True)
        return jsonify({"error": "Erreur interne serveur"}), 500


@app.route('/health', methods=['GET'])
@app.route('/', methods=['GET'])
def health():
    """Health check"""
    logger.debug("🏥 Health check")
    return jsonify({
        "status": "active",
        "api": "Facial Recognition",
        "workers_online": True,
        "stats": stats
    }), 200

@app.errorhandler(404)
def not_found(e):
    """Route non trouvée"""
    logger.warning(f"404 - Route non trouvée: {request.path}")
    return jsonify({"error": "Route non trouvée"}), 404

@app.errorhandler(500)
def server_error(e):
    """Erreur serveur"""
    logger.error(f"500 - Erreur serveur: {str(e)}", exc_info=True)
    return jsonify({"error": "Erreur interne serveur"}), 500

if __name__ == '__main__':
    try:
        logger.info("🚀 Démarrage du serveur Flask...")
        logger.info(f"📍 http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("⏹️  Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.critical(f"Erreur démarrage: {str(e)}", exc_info=True)