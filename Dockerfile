# ============================================================
# Dockerfile — Recognition Service (FaceNet + Anti-Spoofing)
# ============================================================

# 1. Base Python légère (compatible x86_64 et ARM64)
FROM python:3.10-slim

# 2. Dossier de travail dans le conteneur
WORKDIR /app

# 3. Dépendances système nécessaires pour OpenCV et PIL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Installer PyTorch CPU uniquement (réduit l'image de ~9 Go → ~1 Go)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# 5. Copier et installer les autres dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copier le code source et le modèle anti-spoofing
COPY app.py .
COPY anti_spoofing.py .
COPY MiniFASNetV2.onnx .

# 7. Pré-télécharger le modèle FaceNet (évite le délai au 1er appel)
#    Les modèles sont mis en cache dans ~/.cache/torch/
RUN python -c "\
from facenet_pytorch import MTCNN, InceptionResnetV1; \
print('Téléchargement MTCNN...'); MTCNN(); \
print('Téléchargement FaceNet VGGFace2...'); InceptionResnetV1(pretrained='vggface2'); \
print('Modèles prêts !')"

# 8. Port exposé par l'API Flask
EXPOSE 5000

# 9. Variables d'environnement utiles
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 10. Démarrer l'API avec Gunicorn (production-ready)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "2", \
     "--timeout", "120", \
     "--log-level", "info", \
     "app:app"]
