# Utilise Python allégé
FROM python:3.10-slim

# Dossier de travail dans Docker
WORKDIR /app

# Outils système nécessaires pour OpenCV et Pillow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer PyTorch (Version CPU pour gagner 4Go d'espace)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY . .

# Exposer le port
EXPOSE 5000

# Lancer l'API via Gunicorn (2 workers pour gérer plusieurs requêtes en même temps)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
