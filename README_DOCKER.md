# 🚀 Guide — Lancer le Recognition Server avec Docker Hub

---

## 👨‍💻 Pour le développeur (toi) — Publier l'image

### Étape 1 — Se connecter à Docker Hub

> Créer un compte gratuit sur https://hub.docker.com si besoin.

```bash
docker login
```

### Étape 2 — Builder l'image

```bash
# Remplace "ton_username" par ton pseudo Docker Hub (ex: cash123)
docker build -t ton_username/recognition-server:latest .
```

### Étape 3 — Pousser l'image sur Docker Hub

```bash
docker push ton_username/recognition-server:latest
```

✅ L'image est maintenant disponible publiquement sur Docker Hub !

---

## 👤 Pour ton camarade — Récupérer et lancer l'image

> Il doit avoir **Docker Desktop** installé : https://www.docker.com/products/docker-desktop/

### 1. Télécharger l'image depuis Docker Hub

```bash
# Remplace "ton_username" par le pseudo Docker Hub du développeur
docker pull ton_username/recognition-server:latest
```

### 2. Lancer le serveur

```bash
docker run -d -p 5000:5000 --name recognition-api ton_username/recognition-server:latest
```

### 3. Vérifier que ça fonctionne

Ouvre ton navigateur :

```
http://localhost:5000/health
```

Réponse attendue :
```json
{"status": "online", "model": "FaceNet VGGFace2", "device": "cpu"}
```

---

## 🔌 Endpoints de l'API

| Route | Méthode | Description |
|---|---|---|
| `/health` | GET | Vérifie si le serveur est actif |
| `/verify` | POST | Vérifie un visage (1:1) |
| `/stats` | GET | Statistiques de session |

### Exemple de requête `/verify`

```json
POST http://localhost:5000/verify
Content-Type: application/json

{
  "profile_url": "https://exemple.com/photo_profil.jpg",
  "face_image": "<base64_de_la_photo_capturée>",
  "card_id": "EMP001",
  "threshold": 0.65
}
```

---

## 📋 Commandes utiles

| Commande | Description |
|---|---|
| `docker ps` | Voir les conteneurs en cours |
| `docker logs recognition-api` | Voir les logs du serveur |
| `docker stop recognition-api` | Arrêter le serveur |
| `docker start recognition-api` | Redémarrer le serveur |
| `docker rm recognition-api` | Supprimer le conteneur |

---

## ⚠️ Notes importantes

- Le port **5000** doit être libre sur la machine
- Si le port est occupé, changer : `-p 5001:5000` (accès via `localhost:5001`)
- L'image pèse environ **2-3 Go** (modèle FaceNet inclus)
- Le 1er démarrage peut prendre **30-60 secondes** (chargement du modèle IA)

---

## 🐛 Problèmes courants

**Port déjà utilisé :**
```bash
docker run -d -p 5001:5000 --name recognition-api ton_username/recognition-server:latest
```

**Voir les erreurs :**
```bash
docker logs -f recognition-api
```

**Relancer depuis zéro :**
```bash
docker stop recognition-api && docker rm recognition-api
docker run -d -p 5000:5000 --name recognition-api ton_username/recognition-server:latest
```

**Mettre à jour l'image (nouvelle version) :**
```bash
docker pull ton_username/recognition-server:latest
docker stop recognition-api && docker rm recognition-api
docker run -d -p 5000:5000 --name recognition-api ton_username/recognition-server:latest
```
