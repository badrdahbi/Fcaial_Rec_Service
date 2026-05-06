import base64
from embeddings import extract_embedding_from_image

# Test avec Base64 direct
try:
    # Colle ta chaîne Base64 ici (remplace cette valeur)
    base64_image = "data:image/jpeg;base64,/9j/4AAQ..."

    print(f"Test avec Base64 ({len(base64_image)} caractères)")

    embedding = extract_embedding_from_image(base64_image)
    print("✅ Embedding extrait avec succès !")
    print(f"Shape: {embedding.shape}")
    print(f"Type: {type(embedding)}")
    print(f"Premiers 5 valeurs: {embedding[:5]}")

except ValueError as e:
    print(f"❌ Erreur: {e}")
