import base64
import io

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


MODEL_CACHE = {
    "mtcnn": None,
    "resnet": None,
    "device": None,
}


def _load_models(device: torch.device = None):
    """Charge MTCNN et ResNet si nécessaire."""
    if MODEL_CACHE["mtcnn"] is not None and MODEL_CACHE["resnet"] is not None:
        return MODEL_CACHE["mtcnn"], MODEL_CACHE["resnet"], MODEL_CACHE["device"]

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    MODEL_CACHE["mtcnn"] = mtcnn
    MODEL_CACHE["resnet"] = resnet
    MODEL_CACHE["device"] = device
    return mtcnn, resnet, device


def _load_pil_image(image_input):
    """Convertit une chaîne Base64 en objet PIL.Image RGB."""
    if not isinstance(image_input, str):
        raise ValueError("L'entrée doit être une chaîne Base64.")

    # Nettoyer le Data URI si présent
    if image_input.startswith("data:"):
        _, encoded = image_input.split(",", 1)
    else:
        encoded = image_input

    try:
        image_bytes = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as ex:
        raise ValueError("Chaîne Base64 invalide ou format image non supporté.") from ex


def extract_embedding_from_image(image_input, device: torch.device = None) -> np.ndarray:
    """Extrait l'embedding de visage d'une image en Base64.

    L'entrée doit être une chaîne Base64 représentant l'image.
    Supporte les Data URIs (data:image/jpeg;base64,...).
    """
    mtcnn, resnet, device = _load_models(device)
    image = _load_pil_image(image_input)

    # Utiliser MTCNN directement pour extraire le visage
    face_tensor = mtcnn(image)
    
    if face_tensor is None:
        raise ValueError("Aucun visage détecté dans l'image.")
    with torch.no_grad():
        embedding = resnet(face_tensor.unsqueeze(0).to(device))

    return embedding.detach().cpu().numpy().flatten()


if __name__ == "__main__":
    print("Cette bibliothèque fournit extract_embedding_from_image(image_input).")
