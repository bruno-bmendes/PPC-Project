import unicodedata
import re
import base64

def normalize_title(texto: str) -> str:
    # Converte para minúsculas
    texto = texto.lower()

    # Remove acentos
    texto = unicodedata.normalize("NFKD", texto)
    texto = texto.encode("ASCII", "ignore").decode("utf-8")

    # Substitui espaços e hífens por underscores
    texto = re.sub(r"[ \-]+", "_", texto)

    # Remove qualquer caractere que não seja alfanumérico ou underscore
    texto = re.sub(r"[^\w_]", "", texto)

    return texto

def get_base_64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()