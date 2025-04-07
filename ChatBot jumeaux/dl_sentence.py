from huggingface_hub import snapshot_download
import os

# Définir le chemin de sauvegarde
model_path = "./models/paraphrase-multilingual-mpnet-base-v2"

# Télécharger explicitement tous les fichiers du modèle
snapshot_download(
    repo_id="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    local_dir=model_path,
    ignore_patterns=["*.md", "*.git*"]  # Ignorer les fichiers non essentiels
)

# Vérifier que les fichiers essentiels sont présents
files = os.listdir(model_path)
print(f"Fichiers téléchargés: {files}")
if "pytorch_model.bin" in files:
    print("Le fichier pytorch_model.bin est présent!")
else:
    print("ERREUR: pytorch_model.bin manquant!")