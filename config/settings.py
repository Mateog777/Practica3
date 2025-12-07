# config/settings.py
from pathlib import Path
import os
import torch
from dotenv import load_dotenv

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==========================
#  SECCIÓN GAN VEHÍCULOS
# ==========================

# Dataset local (si quieres re-entrenar en VS Code)
VEHICLES_DATA_DIR = PROJECT_ROOT / "data" / "vehicles"

# Carpeta donde se guardan los modelos de la GAN
VEHICLES_MODELS_DIR = PROJECT_ROOT / "models" / "vehicles"
VEHICLES_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Carpeta donde se guardan las imágenes generadas
VEHICLES_IMAGES_DIR = PROJECT_ROOT / "generated_images" / "vehicles"
VEHICLES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoints esperados (los mismos nombres que guardas en Kaggle)
VEH_BEST_CKPT = VEHICLES_MODELS_DIR / "generator_vehicles_best.pth"
VEH_FINAL_CKPT = VEHICLES_MODELS_DIR / "generator_vehicles_final.pth"

# Hiperparámetros DCGAN
IMAGE_SIZE = 64
BATCH_SIZE = 64
NZ = 100      # dimensión del vector de ruido
NGF = 64
NDF = 64

NUM_EPOCHS_VEHICLES = 150   # puedes bajarlo para pruebas rápidas
LR_G = 0.0002
LR_D = 0.0001
BETA1 = 0.5

# Dispositivo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
#  (LO DEMÁS: LLMs, DB, ETC.)
# ==========================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.1-70b-versatile")
