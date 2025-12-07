# gan/vehicle_gan_inference.py

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from torchvision.utils import save_image

from config.settings import (
    VEH_BEST_CKPT,
    VEH_FINAL_CKPT,
    VEHICLES_IMAGES_DIR,
    NZ,
    NGF,
    IMAGE_SIZE,
    DEVICE,
)
from gan.gan_models import Generator


# cache simple para no cargar el modelo cada vez
_GENERATOR_CACHE: dict[str, Generator] = {}


def _load_checkpoint(kind: str) -> Path:
    """
    kind: 'best' o 'final'
    """
    kind = kind.lower()
    if kind == "best":
        ckpt_path = Path(VEH_BEST_CKPT)
    elif kind == "final":
        ckpt_path = Path(VEH_FINAL_CKPT)
    else:
        raise ValueError("kind debe ser 'best' o 'final'")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint '{kind}': {ckpt_path}\n"
            "Copia aquí el archivo .pth descargado de Kaggle."
        )
    return ckpt_path


def get_vehicle_generator(kind: str = "best") -> Generator:
    """
    Carga (y cachea) el generador de vehículos entrenado.
    """
    kind = kind.lower()
    if kind in _GENERATOR_CACHE:
        return _GENERATOR_CACHE[kind]

    ckpt_path = _load_checkpoint(kind)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    nz = ckpt.get("nz", NZ)
    ngf = ckpt.get("ngf", NGF)

    netG = Generator(nz=nz, ngf=ngf, nc=3).to(DEVICE)
    netG.load_state_dict(ckpt["model_state_dict"])
    netG.eval()

    _GENERATOR_CACHE[kind] = netG
    print(f"✅ Generador de vehículos ({kind}) cargado desde: {ckpt_path}")
    return netG


def generate_vehicle_images(
    kind: str = "best",
    num_images: int = 4,
    base_filename: str | None = None,
) -> List[str]:
    """
    Genera `num_images` imágenes de vehículos usando el modelo `kind`
    ('best' o 'final') y las guarda en generated_images/vehicles.

    Devuelve una lista de rutas (strings) de las imágenes generadas.
    """

    VEHICLES_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    netG = get_vehicle_generator(kind)
    nz = netG.nz if hasattr(netG, "nz") else NZ

    # ruido
    noise = torch.randn(num_images, nz, 1, 1, device=DEVICE)

    with torch.no_grad():
        fake = netG(noise).cpu()

    paths: List[str] = []
    kind_tag = "best" if kind.lower() == "best" else "final"

    for idx in range(num_images):
        filename = (
            base_filename
            if base_filename is not None and num_images == 1
            else f"vehicle_{kind_tag}_{idx+1}.png"
        )

        out_path = VEHICLES_IMAGES_DIR / filename

        # borrar si ya existe
        if out_path.exists():
            out_path.unlink()

        # las imágenes de la GAN suelen estar en [-1,1] → normalizar a [0,1]
        img = (fake[idx] + 1) / 2
        save_image(img, out_path)
        paths.append(str(out_path))

    print(f"🚗 Imágenes generadas ({kind_tag}):")
    for p in paths:
        print("   ", p)

    return paths
