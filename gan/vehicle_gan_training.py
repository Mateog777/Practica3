# gan/vehicle_gan_training.py

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from config.settings import (
    VEHICLES_DATA_DIR,
    VEHICLES_MODELS_DIR,
    VEHICLES_IMAGES_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NZ,
    NGF,
    NDF,
    NUM_EPOCHS_VEHICLES,
    LR_G,
    LR_D,
    BETA1,
    DEVICE,
)
from gan.gan_models import Generator, Discriminator, weights_init


class VehicleDCGANExperiment:
    """
    'Taller' de DCGAN para el dataset de vehículos.
    - Permite re-entrenar en VS Code.
    - Guarda mejor modelo y modelo final.
    - Guarda grids de imágenes reales/generadas y curva de pérdidas.
    - Si los archivos ya existen, los borra antes de escribir.
    """

    def __init__(self) -> None:
        self.device = DEVICE

        # --- rutas de salida ---
        self.models_dir = Path(VEHICLES_MODELS_DIR)
        self.images_dir = Path(VEHICLES_IMAGES_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.best_ckpt_path = self.models_dir / "generator_vehicles_best.pth"
        self.final_ckpt_path = self.models_dir / "generator_vehicles_final.pth"
        self.loss_plot_path = self.images_dir / "vehicle_gan_losses.png"
        self.real_grid_path = self.images_dir / "real_vehicles_grid.png"

        # --- dataset y dataloader ---
        if not VEHICLES_DATA_DIR.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta de dataset local: {VEHICLES_DATA_DIR}\n"
                "Copia el dataset de Kaggle en data/vehicles (ImageFolder)."
            )

        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = datasets.ImageFolder(root=str(VEHICLES_DATA_DIR), transform=transform)
        self.classes = dataset.classes
        self.dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2 if os.name != "nt" else 0,  # en Windows a veces da problemas
        )

        print("✅ Dataset de vehículos cargado")
        print("   Carpeta:", VEHICLES_DATA_DIR)
        print("   Clases:", self.classes)
        print("   Nº imágenes:", len(dataset))

        # --- modelos ---
        self.nz = NZ
        self.netG = Generator(nz=NZ, ngf=NGF, nc=3).to(self.device)
        self.netD = Discriminator(ndf=NDF, nc=3).to(self.device)
        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        # pérdida y optimizadores
        self.criterion = nn.BCELoss()
        self.real_label = 0.9
        self.fake_label = 0.0

        self.optD = optim.Adam(self.netD.parameters(), lr=LR_D, betas=(BETA1, 0.999))
        self.optG = optim.Adam(self.netG.parameters(), lr=LR_G, betas=(BETA1, 0.999))

        # entrenamiento
        self.num_epochs = NUM_EPOCHS_VEHICLES
        self.fixed_noise = torch.randn(64, NZ, 1, 1, device=self.device)
        self.G_losses: List[float] = []
        self.D_losses: List[float] = []
        self.best_g_loss = float("inf")

    # ------------------------------------------------------------------
    # Utilidades para guardar (borrando si existe)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_remove(path: Path) -> None:
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # Visualización de lote real (para informe)
    # ------------------------------------------------------------------
    def save_real_grid(self) -> None:
        images, _ = next(iter(self.dataloader))
        grid = utils.make_grid(images[:64], padding=2, normalize=True)

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Imágenes reales - Vehículos")
        plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))

        self._safe_remove(self.real_grid_path)
        fig.savefig(self.real_grid_path, bbox_inches="tight")
        plt.close(fig)
        print(f"📸 Grid de imágenes reales guardado en: {self.real_grid_path}")

    # ------------------------------------------------------------------
    # Guardar imágenes generadas
    # ------------------------------------------------------------------
    def save_fake_grid(self, epoch: int) -> None:
        self.netG.eval()
        with torch.no_grad():
            fake = self.netG(self.fixed_noise).detach().cpu()
        grid = utils.make_grid(fake, padding=2, normalize=True)

        fig = plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title(f"Imágenes generadas (vehículos) - Época {epoch}")
        plt.imshow(np.transpose(grid, (1, 2, 0)))

        out_path = self.images_dir / f"fake_vehicles_epoch_{epoch}.png"
        self._safe_remove(out_path)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"🚗 Imagen generada guardada en: {out_path}")

        self.netG.train()

    # ------------------------------------------------------------------
    # Guardar checkpoints
    # ------------------------------------------------------------------
    def save_best_checkpoint(self, epoch: int, g_loss: float) -> None:
        if g_loss < self.best_g_loss:
            self.best_g_loss = g_loss
            self._safe_remove(self.best_ckpt_path)
            torch.save(
                {
                    "model_state_dict": self.netG.state_dict(),
                    "nz": self.nz,
                    "ngf": NGF,
                    "image_size": IMAGE_SIZE,
                    "epoch": epoch,
                    "G_losses": self.G_losses,
                    "D_losses": self.D_losses,
                },
                self.best_ckpt_path,
            )
            print(f"💾 Mejor modelo actualizado en época {epoch} con Loss_G={g_loss:.4f}")

    def save_final_checkpoint(self) -> None:
        self._safe_remove(self.final_ckpt_path)
        torch.save(
            {
                "model_state_dict": self.netG.state_dict(),
                "nz": self.nz,
                "ngf": NGF,
                "image_size": IMAGE_SIZE,
                "epoch": self.num_epochs,
                "G_losses": self.G_losses,
                "D_losses": self.D_losses,
            },
            self.final_ckpt_path,
        )
        print(f"✅ Modelo final guardado en: {self.final_ckpt_path}")

    # ------------------------------------------------------------------
    # Guardar curva de pérdidas
    # ------------------------------------------------------------------
    def save_loss_plot(self) -> None:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(self.G_losses, label="Loss G")
        plt.plot(self.D_losses, label="Loss D")
        plt.xlabel("Época")
        plt.ylabel("Pérdida")
        plt.title("Curvas de entrenamiento DCGAN - Vehículos")
        plt.legend()

        self._safe_remove(self.loss_plot_path)
        fig.savefig(self.loss_plot_path, bbox_inches="tight")
        plt.close(fig)
        print(f"📉 Curva de pérdidas guardada en: {self.loss_plot_path}")

    # ------------------------------------------------------------------
    # Loop de entrenamiento principal
    # ------------------------------------------------------------------
    def train(self) -> None:
        print("🚀 Inicio de entrenamiento DCGAN de vehículos")
        self.save_real_grid()

        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            for i, (real_images, _) in enumerate(self.dataloader):
                # --------------------
                # Entrenar Discriminador
                # --------------------
                self.netD.zero_grad()

                real_images = real_images.to(self.device)
                b_size = real_images.size(0)

                # Reales
                real_labels = torch.full(
                    (b_size,),
                    self.real_label,
                    dtype=torch.float,
                    device=self.device,
                )
                output_real = self.netD(real_images)
                errD_real = self.criterion(output_real, real_labels)
                errD_real.backward()
                D_x = output_real.mean().item()

                # Falsas
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                fake_images = self.netG(noise)
                fake_labels = torch.full(
                    (b_size,),
                    self.fake_label,
                    dtype=torch.float,
                    device=self.device,
                )
                output_fake = self.netD(fake_images.detach())
                errD_fake = self.criterion(output_fake, fake_labels)
                errD_fake.backward()
                D_G_z1 = output_fake.mean().item()

                errD = errD_real + errD_fake
                self.optD.step()

                # --------------------
                # Entrenar Generador
                # --------------------
                self.netG.zero_grad()
                # queremos que el discriminador crea que las falsas son reales
                target_labels_for_g = torch.full(
                    (b_size,),
                    self.real_label,
                    dtype=torch.float,
                    device=self.device,
                )
                output_for_g = self.netD(fake_images)
                errG = self.criterion(output_for_g, target_labels_for_g)
                errG.backward()
                D_G_z2 = output_for_g.mean().item()
                self.optG.step()

            # fin época
            self.G_losses.append(errG.item())
            self.D_losses.append(errD.item())

            print(
                f"[{epoch}/{self.num_epochs}] "
                f"Loss_D: {errD.item():.4f}  Loss_G: {errG.item():.4f}  "
                f"D(x): {D_x:.4f}  D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}"
            )

            # visualizar y guardar cada 10 épocas
            if epoch % 10 == 0 or epoch == 1:
                self.save_fake_grid(epoch)

            # actualizar mejor modelo
            self.save_best_checkpoint(epoch, errG.item())

        end_time = time.time()
        print(f"⏱ Entrenamiento terminado en {(end_time - start_time)/60:.2f} minutos")

        # guardar modelo final y curvas
        self.save_final_checkpoint()
        self.save_loss_plot()


def run_vehicle_gan_training() -> None:
    experiment = VehicleDCGANExperiment()
    experiment.train()


if __name__ == "__main__":
    run_vehicle_gan_training()
