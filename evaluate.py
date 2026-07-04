import os
import argparse
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.nn.functional as F
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from torchvision import transforms
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
# pyrefly: ignore [missing-import]
import lpips
# pyrefly: ignore [missing-import]
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
# pyrefly: ignore [missing-import]
from torchmetrics.image.fid import FrechetInceptionDistance

# ==========================================
# 1. Architecture Definition (Self-Contained)
# ==========================================
class BottleNeckSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), dim=1)
        return x

class SAR2EOGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.down1 = UNetDownBlock(in_channels, 64, normalize=False)
        self.down2 = UNetDownBlock(64, 128)
        self.down3 = UNetDownBlock(128, 256)
        self.down4 = UNetDownBlock(256, 512, dropout=0.5)
        self.down5 = UNetDownBlock(512, 512, dropout=0.5)
        self.down6 = UNetDownBlock(512, 512, dropout=0.5)
        self.bottleneck_attention = BottleNeckSelfAttention(512)
        self.up1 = UNetUpBlock(512, 512, dropout=0.5)
        self.up2 = UNetUpBlock(1024, 512, dropout=0.5)
        self.up3 = UNetUpBlock(1024, 256)
        self.up4 = UNetUpBlock(512, 128)
        self.up5 = UNetUpBlock(256, 64)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        latent_space = self.bottleneck_attention(d6)
        u1 = self.up1(latent_space, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        return self.final_layer(u5)

# ==========================================
# 2. Dataset Loader for Evaluation
# ==========================================
class EvalDataset(Dataset):
    def __init__(self, sar_dir, eo_dir, target_size=256):
        self.sar_dir = sar_dir
        self.eo_dir = eo_dir
        self.target_size = target_size
        
        self.sar_files = sorted([f for f in os.listdir(sar_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.sar_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.eo_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.sar_files)

    def __getitem__(self, idx):
        sar_fname = self.sar_files[idx]
        eo_fname = sar_fname.replace('_s1_', '_s2_')
        
        sar_path = os.path.join(self.sar_dir, sar_fname)
        eo_path = os.path.join(self.eo_dir, eo_fname)
        
        sar_img = Image.open(sar_path).convert('L')
        if not os.path.exists(eo_path):
            raise FileNotFoundError(f"Missing matching ground truth EO image: {eo_path}")
            
        eo_img = Image.open(eo_path).convert('RGB')
        
        sar_img = sar_img.resize((self.target_size, self.target_size), Image.BILINEAR)
        eo_img = eo_img.resize((self.target_size, self.target_size), Image.BILINEAR)
        
        sar_tensor = self.sar_normalize(sar_img)
        eo_tensor = self.eo_normalize(eo_img)
        
        return sar_tensor, eo_tensor

# ==========================================
# 3. Main Evaluation Engine
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate SAR-to-EO Generator Model")
    parser.add_argument("--sar_dir", type=str, required=True, help="Directory containing input SAR patches")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth EO patches")
    parser.add_argument("--weights", type=str, required=True, help="Path to generator weights (.pth)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = SAR2EOGenerator().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Load Dataset
    dataset = EvalDataset(args.sar_dir, args.gt_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='vgg').to(device)
    fid_metric = FrechetInceptionDistance(feature=64).to(device)

    total_psnr, total_ssim, total_lpips = 0.0, 0.0, 0.0
    batches = 0

    denorm = lambda x: torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    print("Beginning metric calculation on validation holdout...")
    with torch.no_grad():
        for sar, eo in loader:
            sar, eo = sar.to(device), eo.to(device)
            gen_eo = model(sar)

            # [0, 1] range for PSNR/SSIM
            gen_denorm = denorm(gen_eo)
            eo_denorm = denorm(eo)

            total_psnr += psnr_metric(gen_denorm, eo_denorm).item()
            total_ssim += ssim_metric(gen_denorm, eo_denorm).item()

            # [-1, 1] range for LPIPS
            total_lpips += lpips_metric(gen_eo, eo).mean().item()

            # [0, 255] uint8 range for FID
            gen_uint8 = (gen_denorm * 255).to(torch.uint8)
            eo_uint8 = (eo_denorm * 255).to(torch.uint8)
            fid_metric.update(eo_uint8, real=True)
            fid_metric.update(gen_uint8, real=False)

            batches += 1

    final_psnr = total_psnr / batches
    final_ssim = total_ssim / batches
    final_lpips = total_lpips / batches
    final_fid = fid_metric.compute().item()

    print("\n=== FINAL EVALUATION METRICS ===")
    print(f"PSNR (Pixel-level)  : {final_psnr:.4f}")
    print(f"SSIM (Pixel-level)  : {final_ssim:.4f}")
    print(f"LPIPS (Perceptual)  : {final_lpips:.4f}")
    print(f"FID (Perceptual)    : {final_fid:.4f}")

if __name__ == "__main__":
    main()
