import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

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
# 2. Inference Execution Engine
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="GalaxEye SAR-to-EO Inference Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input SAR patches")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated EO patches")
    parser.add_argument("--weights", type=str, required=True, help="Path to generator weights")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model and load weights completely locally
    model = SAR2EOGenerator().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    # Normalization pipelines mirroring the training dataloader
    transform_in = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Process files
    valid_extensions = {".png", ".jpg", ".jpeg"}
    files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    print(f"Starting inference on {len(files)} files...")

    with torch.no_grad():
        for filename in files:
            input_path = os.path.join(args.input_dir, filename)
            output_path = os.path.join(args.output_dir, filename)

            # Load 8-bit SAR as 1-channel grayscale
            sar_img = Image.open(input_path).convert("L")
            if sar_img.size != (256, 256):
                sar_img = sar_img.resize((256, 256), Image.BILINEAR)

            # Transform and run inference
            input_tensor = transform_in(sar_img).unsqueeze(0).to(device)
            output_tensor = model(input_tensor).squeeze(0).cpu()

            # Denormalize from [-1, 1] to [0, 1]
            output_tensor = (output_tensor + 1.0) / 2.0
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

            # Convert to PIL and save
            output_img = transforms.ToPILImage()(output_tensor)
            output_img.save(output_path)

    print("Inference completed successfully.")

if __name__ == "__main__":
    main()
