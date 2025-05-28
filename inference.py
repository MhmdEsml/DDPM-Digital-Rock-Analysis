import torch
import numpy as np
import gdown
from tqdm import tqdm
import os
import imageio
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Generate images using DDPM")
parser.add_argument("--type", type=str, choices=["sandstone", "carbonate"], help="Type of material: sandstone or carbonate", required=True)
parser.add_argument("--num_generate_images", type=int, help="Number of images to generate", default=8)
parser.add_argument("--num_loop", type=int, help="Number of loops", default=1)
args = parser.parse_args()

while True:
    type = args.type
    if type == 'carbonate':
        file_id = '1qX8tad72YGrlClVD1PcUhF7d1KQLIYKb'
        break
    elif type == 'sandstone':
        file_id = '1YsAb5Rmevolc39myndNjoIyPofeGgUtu'
        break
    else:
        print("Invalid type! Please choose between 'sandstone' and 'carbonate'.")

url_template = 'https://drive.google.com/uc?id={}'
gdown.download(url_template.format(file_id), 'model.pth')

IMG_SIZE = 256
NUM_GENERATE_IMAGES = args.num_generate_images
NUM_LOOP = args.num_loop
NUM_TIMESTEPS = 500
MIXED_PRECISION = "fp16"
device = "cuda" if torch.cuda.is_available() else "cpu"

class Diffusion:
    def __init__(self, beta_start=1e-4, beta_end=0.02, timesteps=1000, clip_min=-1.0, clip_max=1.0):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.clip_min = clip_min
        self.clip_max = clip_max

        betas = np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)
        self.num_timesteps = int(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)
        self.alphas_cumprod_prev = torch.tensor(alphas_cumprod_prev, dtype=torch.float32, device=device)

        self.sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=torch.float32, device=device)
        self.sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=torch.float32, device=device)
        self.log_one_minus_alphas_cumprod = torch.tensor(np.log(1.0 - alphas_cumprod), dtype=torch.float32, device=device)
        self.sqrt_recip_alphas_cumprod = torch.tensor(np.sqrt(1.0 / alphas_cumprod), dtype=torch.float32, device=device)
        self.sqrt_recipm1_alphas_cumprod = torch.tensor(np.sqrt(1.0 / alphas_cumprod - 1), dtype=torch.float32, device=device)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = torch.tensor(posterior_variance, dtype=torch.float32, device=device)

        self.posterior_log_variance_clipped = torch.tensor(np.log(np.maximum(posterior_variance, 1e-20)), dtype=torch.float32, device=device)

        self.posterior_mean_coef1 = torch.tensor(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod), dtype=torch.float32, device=device)
        self.posterior_mean_coef2 = torch.tensor((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod), dtype=torch.float32, device=device)

    def _extract(self, a, t, x_shape):
        batch_size = x_shape[0]
        out = a[t].view(batch_size, 1, 1, 1)
        return out

    def q_mean_variance(self, x_start, t):
        x_start_shape = x_start.shape
        mean = self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
        variance = self._extract(1.0 - self.alphas_cumprod, t, x_start_shape)
        log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start_shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise):
        x_start_shape = x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod, t, x_start_shape) * x_start
            + self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_shape) * noise
        )

    def predict_start_from_noise(self, x_t, t, noise):
        x_t_shape = x_t.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod, t, x_t_shape) * x_t
            - self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t_shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        x_t_shape = x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t_shape) * x_start
            + self._extract(self.posterior_mean_coef2, t, x_t_shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t_shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t_shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, pred_noise, x, t, clip_denoised=True):
        x_recon = self.predict_start_from_noise(x, t=t, noise=pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, self.clip_min, self.clip_max)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, pred_noise, x, t, clip_denoised=True):
        model_mean, _, model_log_variance = self.p_mean_variance(pred_noise, x=x, t=t, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        nonzero_mask = torch.reshape(1 - (t == 0).to(torch.float32), [x.size(0), 1, 1, 1])
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise

gdf_util = Diffusion(timesteps=NUM_TIMESTEPS)

def generate_images(model, step, img_size=512, img_channels=1, num_images=2, timesteps=1000):

    samples = torch.randn((num_images, img_channels, img_size, img_size), dtype=torch.float32).to(device)

    for t in tqdm(reversed(range(0, timesteps)), desc=f"Generating images at loop {step + 1}", total=timesteps, position=0, leave=True):
        tt = torch.full((num_images,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            pred_noise = model(samples, tt, return_dict=False)[0]

        samples = gdf_util.p_sample(
                pred_noise, samples, tt, clip_denoised=True
            )
    return samples

def main():
    model = torch.load('./model.pth', weights_only=False)
    model = model.to(device)

    if not os.path.exists('./Generated_images'):
        os.makedirs('./Generated_images')

    for j in tqdm(range(NUM_LOOP), position=0, leave=True):
        images = generate_images(model, j, img_size=IMG_SIZE, img_channels=1, num_images=NUM_GENERATE_IMAGES, timesteps=NUM_TIMESTEPS)

        images_processed = (images.cpu().numpy() * 127.5 + 127.5).round().astype("uint8")

        for i in range(1, NUM_GENERATE_IMAGES + 1):
            image_to_save = np.squeeze(images_processed[i-1])
            image_to_save = Image.fromarray(image_to_save, mode='L')

            image_path = os.path.join(f'./Generated_images', f'generated_image_{i + j * NUM_GENERATE_IMAGES:04d}.png')
            imageio.imwrite(image_path, np.array(image_to_save))

    os.system("zip -r ./Generated_images.zip ./Generated_images")

if __name__ == "__main__":
    main()
