import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from PIL import Image
import numpy as np
import os

# --- CONFIGURATION ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f" Using device: {DEVICE}")

WIDTH = 512
HEIGHT = 512
FPS = 30
TOTAL_FRAMES = 400
STEPS_PER_FRAME = 15

# Folders
INPUT_FOLDER = "images"
OUTPUT_FOLDER = "outputs"

# --- PHYSICS MAP LOADER ---
def load_parameter_map(image_path):
    # Load and resize image
    try:
        img = Image.open(image_path).convert('L') 
        img = img.resize((WIDTH, HEIGHT))
    except Exception as e:
        print(f" Error loading image: {e}")
        return None, None
    
    img_data = np.array(img) / 255.0 
    param_map = torch.tensor(img_data, dtype=torch.float32, device=DEVICE)
    
    # Map Brightness to Physics Constants
    # Feed Rate: 0.030 (Dark) -> 0.060 (Bright)
    feed_map = 0.030 + (param_map * (0.060 - 0.030))
    # Kill Rate: 0.060 (Dark) -> 0.062 (Bright)
    kill_map = 0.060 + (param_map * (0.062 - 0.060))
    
    return feed_map.unsqueeze(0).unsqueeze(0), kill_map.unsqueeze(0).unsqueeze(0)

# --- SIMULATION ENGINE ---
class BioPainter:
    def __init__(self, feed_map, kill_map):
        self.feed_map = feed_map
        self.kill_map = kill_map
        
        self.u = torch.ones(1, 1, HEIGHT, WIDTH, device=DEVICE)
        self.v = torch.zeros(1, 1, HEIGHT, WIDTH, device=DEVICE)
        
        # Random seeds
        self.v += torch.rand_like(self.v) * 0.2
        
        # Laplacian Kernel
        k = torch.tensor([[0.05, 0.2, 0.05],
                          [0.2, -1.0, 0.2],
                          [0.05, 0.2, 0.05]], device=DEVICE)
        self.kernel = k.unsqueeze(0).unsqueeze(0).float()

    def step(self):
        lu = F.conv2d(self.u, self.kernel, padding=1)
        lv = F.conv2d(self.v, self.kernel, padding=1)
        uvv = self.u * self.v * self.v
        
        du = (1.0 * lu) - uvv + (self.feed_map * (1 - self.u))
        dv = (0.5 * lv) + uvv - ((self.feed_map + self.kill_map) * self.v)
        
        self.u += du * 1.0
        self.v += dv * 1.0
        self.u = torch.clamp(self.u, 0, 1)
        self.v = torch.clamp(self.v, 0, 1)

    def get_frame(self):
        return self.v[0, 0].cpu().numpy()

# --- JOB RUNNER ---
def process_image(filename):
    input_path = os.path.join(INPUT_FOLDER, filename)
    
    name_only = os.path.splitext(filename)[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"{name_only}.mp4")

    print(f"\n Processing: {filename} -> {output_path}")

    # 1. Load Maps
    feed, kill = load_parameter_map(input_path)
    if feed is None: return 

    # 2. Init Sim
    sim = BioPainter(feed, kill)

    # 3. Setup Plot
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    img = ax.imshow(sim.get_frame(), cmap='twilight_shifted', vmin=0, vmax=0.6)
    
    def update(frame):
        for _ in range(STEPS_PER_FRAME):
            sim.step()
        img.set_array(sim.get_frame())
        return [img]

    # 4. Render
    ani = animation.FuncAnimation(fig, update, frames=tqdm(range(TOTAL_FRAMES), desc="Rendering"), interval=20)
    
    try:
        ani.save(output_path, writer='ffmpeg', fps=FPS, bitrate=5000)
        print(f"Success: Saved to {output_path}")
    except Exception as e:
        print(f"FFmpeg error: {e}")
        gif_path = output_path.replace(".mp4", ".gif")
        ani.save(gif_path, writer='pillow', fps=15)
        print(f"Saved GIF instead: {gif_path}")
    
    plt.close(fig)

# --- MAIN LOOP ---
if __name__ == "__main__":

    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created '{INPUT_FOLDER}/' directory. Please put images there!")
        exit()
        
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 2. Find all images
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"No images found in '{INPUT_FOLDER}/'. Add some .jpg or .png files!")
    else:
        print(f"Found {len(files)} images. Starting batch job...")
        for f in files:
            process_image(f)
        print("\n✨ All jobs finished.")