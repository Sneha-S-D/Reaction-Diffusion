
# Reaction-Diffusion: Spatially-Varying Control


https://github.com/user-attachments/assets/91da94f0-41a7-4e94-b860-4da8c3c67a35

> *Above: Emergent biological texture synthesis guided by spatial parameter mapping. A static image determines the local laws of physics, forcing "digital bacteria" to self-organize into the target shape.*

##  Overview
This project implements a **differentiable physics engine** in PyTorch to simulate the Gray-Scott Reaction-Diffusion system. 

Unlike standard simulations that use global constants (creating the same texture everywhere), this engine introduces **spatially-varying coefficients**. By mapping semantic image features (brightness) to local bifurcation parameters, we can constrain chaotic Turing patterns to self-organize into complex target geometries.

Essentially, it is a **"Bio-Painter"**—instead of drawing pixels, you terraform a mathematical environment where patterns grow exactly where you want them.

##  Key Features
* **Differentiable Physics:** Implements Laplacian operators as fixed-function convolutional kernels (`F.conv2d`) within the PyTorch computational graph.
* **Spatially-Varying Control:** Feed ($f$) and Kill ($k$) rates are treated as tensor fields rather than scalars. This allows for pixel-perfect control over local pattern formation.
* **Hardware Accelerated:** Optimized for Apple Silicon (M-Series) using Metal Performance Shaders (MPS) for real-time high-resolution simulation ($512^2$ grid at 60 FPS).
* **Automated Batch Processing:** The system automatically scans input directories, processes images, and renders video outputs without manual intervention.

## How It Works (The Science)

The simulation solves the discretized Gray-Scott Partial Differential Equations (PDEs):

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + f(1-u)
$$
$$
\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (f+k)v
$$

Where:
* **$u$**: The "Food" chemical.
* **$v$**: The "Bacteria" chemical (which reproduces by eating $u$).
* **$f$ (Feed Rate)**: How fast food is added.
* **$k$ (Kill Rate)**: How fast bacteria die.

### The "Spatial Control" Twist
In a normal simulation, $f$ and $k$ are single numbers. In this project, they are **maps** derived from your input image:

| Image Region | Brightness | Physics Mode | Resulting Texture |
| :--- | :--- | :--- | :--- |
| **Shadows** | Dark ($\approx 0.0$) | Low Feed ($f \approx 0.030$) | **Spots / Dots** |
| **Highlights** | Bright ($\approx 1.0$) | High Feed ($f \approx 0.060$) | **Stripes / Solitons** |

This creates a **Turing Instability** where the chemicals have no choice but to form spots in the dark areas and stripes in the bright areas, naturally reconstructing your image out of living texture.

##  Usage

### 1. Installation
Clone the repo and install the dependencies (PyTorch, NumPy, Matplotlib, FFmpeg).
```bash
pip install -r requirements.txt

```

### 2. Add Images

Place any `.jpg` or `.png` images into the `images/` folder. High-contrast images (silhouettes, portraits, logos) work best.

### 3. Run the Simulation

```bash
python main.py

```

The script will automatically:

1. Detect images in `images/`.
2. Generate the biological growth animation.
3. Save high-quality `.mp4` videos to the `outputs/` folder.

## Project Structure

```text
reaction-diffusion-control/
├── assets/                 # Demo assets for README
├── images/                 # PUT YOUR INPUT IMAGES HERE
├── outputs/                # Generated videos appear here (ignored by Git)
├── src/
│   └── main.py             # Core physics engine
├── .gitignore              # Keeps repo clean
└── requirements.txt        # Dependencies

```

##  Configuration

You can tweak the physics mapping in `main.py` to discover new patterns:

```python
# Map Brightness to Physics Constants
# Try changing these ranges to see different biological behaviors!
feed_map = 0.030 + (param_map * (0.060 - 0.030)) 
kill_map = 0.060 + (param_map * (0.062 - 0.060))

```

## Citations & Theory

* **Gray-Scott Model:** Pearson, J. E. (1993). *Complex patterns in a simple system.* Science.
* **Turing Patterns:** Turing, A. M. (1952). *The Chemical Basis of Morphogenesis.*

---


