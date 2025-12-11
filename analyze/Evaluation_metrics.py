import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# ===============================================================
# Directories
# ===============================================================
LOW_DIR = "low"
ENH_DIR_BASE = "enhance_baseline"
ENH_DIR_PROP = "enhance_result_loss2"

PATCH_SIZE = 32
NUM_PATCHES = 20
DARK_THRESHOLD = 40
DARK_RATIO = 0.8
GRADIENT_THRESHOLD = 5.0


# ===============================================================
# Utility functions
# ===============================================================
def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def luminance(rgb):
    return rgb[...,0]*0.299 + rgb[...,1]*0.587 + rgb[...,2]*0.114

def is_flat_region(patch):
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    return np.mean(grad_mag) < GRADIENT_THRESHOLD

def sample_dark_patches(luma, patch_size, num_patches):
    h, w = luma.shape
    patches = []
    tries = 0
    while len(patches) < num_patches and tries < num_patches * 20:
        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)
        p = luma[y:y+patch_size, x:x+patch_size]

        if np.mean((p < DARK_THRESHOLD).astype(np.float32)) >= DARK_RATIO:
            if is_flat_region(p):
                patches.append((x, y))
        tries += 1
    return patches

def noise_estimation(patch):
    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    return np.std(laplacian)


# ===============================================================
# Visualization (3 new charts)
# ===============================================================
def plot_results(results_base, results_prop):
    names = [r["name"] for r in results_base]
    base_ratio = np.array([r["ratio"] for r in results_base])
    prop_ratio = np.array([r["ratio"] for r in results_prop])

    x = np.arange(len(names))

    plt.figure(figsize=(18, 10))

    # ---------------------------------------------------------
    # Chart 1: Bar chart (baseline vs proposed per image)
    # ---------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    width = 0.35
    ax1.bar(x - width/2, base_ratio, width, label="Baseline", alpha=0.8)
    ax1.bar(x + width/2, prop_ratio, width, label="Proposed", alpha=0.8)
    ax1.axhline(1.0, color='red', linestyle='--', label="No amplification")
    ax1.set_title("Noise Amplification per Image")
    ax1.set_ylabel("Amplification Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # ---------------------------------------------------------
    # Chart 2: Scatter baseline vs proposed
    # ---------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    ax2.scatter(base_ratio, prop_ratio, alpha=0.6)
    max_v = max(base_ratio.max(), prop_ratio.max())
    ax2.plot([0, max_v], [0, max_v], 'r--', label="y = x")
    ax2.set_xlabel("Baseline Amplification")
    ax2.set_ylabel("Proposed Amplification")
    ax2.set_title("Scatter: Baseline vs Proposed")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ---------------------------------------------------------
    # Chart 3: Boxplot (overall distribution)
    # ---------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    ax3.boxplot([base_ratio, prop_ratio], labels=["Baseline", "Proposed"])
    ax3.set_title("Amplification Distribution")
    ax3.set_ylabel("Amplification Ratio")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("noise_compare_v2.png", dpi=150)
    print("\n✓ Saved: noise_compare_v2.png")
    plt.show()


# ===============================================================
# Main
# ===============================================================
def main():
    random.seed(0)
    files = sorted(os.listdir(LOW_DIR))

    results_base = []
    results_prop = []

    base_before_all = []
    base_after_all = []
    prop_before_all = []
    prop_after_all = []

    print("\n=== Noise Analysis (Baseline vs Proposed) ===")

    for fname in files:
        low_path = os.path.join(LOW_DIR, fname)
        base_path = os.path.join(ENH_DIR_BASE, fname)
        prop_path = os.path.join(ENH_DIR_PROP, fname)

        if not os.path.exists(base_path) or not os.path.exists(prop_path):
            print(f"[Skip] Missing result for {fname}")
            continue

        low = load_img(low_path)
        enh_base = load_img(base_path)
        enh_prop = load_img(prop_path)

        l_low = luminance(low)
        l_base = luminance(enh_base)
        l_prop = luminance(enh_prop)

        patches = sample_dark_patches(l_low, PATCH_SIZE, NUM_PATCHES)
        if not patches:
            print(f"[Skip] No dark-flate patch in {fname}")
            continue

        base_b = []
        base_a = []
        prop_b = []
        prop_a = []

        for (x, y) in patches:
            p_low = l_low[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            p_b = l_base[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            p_p = l_prop[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

            n_low = noise_estimation(p_low)
            n_b = noise_estimation(p_b)
            n_p = noise_estimation(p_p)

            base_b.append(n_low)
            base_a.append(n_b)
            prop_b.append(n_low)
            prop_a.append(n_p)

            base_before_all.append(n_low)
            base_after_all.append(n_b)
            prop_before_all.append(n_low)
            prop_after_all.append(n_p)

        results_base.append({
            "name": fname,
            "ratio": np.mean(base_a) / np.mean(base_b)
        })
        results_prop.append({
            "name": fname,
            "ratio": np.mean(prop_a) / np.mean(prop_b)
        })

    # ============================================================
    # Print Summary (Clean and clear)
    # ============================================================
    print("\n=== Summary ===")
    base_ratio = np.mean(base_after_all) / np.mean(base_before_all)
    prop_ratio = np.mean(prop_after_all) / np.mean(prop_before_all)

    print(f"\nBaseline Method:")
    print(f"  Low Noise Mean:      {np.mean(base_before_all):.4f}")
    print(f"  Enhanced Noise Mean: {np.mean(base_after_all):.4f}")
    print(f"  Amplification:       {base_ratio:.3f}x")

    print(f"\nProposed Method:")
    print(f"  Low Noise Mean:      {np.mean(prop_before_all):.4f}")
    print(f"  Enhanced Noise Mean: {np.mean(prop_after_all):.4f}")
    print(f"  Amplification:       {prop_ratio:.3f}x")

    print(f"\nImprovement:")
    print(f"  Proposed reduces amplification by {(1 - prop_ratio/base_ratio)*100:.2f}%")

    # ============================================================
    # Plot comparison
    # ============================================================
    plot_results(results_base, results_prop)


if __name__ == "__main__":
    main()
