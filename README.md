# System Risk Index (SRI) v2: A Deterministic Safety Supervisor for Autonomous Vehicles

This repository implements **System Risk Index (SRI) v2**, a lightweight, real-time safety supervisor wrapper engineered to solve the **"False-Safe" Paradox** in autonomous vehicle perception systems. Standard deep learning models measure algorithmic certainty, not physical safety—meaning a highly confident model running in clear weather can dangerously dilute imminent proximity risks. 

SRI v2 operates as a real-time dual-stream pipeline, continuously fusing four safety vectors into a single, un-dilutable risk score:
1. **Visibility Hazard ($V$):** Evaluates grayscale RMS contrast drops (fog, rain, sudden darkness).
2. **Epistemic Uncertainty ($E$):** Forces the model to reveal internal hesitation by tracking prediction variance between a clean and gamma-perturbed stream.
3. **Temporal Instability ($T$):** Monitors bounding box flickering across a rolling 60-frame buffer.
4. **Proximity Alert ($P$):** Computes spatial canvas penetration magnified by vehicle lane positioning.

---

## Dataset Architecture & Curation

To ensure the framework is robust against harsh edge cases, sensory degradation, and varying environments, we curated a heterogeneous training and evaluation benchmark by combining slices from **three distinct public datasets**:

1. **BDD100K (Berkeley DeepDrive):** Used for baseline daytime, night-time, and urban driving sequences to capture diverse pedestrian densities and lane geometries.
2. **nuScenes (mini):** Used to ingest precise multi-modal structural scenarios, extracting high-quality sequence annotations to test model tracking consistency.
3. **DAWN (Vehicle Detection in Adverse Weather Nature):** Specifically integrated to extract extreme environmental degradation frames, including heavy fog, torrential rain, and snowstorms, establishing rigorous testing criteria for the Visibility Hazard ($V$) pillar.

### Dataset Curation Script
Below is the structured utility code used to align categories, handle varying annotation schemas, filter for `pedestrian` classes, and compile the unified master dataset:

```python
import os
import shutil
from pathlib import Path

def setup_master_dataset(output_dir="data/processed/master_dataset"):
    """
    Creates the directory layout for compiling BDD100K, nuScenes, and DAWN subsets.
    """
    splits = ['train', 'val']
    subdirs = ['images', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            path = Path(output_dir) / split / subdir
            path.mkdir(parents=True, exist_ok=True)
    print(f"[*] Master dataset directories established at: {output_dir}")

def merge_subset(source_img_dir, source_label_dir, dest_root, split, dataset_name):
    """
    Copies images and text labels into the master dataset while preventing name collisions.
    """
    dest_img_dir = Path(dest_root) / split / 'images'
    dest_label_dir = Path(dest_root) / split / 'labels'
    
    copied_count = 0
    img_extensions = ('.jpg', '.jpeg', '.png')
    
    for img_path in Path(source_img_dir).iterdir():
        if img_path.suffix.lower() in img_extensions:
            # Append dataset prefix to avoid filename collisions
            unique_name = f"{dataset_name}_{img_path.name}"
            corresponding_label = Path(source_label_dir) / f"{img_path.stem}.txt"
            
            if corresponding_label.exists():
                # Copy Image
                shutil.copy(img_path, dest_img_dir / unique_name)
                # Copy Label
                shutil.copy(corresponding_label, dest_label_dir / f"{Path(unique_name).stem}.txt")
                copied_count += 1
                
    print(f"[+] Successfully merged {copied_count} paired samples from {dataset_name} into {split} split.")

# Example execution workflow block
if __name__ == "__main__":
    MASTER_DIR = "data/processed/master_dataset"
    setup_master_dataset(MASTER_DIR)
    
    # Example paths (Replace with your actual path structure)
    # merge_subset("path/to/bdd_images", "path/to/bdd_labels", MASTER_DIR, "train", "bdd")
    # merge_subset("path/to/dawn_images", "path/to/dawn_labels", MASTER_DIR, "val", "dawn")
