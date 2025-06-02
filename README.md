# Selective Search PyTorch – Enhanced Fork

This is a **fork** of the original repository [`vadimkantorov/selective_search_pytorch`](https://github.com/vadimkantorov/selective_search_pytorch), reimplemented in PyTorch for object proposal generation, with extended Python utilities and easier integration.

---

## Install via pip (directly from GitHub)

You can install this package **directly from GitHub** using pip — no need to clone or compile:

```bash
pip install git+https://github.com/yourusername/selective_search_pytorch.git
```

## What’s new in this fork?
	•	Precompiled .so shared library included — no compilation needed
	•	Python API with utils.py to run Selective Search from scripts or notebooks
	•	Functions to select the best segmentation mask based on smoothness metrics
	•	Easy extraction of bounding boxes per segmentation region
	•	Visualization helpers for masks and bounding boxes

## Requirements
	•	Python 3.7+
	•	PyTorch
	•	OpenCV-Python
	•	NumPy
	•	Matplotlib

You can install these via pip:
```bash
pip install torch opencv-python numpy matplotlib
```
## Quick Usage Example

```bash 
from selective_search import *
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# Load the Selective Search algorithm
algo = load_selective_search()

# Preprocess input image
img_path = "examples/astronaut.jpg"
img_bgr1hw3_255 = prepocess_img(img_path)

# Run Selective Search
boxes_xywh, regions, reg_lab = algo(
    img_bgrbhw3_255=img_bgr1hw3_255,
    generator=torch.Generator().manual_seed(42),
    print=print
)

# Convert region labels to numpy array
reg_lab_np = reg_lab.squeeze(0).squeeze(1).cpu().numpy()  # Shape: (levels, H, W)

# Visualize segmentation levels with smoothness metric
for i in range(reg_lab_np.shape[0]):
    perimetro_norm = calculate_normalized_perimeter(reg_lab_np[i])
	plt.figure(figsize=(6, 6))
    plt.imshow(reg_lab_np[i], cmap='nipy_spectral')
    plt.title(f'Segmentation Level {i}\nNormalized Perimeter = {perimetro_norm:.4f}')
    plt.axis('off')
    plt.show()

# Select the smoothest mask
smooth_mask = select_by_softness(reg_lab_np, level=1)
smooth_maske_uint8 = smooth_mask.astype(np.uint8)

# Extract bounding boxes from mask
bboxes = extract_bboxes_per_class(smooth_maske_uint8)

# Visualize bounding boxes on original image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
for x, y, w, h in bboxes:
    plt.gca().add_patch(patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none'))
plt.title(f'Bounding Boxes: {len(bboxes)}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mascara_suave, cmap='nipy_spectral')
plt.title('Selected Smooth Mask')
plt.axis('off')
plt.show()
```
## Repository Structure
```bash 
selective_search_pytorch/
├── selective_search/                  # Python package with utils and API
│   ├── __init__.py
│   ├── utils.py                      # Helper functions for masks and boxes
│   └── selectivesearch_py.py         # Core PyTorch wrapper
├── opencv_custom/
│   ├── selectivesearchsegmentation_opencv_custom_.so  # Precompiled shared lib
├── examples/
│   └── astronaut.jpg
├── setup.py                          # Installation script for pip
├── MANIFEST.in                      # Package data inclusion
├── README.md
```
## ⚠️ Notes
	•	The .so shared library is included and precompiled for ease of use.
	•	No compilation or setup beyond pip install required.
	•	Tested on macOS and Linux.

## 🙏 Acknowledgments
This fork is based on the original work by Vadim Kantorov.

⸻

If you want me to help generate or review your setup.py or MANIFEST.in to make sure pip installs work perfectly, just ask!