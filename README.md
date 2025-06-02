# Selective Search PyTorch – Enhanced Fork

This is a **fork** of the original repository [`vadimkantorov/selective_search_pytorch`](https://github.com/vadimkantorov/selective_search_pytorch), which provides a reimplementation of the Selective Search algorithm in PyTorch. Selective Search is widely used for object proposal generation, producing bounding boxes and region segmentation masks.

---

## 🔧 What's Different in This Fork?

This fork extends the original project with several **Pythonic utilities** and usability enhancements:

✅ `utils.py` module for easy usage within Python scripts (not just via CLI)  
✅ Precompiled `.so` (shared object) file included – **no need to build anything**  
✅ Functions to evaluate and select the **best segmentation masks** based on geometric smoothness  
✅ Easy extraction of **bounding boxes per region class**  
✅ Built-in **visualization tools** to explore segmentations and proposals

---

## 🚀 Highlights

- Run Selective Search directly from Python — just `import selective_search`
- Segmentations returned as tensor masks at multiple hierarchy levels
- Select the smoothest segmentation mask based on normalized perimeter
- Extract bounding boxes per segment label
- Works out-of-the-box — no compilation necessary!

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/selective_search_pytorch.git
cd selective_search_pytorch