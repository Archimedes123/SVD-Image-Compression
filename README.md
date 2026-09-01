# SVD Image Compression

A Python implementation of image compression using **Singular Value Decomposition (SVD)** — a linear algebra technique that approximates an image using far less data than the original, by keeping only its most significant structural information and discarding the rest.

## How it works

Any grayscale image can be represented as a matrix of pixel values. SVD decomposes that matrix into three smaller matrices:

A = U · Σ · Vᵀ


- **U** and **Vᵀ** capture the structural patterns in the image
- **Σ** (Sigma) is a diagonal matrix of *singular values*, ranked from most to least significant

The key insight: the first few singular values usually carry most of the visual information in an image (edges, shapes, contrast), while later ones capture fine detail and noise. By reconstructing the image using only the top *k* singular values instead of all of them, you get a compressed approximation that keeps the image recognizable while representing it with far less data.

This project:
1. Loads an image and converts it to grayscale using **Pillow (PIL)**
2. Runs `numpy.linalg.svd` on the resulting pixel matrix
3. Reconstructs an approximate version using only the top `num_singular_values` values
4. Displays the original and compressed images side by side using **Matplotlib**
5. Saves the compressed result as a new image file

## Features

- Uses **NumPy** for the SVD decomposition and matrix reconstruction
- Adjustable compression level via `num_singular_values` — lower values mean more compression and more visible quality loss
- Side-by-side visual comparison of original vs. compressed
- Saves the compressed output as a standalone image file

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/Archimedes123/SVD-Image-Compression.git
cd SVD-Image-Compression

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the script
python3 src/svd_image_compression.py
```

> **Note:** the current version has the image path and compression level (`num_singular_values`) hardcoded near the bottom of the script rather than passed as command-line arguments. Open `src/svd_image_compression.py` and edit those two variables to use a different image or compression level.

## Example

Using `Images/asus.jpg` with `num_singular_values = 100`:

![Compression comparison](Images/compression_comparison.png)

Even at 100 retained singular values, the image stays clearly recognizable, though fine detail (particularly in the darker regions) shows visible artifacting compared to the original — a direct illustration of the compression/quality trade-off SVD makes possible.

## What I learned

This project was a hands-on way to see an abstract linear algebra concept — usually taught purely on paper — actually applied to something visual and tangible: turning a matrix factorization into a real, tunable trade-off between file size and image quality. It also became a good lesson in writing code that runs on more than just my own machine — the original version had hardcoded file paths tied to my personal folder structure, which I've since fixed to use relative paths so it runs cleanly on any machine after cloning.

## Known limitations / next steps

- **Grayscale only.** Colour images would need SVD applied independently to each RGB channel.
- **Hardcoded compression level and image path.** A natural next step is accepting these as command-line arguments (using `argparse`) instead of editing the script directly each time.
- **No compression ratio or quality metric reported.** Right now the result is judged by eye. Calculating the actual storage saved (comparing the size of `U`, `Σ`, `Vᵀ` truncated to *k* against the original) or a quality metric like PSNR would make the trade-off measurable, not just visual.

## Dependencies

See `requirements.txt`. Built with NumPy (SVD/matrix operations), Pillow (image loading), and Matplotlib (visualisation).