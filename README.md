
# 🖼️ Real-ESRGAN Image Enhancer

This repository provides a Python tool for enhancing images using the Real-ESRGAN model. It features adaptive configuration based on system resources to optimize performance and handle large images efficiently. The tool leverages GPU acceleration when available and includes a user-friendly interface for selecting files.

## 📑 Table of Contents

1. [Repository Structure](#-repository-structure)
2. [Project Description](#-project-description)
3. [How to Use](#-how-to-use)
4. [Requirements](#️-requirements)
5. [Download as .zip](#-download-as-zip)
6. [Authors](#-authors)
7. [Course Information](#-course-information)
8. [Contribution](#-contribution)
9. [License](#-license)

## 📁 Repository Structure

```
.
|-- config
|   |-- default_config.json
|-- models
|   |-- RealESRGAN_x4plus.pth
|-- main.py
|-- .gitignore
|-- README.md
```

- **`config/default_config.json`**: Configuration file containing settings for image processing, such as chunk height, tile size, and model parameters.
- **`models/RealESRGAN_x4plus.pth`**: Pre-trained Real-ESRGAN model file used for super-resolution.
- **`main.py`**: Main script containing all logic for system initialization, configuration management, and image processing.
- **`.gitignore`**: Specifies files and directories to exclude from version control (e.g., `.history`, `.vs`).

## 📂 Project Description

**Description:** This tool allows users to upscale images using the Real-ESRGAN model. It processes images in chunks to manage large files effectively and optimizes settings based on system memory and GPU availability. The script includes a graphical interface for selecting input and output files and displays system information and processing progress with colorful ASCII art.

- **Main File:**
  - `main.py`: Handles configuration loading, system checks, image validation, and chunk-based processing using Real-ESRGAN.
- **Configuration:**
  - Settings are loaded from `config/default_config.json`, enabling customization of parameters like chunk height, tile size, and overlap.
- **Features:**
  - Adaptive configuration adjusts processing based on available RAM and GPU memory.
  - Support for multiple image formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).
  - GPU acceleration with CUDA for faster processing.
  - Memory management to prevent out-of-memory errors.
  - Optional video processing capability (demonstrated in a code snippet, requires integration).

## 🚀 How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/real-esrgan-enhancer.git
   ```
   Replace `your-username` with your GitHub username.

2. **Install dependencies:**
   ```bash
   pip install torch opencv-python pillow basicsr
   ```
   - For GPU support, install the appropriate version of PyTorch with CUDA from [PyTorch's website](https://pytorch.org/get-started/locally/).
   - Ensure the model file `RealESRGAN_x4plus.pth` is in the `models` directory. Download it from [the Real-ESRGAN repository](https://github.com/xinntao/Real-ESRGAN) if needed.

3. **Run the script:**
   ```bash
   python main.py
   ```
   - A file dialog will open to select the input image.
   - Choose the output file path in the next dialog.
   - The script will process the image and save the enhanced version to the specified location.

4. **Configuration (Optional):**
   - Edit `config/default_config.json` to adjust processing settings, such as `chunk_height` or `tile_size`, if desired.

5. **For Video Processing (Optional):**
   - The repository includes a snippet for video enhancement. To use it, modify `main.py` to incorporate the video processing code (see provided example), set input and output video paths, and run the script.

## 🛠️ Requirements

- **Python 3.8+**
- **PyTorch** (with optional CUDA support for GPU acceleration)
- **OpenCV** (`opencv-python`) - For image and potential video processing
- **Pillow** - For image handling
- **basicsr** - Provides foundational architecture for Real-ESRGAN
- **tkinter** - For GUI file dialogs (typically included with Python)
- **colorama** - For colored terminal output (optional, included in script)
- **psutil** - For detailed system memory info (optional, recommended)

Install dependencies via:
```bash
pip install torch opencv-python pillow basicsr colorama psutil
```

## 📦 Download as .zip

If you prefer, you can download the entire repository as a .zip file:

1. Visit the repository page on GitHub.
2. Click the green **Code** button.
3. Select **Download ZIP** and extract the contents to your local directory.

## 👥 Authors

- Andrés Velázquez Vela

## 📘 Course Information

This project is not explicitly tied to a specific course but demonstrates advanced image processing techniques that could be relevant to courses in computer vision, machine learning, or software engineering. It was developed as a standalone tool for practical application.

## 🤝 Contribution

Contributions are welcome! Potential enhancements include:
- Adding command-line arguments for batch processing or file specification.
- Fully integrating video processing functionality into the main script.
- Improving error handling and user feedback.
- Optimizing chunk-based processing for different hardware configurations.

Feel free to submit a pull request or open an issue to discuss improvements.

## 📄 License

This project is free to use, modify, and distribute without restrictions.

---
