# 🚗 Automatic Number Plate Recognition (ANPR)

A modern, lightweight Computer Vision web application built with **Streamlit**, **OpenCV**, and **EasyOCR**. It detects and extracts vehicle registration numbers from images, with dedicated support for **Indian vehicle registration plate formats** (Standard State/UT plates and Bharat (BH) Series).

---

## ✨ Features

- **Intuitive Web UI**: Modern Streamlit interface with clear metrics, custom HSRP license plate badge, and side-by-side visualization.
- **Indian Plate Formats Prioritized**:
  - **Standard State Formats**: `MH 12 DE 1433`, `DL 3C AB 1234`, `GJ 08 DG 9292`, `KA 05 MN 9876`, etc. (Displays state name like *Maharashtra*, *Gujarat*, *Delhi*, *Karnataka*).
  - **Bharat (BH) Series**: `22 BH 6543 AA`, etc. (Displays registration year, series, and pan-India status).
  - **Indian Military / Armed Forces (Defence)**: `↑ 20B 132976 N` (Displays Ministry of Defence authority, procurement year, vehicle class, serial number, and check code).
- **Intelligent Preprocessing & Cleansing**:
  - Automatically strips HSRP `IND` prefixes.
  - Heuristic character confusion correction (e.g., handles OCR mix-ups between `O`/`0`, `I`/`1`, `Z`/`2`, `B`/`8`).
- **OpenCV Computer Vision Pipeline**:
  - Grayscale conversion.
  - Edge-preserving noise reduction via Bilateral Filtering.
  - Contrast enhancement with CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - Canny Edge Detection and rectangular contour candidate extraction.
  - Interactive pipeline visualizer to view intermediate OpenCV stages in real-time.
- **EasyOCR Integration**: Deep-learning text detection (CRAFT) and recognition (CRNN) running locally on CPU without requiring external APIs or GPU.
- **Visual Bounding Boxes**: Automatically draws green detection bounding boxes with confidence tags around the detected license plate.
- **Preloaded Sample Images**: Test immediately with built-in vehicle images without needing to upload your own.

---

## 🛠️ Tech Stack

- **Language**: Python 3.10+
- **Frontend / App Server**: [Streamlit](https://streamlit.io/)
- **Image Processing**: [OpenCV](https://opencv.org/) (`opencv-python-headless`)
- **OCR Engine**: [EasyOCR](https://github.com/JaidedAI/EasyOCR) (PyTorch CPU backend)
- **Scientific Computing**: NumPy
- **Image Handling**: Pillow (PIL)

---

## 📁 Project Structure

```text
CVOC PROJECT/
│
├── app.py                     # Main Streamlit web application
├── create_sample_images.py    # Generates realistic test vehicle images
├── test_anpr.py               # Automated unit tests for pipeline & regex
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
│
├── sample_images/             # Preloaded vehicle test images
│   ├── car_sample_mh12.jpg    # Maharashtra plate: MH 12 DE 1433
│   ├── car_sample_dl3c.jpg    # Delhi plate: DL 3C AB 1234
│   ├── car_sample_22bh.jpg    # BH Series plate: 22 BH 6543 AA
│   └── car_sample_ka05.jpg    # Karnataka plate: KA 05 MN 9876
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10 or higher installed.

### 2. Clone / Open Project Directory
```bash
cd "CVOC PROJECT"
```

### 3. Create and Activate Virtual Environment
```bash
# On Linux / macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Note: If installing PyTorch CPU separately on Linux/Windows for a lightweight install:)*
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 5. Generate Test Images (Optional)
```bash
python create_sample_images.py
```

### 6. Run the Application
```bash
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 🧪 Testing the Pipeline

To run automated checks on the regex validation logic and EasyOCR detection:
```bash
python test_anpr.py
```

---

## 🔍 How It Works

1. **Upload & Ingestion**: The user uploads an image or selects a preloaded sample.
2. **OpenCV Preprocessing**:
   - `cv2.cvtColor` converts the image to grayscale.
   - `cv2.bilateralFilter` removes sensor noise while keeping plate edges sharp.
   - `cv2.createCLAHE` normalizes illumination and boosts contrast between plate digits and background.
   - `cv2.Canny` detects high-gradient boundaries.
3. **Plate Localization**: Rectangular contours with Indian plate aspect ratios (~1.8 to 6.0) are extracted as candidate crops.
4. **Text Extraction (EasyOCR)**: EasyOCR processes candidate crops and full image to detect character sequences.
5. **Pattern Validation & Normalization**: Regular expressions filter candidates against Indian motor vehicle standards and normalize characters.
6. **Result Presentation**: The extracted number is highlighted in an authentic HSRP badge with confidence metrics, state info, and bounding box visualization.

---

## 📄 License
This project is open-source and available under the MIT License.
