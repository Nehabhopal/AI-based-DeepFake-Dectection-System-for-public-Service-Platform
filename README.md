# DeepSafe – AI-Based DeepFake Detection System for Public Service Platforms

An AI-powered system that detects deepfake (manipulated or synthetically generated) media, built to help public service platforms verify the authenticity of images and videos submitted by users and protect citizens from misinformation, impersonation, and identity fraud.

## Overview

Deepfakes are becoming a serious threat to public service platforms such as e-governance portals, identity verification systems, and citizen helplines. **DeepSafe** uses a deep learning model to classify media as **Real** or **Fake**, giving platform operators a fast and reliable way to screen content before it is trusted or acted upon.

## Features

- Deepfake detection using a trained deep learning model
- Classification of media as **Real** or **Fake** with a confidence score
- Dataset splitting utility for training, validation and testing
- Model verification script to check that the trained model loads and works
- Designed to plug into public service platforms (e.g. identity checks, document and media verification)

## Tech Stack

| Component        | Technology                                     |
|------------------|------------------------------------------------|
| Language         | Python                                         |
| Deep Learning    | TensorFlow *(update if using PyTorch)* |
| Data Handling    | NumPy, OpenCV                                  |
| Application      | DeepSafe app *(Flask – update as applicable)* |

## Project Structure

```
AI-based-DeepFake-Dectection-System-for-public-Service-Platform/
├── DeepSafe/            # Main application / project code
├── check_model.py       # Script to verify and test the trained model
├── split_dataset.py     # Script to split the dataset into train / val / test sets
└── .gitignore
```

## Prerequisites

- Python 3.8 or higher
- pip
- A deepfake dataset (e.g. FaceForensics++, DFDC, or Celeb-DF)
- (Optional) A GPU for faster training

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nehabhopal/AI-based-DeepFake-Dectection-System-for-public-Service-Platform.git
   cd AI-based-DeepFake-Dectection-System-for-public-Service-Platform
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > If there is no `requirements.txt` yet, install the libraries you use (for example `tensorflow`, `numpy`, `opencv-python`).

## Usage

### 1. Prepare the dataset
Place your dataset in the expected folder (with `real` and `fake` classes), then split it:
```bash
python split_dataset.py
```

### 2. Train the model
Train from inside the `DeepSafe` project (see the training script there).

### 3. Verify the model
```bash
python check_model.py
```

### 4. Run the application
```bash
cd DeepSafe
python app.py        # Update with your actual entry file
```

## How It Works

1. The user uploads an image.
2. The media is preprocessed (face extraction, resizing, normalization).
3. The trained model analyses it for manipulation artifacts.
4. The system returns a **Real / Fake** prediction with a confidence score.

## Use Cases

- Identity verification on e-governance portals
- Screening user-submitted media on citizen service platforms
- Preventing misinformation and impersonation
- Supporting digital forensics and public safety teams

## Future Improvements

- Real-time video stream detection
- Support for audio deepfakes
- Explainable AI (heatmaps showing manipulated regions)
- REST API for easy integration with other platforms
- Model performance dashboard (accuracy, precision, recall, F1-score)

## Contributing

Contributions are welcome. Fork the repository, create a feature branch, and open a pull request.

## Author

**Neha Bhopal** – [GitHub](https://github.com/Nehabhopal)

## License

Add a license of your choice (e.g. MIT) to the repository.
