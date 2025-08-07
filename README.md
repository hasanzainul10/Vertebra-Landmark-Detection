# Cobb Angle Analysis System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7+-red.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)

An advanced AI-powered system for automatic Cobb angle measurement in spinal X-ray images using deep learning.

## 🚀 Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repository-url>
cd Vertebra-Landmark-Detection

# Run setup script
python setup.py
```

### 2. Download Model

Place the pre-trained model file (`model_last.pth`) in the `pretrained_model/` directory.

### 3. Analyze an Image

```bash
python cobb_angle_inference.py --image_path path/to/your/xray.jpg
```

## 📋 Features

- **🤖 AI-Powered Detection**: Automatic vertebral landmark detection using deep learning
- **📐 Precise Measurements**: Medical-standard Cobb angle calculations
- **🎨 Professional Visualization**: Clear visual overlays with measurement lines
- **📊 Comprehensive Analysis**: Primary, secondary, and tertiary angle measurements
- **💾 Detailed Reporting**: Saves results as images, data files, and text reports
- **⚡ Fast Processing**: GPU acceleration support for quick analysis

## 📖 Documentation

- **[User Manual](USER_MANUAL.md)** - Comprehensive guide for all users
- **[Quick Reference](QUICK_REFERENCE.md)** - Fast commands and troubleshooting
- **[Requirements](requirements.txt)** - Python dependencies

## 🔬 How It Works

### Cobb Angle Measurement

The system follows the standard medical protocol for Cobb angle measurement:

1. **Landmark Detection**: AI identifies vertebral landmarks in X-ray images
2. **End Vertebrae Selection**: Finds the most tilted vertebrae at curve ends
3. **Parallel Lines**: Draws lines parallel to vertebral endplates
4. **Perpendicular Lines**: Extends perpendicular lines until intersection
5. **Angle Calculation**: Measures the angle at the intersection point

### Primary, Secondary, and Tertiary Angles

- **Primary**: Largest Cobb angle (determines treatment approach)
- **Secondary**: Second largest angle (compensatory curve)
- **Tertiary**: Third largest angle (usually minor variation)

## 📊 Severity Classification

| Cobb Angle | Severity           | Clinical Action           |
| ---------- | ------------------ | ------------------------- |
| < 10°      | Normal             | No treatment needed       |
| 10° - 25°  | Mild Scoliosis     | Monitor, possible bracing |
| 25° - 40°  | Moderate Scoliosis | Bracing, physical therapy |
| > 40°      | Severe Scoliosis   | Surgical consultation     |

## 🛠️ Usage Examples

### Basic Analysis

```bash
python cobb_angle_inference.py --image_path xray.jpg
```

### Advanced Options

```bash
python cobb_angle_inference.py \
    --image_path xray.jpg \
    --model_path pretrained_model/model_last.pth \
    --output_dir results \
    --device cuda
```

### CPU Processing

```bash
python cobb_angle_inference.py --image_path xray.jpg --device cpu
```

## 📁 Output Files

The system generates several output files:

- `*_cobb_analysis.jpg` - Visualized results with measurements
- `*_landmarks.npy` - Detected landmark coordinates
- `*_cobb_angles.npy` - Calculated Cobb angles
- `*_report.txt` - Detailed analysis report

## 🎨 Visual Output

The analysis displays:

- **Blue rectangles**: All detected vertebrae
- **Green lines**: Primary Cobb angle measurement
- **Magenta lines**: Secondary Cobb angle measurement
- **Cyan lines**: Tertiary Cobb angle measurement
- **Colored arcs**: Angle measurements at intersection points
- **Results panel**: Summary with angles and diagnosis

## ⚙️ System Requirements

### Hardware

- **CPU**: Intel i5 or AMD equivalent (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **Storage**: 2GB free space

### Software

- **Python**: 3.7 or higher
- **CUDA**: 10.0 or higher (if using GPU)
- **OS**: Windows 10/11, macOS 10.15+, or Linux

## 🔧 Installation

### Automatic Setup

```bash
python setup.py
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p pretrained_model cobb_results output
```

## 🚨 Troubleshooting

### Common Issues

**"No landmarks detected!"**

- Check image quality (high contrast, clear spine)
- Ensure spine is centered and visible
- Try different image format

**CUDA out of memory**

```bash
python cobb_angle_inference.py --image_path xray.jpg --device cpu
```

**Model not found**

- Verify `pretrained_model/model_last.pth` exists
- Check file permissions
- Download model file again

For more troubleshooting help, see [Quick Reference](QUICK_REFERENCE.md).

## 📚 Medical Background

### What is Scoliosis?

Scoliosis is a medical condition where the spine curves sideways, forming an "S" or "C" shape instead of being straight. Cobb angles are the standard measurement used by medical professionals to quantify the severity of spinal curvature.

### Types of Scoliosis

- **Idiopathic Scoliosis**: Most common (80% of cases), unknown cause
- **Congenital Scoliosis**: Present at birth, caused by vertebral malformation
- **Neuromuscular Scoliosis**: Caused by neurological conditions

### Clinical Applications

- **Diagnosis**: Quantify curve severity and type
- **Treatment Planning**: Guide bracing and surgical decisions
- **Monitoring**: Track curve progression over time

## ⚠️ Important Notes

- **Research Use**: This tool is for research and educational purposes
- **Medical Disclaimer**: Not intended for clinical diagnosis
- **Professional Verification**: Results should be verified by medical professionals
- **Clinical Decisions**: Always consult qualified healthcare providers

## 🤝 Contributing

We welcome contributions! Please:

- Report bugs with detailed information
- Suggest new features
- Improve documentation
- Submit pull requests

## 📄 License

[Add your license information here]

## 📞 Support

- **Documentation**: Check [User Manual](USER_MANUAL.md) first
- **Issues**: Report technical problems with detailed information
- **Medical Questions**: Consult qualified healthcare providers

## 🙏 Acknowledgments

- Dr. John Robert Cobb for developing the Cobb angle measurement method
- Medical imaging community for validation and feedback
- Open source contributors for supporting libraries

---

**Remember**: This software is provided for research and educational purposes. Clinical decisions should be made by qualified healthcare providers.
