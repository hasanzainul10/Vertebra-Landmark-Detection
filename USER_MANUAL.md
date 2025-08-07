# Cobb Angle Analysis System - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Understanding Cobb Angles](#understanding-cobb-angles)
6. [Using the Analysis Tools](#using-the-analysis-tools)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)
9. [Medical Background](#medical-background)
10. [FAQ](#faq)

---

## Introduction

The Cobb Angle Analysis System is an advanced AI-powered tool for analyzing spinal X-ray images and measuring Cobb angles, which are crucial for diagnosing and monitoring scoliosis. This system uses deep learning to automatically detect vertebral landmarks and calculate precise Cobb angle measurements.

### What is Scoliosis?

Scoliosis is a medical condition where the spine curves sideways, forming an "S" or "C" shape instead of being straight. Cobb angles are the standard measurement used by medical professionals to quantify the severity of spinal curvature.

### Key Features

- **Automatic Landmark Detection**: AI identifies vertebral landmarks in X-ray images
- **Precise Cobb Angle Calculation**: Measures angles using medical standards
- **Visual Analysis**: Displays measurements with clear visual overlays
- **Comprehensive Reporting**: Generates detailed analysis reports
- **Professional Visualization**: Medical-grade measurement displays

---

## System Requirements

### Hardware Requirements

- **CPU**: Intel i5 or AMD equivalent (minimum)
- **RAM**: 8GB (minimum), 16GB (recommended)
- **GPU**: NVIDIA GPU with CUDA support (recommended for faster processing)
- **Storage**: 2GB free space

### Software Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.7 or higher
- **CUDA**: 10.0 or higher (if using GPU)

### Dependencies

```
torch>=1.7.0
torchvision>=0.8.0
numpy>=1.19.0
opencv-python>=4.5.0
scipy>=1.5.0
matplotlib>=3.3.0
Pillow>=8.0.0
tqdm>=4.50.0
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Vertebra-Landmark-Detection
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Pre-trained Model

Place the pre-trained model file (`model_last.pth`) in the `pretrained_model/` directory.

### Step 4: Verify Installation

```bash
python cobb_angle_inference.py --help
```

---

## Quick Start Guide

### Basic Usage

```bash
python cobb_angle_inference.py --image_path path/to/your/xray.jpg
```

### Advanced Usage

```bash
python cobb_angle_inference.py \
    --image_path path/to/your/xray.jpg \
    --model_path pretrained_model/model_last.pth \
    --output_dir results \
    --device cuda
```

### Command Line Arguments

- `--image_path`: Path to the X-ray image (required)
- `--model_path`: Path to pre-trained model (default: pretrained_model/model_last.pth)
- `--output_dir`: Output directory for results (default: cobb_results)
- `--device`: Processing device - cuda or cpu (default: cuda)
- `--no_save`: Don't save output files (display only)

---

## Understanding Cobb Angles

### What are Cobb Angles?

Cobb angles measure the degree of spinal curvature in scoliosis. They are calculated by:

1. Identifying the most tilted vertebrae at the top and bottom of a curve
2. Drawing lines parallel to the endplates of these vertebrae
3. Drawing perpendicular lines from these parallel lines
4. Measuring the angle at the intersection

### Primary, Secondary, and Tertiary Angles

#### Primary Angle

- **Definition**: The largest Cobb angle in the spine
- **Clinical Significance**: Determines treatment approach and severity classification
- **Treatment Priority**: This angle guides medical decisions

#### Secondary Angle

- **Definition**: The second largest Cobb angle
- **Clinical Significance**: Compensatory curve that helps maintain balance
- **Purpose**: May require separate treatment consideration

#### Tertiary Angle

- **Definition**: The third largest Cobb angle
- **Clinical Significance**: Usually minor compensatory curve
- **Purpose**: May represent normal variation or minor curvature

### Severity Classification

| Cobb Angle | Severity           | Clinical Action           |
| ---------- | ------------------ | ------------------------- |
| < 10°      | Normal             | No treatment needed       |
| 10° - 25°  | Mild Scoliosis     | Monitor, possible bracing |
| 25° - 40°  | Moderate Scoliosis | Bracing, physical therapy |
| > 40°      | Severe Scoliosis   | Surgical consultation     |

---

## Using the Analysis Tools

### 1. Single Image Analysis

```bash
python cobb_angle_inference.py --image_path xray.jpg
```

### 2. Batch Processing

```bash
# Process multiple images
for img in *.jpg; do
    python cobb_angle_inference.py --image_path "$img" --output_dir "results_$(basename $img .jpg)"
done
```

### 3. Basic Landmark Detection

```bash
python evaluate_single_image.py --image_path xray.jpg
```

### Output Files Generated

- `*_cobb_analysis.jpg`: Visualized results with measurements
- `*_landmarks.npy`: Detected landmark coordinates
- `*_cobb_angles.npy`: Calculated Cobb angles
- `*_report.txt`: Detailed analysis report

---

## Interpreting Results

### Visual Output Components

#### 1. Detected Vertebrae (Blue)

- **Blue rectangles**: All detected vertebrae
- **Blue circles**: Individual landmark points
- **Purpose**: Shows AI detection accuracy

#### 2. Cobb Angle Measurements

- **Green lines**: Primary Cobb angle measurement
- **Magenta lines**: Secondary Cobb angle measurement
- **Cyan lines**: Tertiary Cobb angle measurement
- **Colored arcs**: Angle measurements at intersection points

#### 3. Results Panel

- **Primary Angle**: Largest curve measurement
- **Secondary Angle**: Second largest curve
- **Tertiary Angle**: Third largest curve
- **Maximum Angle**: Highest measured angle
- **Diagnosis**: Severity classification
- **Landmark Count**: Number of detected vertebrae

### Understanding the Measurements

#### Example Results

```
Primary: 34.5°
Secondary: 29.3°
Tertiary: 5.6°
Maximum Angle: 34.5°
Diagnosis: Moderate Scoliosis
```

**Interpretation**:

- The main curve is 34.5° (moderate scoliosis)
- There's a compensatory curve of 29.3°
- A minor curve of 5.6° (likely normal variation)
- Treatment should focus on the 34.5° primary curve

### Medical Significance

#### For Patients

- **< 10°**: Normal spine, no concerns
- **10° - 25°**: Mild curvature, monitor for progression
- **25° - 40°**: Moderate curvature, consider bracing
- **> 40°**: Severe curvature, surgical evaluation needed

#### For Healthcare Providers

- **Primary angle**: Guides treatment decisions
- **Secondary angle**: Assesses spinal balance
- **Progression**: Compare angles over time
- **Surgical planning**: Use primary angle for fusion levels

---

## Troubleshooting

### Common Issues

#### 1. No Landmarks Detected

**Symptoms**: "No landmarks detected!" error
**Causes**:

- Poor image quality
- Incorrect image format
- Spine not clearly visible
  **Solutions**:
- Use high-quality X-ray images
- Ensure spine is centered and visible
- Check image format (JPEG, PNG supported)

#### 2. CUDA Out of Memory

**Symptoms**: CUDA memory error
**Solutions**:

- Use CPU instead: `--device cpu`
- Reduce image resolution
- Close other GPU applications

#### 3. Model Not Found

**Symptoms**: "Model file does not exist" error
**Solutions**:

- Verify model file location
- Check file permissions
- Download model file again

#### 4. Poor Detection Quality

**Symptoms**: Inaccurate landmark detection
**Causes**:

- Low image contrast
- Motion blur
- Unusual spine positioning
  **Solutions**:
- Use high-contrast X-ray images
- Ensure patient is still during imaging
- Use standard AP (anterior-posterior) views

### Performance Optimization

#### For Faster Processing

- Use GPU acceleration (`--device cuda`)
- Close unnecessary applications
- Use SSD storage for large datasets

#### For Better Accuracy

- Use high-resolution images (1024x512 minimum)
- Ensure good image contrast
- Use standard radiographic positioning

---

## Medical Background

### Scoliosis Types

#### 1. Idiopathic Scoliosis

- **Most common type** (80% of cases)
- **Unknown cause**
- **Usually develops during adolescence**
- **Treatment**: Monitoring, bracing, or surgery

#### 2. Congenital Scoliosis

- **Present at birth**
- **Caused by vertebral malformation**
- **May require early intervention**

#### 3. Neuromuscular Scoliosis

- **Caused by neurological conditions**
- **Examples**: Cerebral palsy, muscular dystrophy
- **Often requires surgical treatment**

### Cobb Angle Measurement History

#### Dr. John Robert Cobb (1903-1967)

- **Developed the Cobb angle measurement in 1948**
- **Standard method for scoliosis assessment**
- **Used worldwide in clinical practice**

#### Measurement Process

1. **Identify end vertebrae** (most tilted at curve ends)
2. **Draw parallel lines** to endplate surfaces
3. **Draw perpendicular lines** from parallels
4. **Measure intersection angle**

### Clinical Applications

#### Diagnosis

- **Quantify curve severity**
- **Determine curve type** (S vs C shape)
- **Assess spinal balance**

#### Treatment Planning

- **Bracing decisions** (typically 25°-40°)
- **Surgical planning** (typically >40°)
- **Monitoring progression**

#### Follow-up

- **Track curve progression**
- **Assess treatment effectiveness**
- **Determine growth completion**

---

## FAQ

### General Questions

**Q: How accurate is the AI detection?**
A: The system achieves high accuracy on standard X-ray images. However, results should be verified by medical professionals for clinical use.

**Q: Can I use this for clinical diagnosis?**
A: This tool is for research and educational purposes. Clinical decisions should be made by qualified healthcare providers.

**Q: What image formats are supported?**
A: JPEG, PNG, and other common image formats supported by OpenCV.

**Q: How long does analysis take?**
A: Typically 5-30 seconds depending on image size and hardware.

### Technical Questions

**Q: Why do I need CUDA?**
A: CUDA enables GPU acceleration for faster processing. CPU-only processing is also available but slower.

**Q: Can I process multiple images at once?**
A: Yes, you can create batch processing scripts or use the system in loops.

**Q: How do I improve detection accuracy?**
A: Use high-quality, high-contrast X-ray images with clear spine visibility.

**Q: What if the model doesn't detect landmarks?**
A: Check image quality, ensure spine is visible, and try different images.

### Medical Questions

**Q: What do the different colored lines mean?**
A: Green = Primary angle, Magenta = Secondary angle, Cyan = Tertiary angle.

**Q: How do I interpret the severity classification?**
A: <10° = Normal, 10-25° = Mild, 25-40° = Moderate, >40° = Severe.

**Q: Why are there three angles?**
A: The spine can have multiple curves. Primary is the largest, secondary and tertiary are smaller compensatory curves.

**Q: How often should Cobb angles be measured?**
A: This depends on age, severity, and treatment plan. Consult with healthcare providers.

### Troubleshooting Questions

**Q: The system crashes when I run it.**
A: Check system requirements, ensure all dependencies are installed, and verify model file location.

**Q: Results look inaccurate.**
A: Verify image quality, check for motion blur, and ensure proper radiographic positioning.

**Q: Processing is very slow.**
A: Use GPU acceleration, close other applications, and check available memory.

**Q: I get different results for the same image.**
A: This may indicate image quality issues or detection uncertainty. Try multiple runs and consult medical professionals.

---

## Support and Contact

### Getting Help

- **Documentation**: Check this manual first
- **Issues**: Report technical problems with detailed information
- **Medical Questions**: Consult qualified healthcare providers

### Contributing

- **Code Improvements**: Submit pull requests
- **Documentation**: Help improve this manual
- **Testing**: Report bugs and suggest features

### Disclaimer

This software is provided for research and educational purposes. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare providers for medical advice.

---

## Version Information

- **Current Version**: 1.0
- **Last Updated**: [Current Date]
- **Compatibility**: Python 3.7+, PyTorch 1.7+

---

_This manual is a living document. Please check for updates and contribute improvements._
