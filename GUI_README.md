# Cobb Angle Analysis GUI Tool

A comprehensive graphical user interface for analyzing Cobb angles in spinal X-ray images with advanced visualization and customization features.

## Features

### 🖼️ Image Management

- **Single Image Selection**: Load individual X-ray images
- **Folder Selection**: Load entire folders of images for batch processing
- **Navigation**: Previous/Next buttons to browse through multiple images
- **Image Info Display**: Shows current image number and filename

### 🔍 Cobb Angle Detection

- **AI-Powered Detection**: Uses pre-trained deep learning model for landmark detection
- **Multi-Angle Analysis**: Detects primary, secondary, and tertiary Cobb angles
- **Real-time Processing**: Background thread processing to keep UI responsive
- **Results Display**: Detailed analysis results in text format

### 🎨 Visualization Features

- **Angle Toggle**: Show/hide individual angles (Primary, Secondary, Tertiary)
- **Customizable Colors**: Choose line and text colors
- **Adjustable Line Width**: Slider to control line thickness (1-10)
- **Text Size Control**: Slider to adjust text size (8-24)
- **Landmark Display**: Shows detected vertebral landmarks in green

### 🔍 Interactive Display

- **Zoomable Canvas**: Mouse wheel zoom in/out (0.1x to 5x)
- **Pan Support**: Click and drag to pan around the image
- **Reset View**: Button to reset zoom and pan to default
- **High-Quality Rendering**: Smooth image scaling and display

### 📊 Results Panel

- **Live Results**: Real-time display of analysis results
- **Severity Classification**: Automatic diagnosis based on Cobb angles
- **Landmark Count**: Shows number of detected landmarks
- **Angle Measurements**: Precise angle values for each detected curve

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. Install the required dependencies:

```bash
pip install -r requirements_gui.txt
```

2. Ensure the pretrained model is available:

```
pretrained_model/
└── model_last.pth
```

3. Run the GUI application:

```bash
python cobb_angle_gui.py
```

## Usage Guide

### 1. Loading Images

- **Single Image**: Click "Select Image" to choose a single X-ray file
- **Multiple Images**: Click "Select Folder" to load all images from a directory
- Supported formats: JPG, JPEG, PNG, BMP, TIFF

### 2. Navigation

- Use "◀ Previous" and "Next ▶" buttons to browse through loaded images
- Current image position is displayed in the top panel

### 3. Analysis

- Click "Detect Cobb Angles" to start the analysis
- The button will show "Detecting..." during processing
- Results appear in the left panel and on the image

### 4. Visualization Controls

- **Angle Visibility**: Check/uncheck boxes to show/hide specific angles
- **Line Width**: Adjust the slider to change line thickness
- **Text Size**: Adjust the slider to change text size
- **Colors**: Click color buttons to choose custom colors

### 5. Display Interaction

- **Zoom**: Use mouse wheel to zoom in/out
- **Pan**: Click and drag to move around the image
- **Reset**: Click "Reset View" to return to default view

## Color Coding

- **Green**: Detected vertebral landmarks
- **Red**: Primary Cobb angle (largest curve)
- **Magenta**: Secondary Cobb angle (compensatory curve)
- **Cyan**: Tertiary Cobb angle (minor curve)

## Severity Classification

The tool automatically classifies scoliosis severity based on Cobb angles:

- **< 10°**: Normal spine (no treatment needed)
- **10-25°**: Mild scoliosis (monitor for progression)
- **25-40°**: Moderate scoliosis (consider bracing)
- **> 40°**: Severe scoliosis (surgical consultation)

## Technical Details

### Model Architecture

- **Backbone**: ResNet-based spinal landmark detection
- **Output**: Heatmaps for landmark detection
- **Processing**: 1024x512 input resolution

### Performance

- **CPU Mode**: Compatible with all systems
- **GPU Mode**: CUDA acceleration for faster processing
- **Memory**: Efficient processing for large images

### File Structure

```
├── cobb_angle_gui.py          # Main GUI application
├── requirements_gui.txt       # Python dependencies
├── GUI_README.md             # This documentation
├── pretrained_model/         # Model directory
│   └── model_last.pth       # Pretrained weights
└── imgs/                    # Sample images
    ├── pic1.png
    └── pic2.png
```

## Troubleshooting

### Common Issues

1. **Model Not Found**

   - Ensure `pretrained_model/model_last.pth` exists
   - Check file permissions

2. **No Landmarks Detected**

   - Verify image quality and contrast
   - Ensure spine is clearly visible
   - Try adjusting image brightness/contrast

3. **GUI Not Responding**

   - Detection runs in background thread
   - Wait for "Detecting..." to complete
   - Check console for error messages

4. **Display Issues**
   - Update graphics drivers
   - Ensure sufficient screen resolution
   - Try different zoom levels

### Performance Tips

- Use GPU mode for faster processing
- Close other applications during analysis
- Use smaller images for quicker results
- Process images in batches for efficiency

## Support

For technical support or feature requests, please refer to the main project documentation or create an issue in the repository.

## License

This GUI tool is part of the Vertebra Landmark Detection project and follows the same licensing terms.
