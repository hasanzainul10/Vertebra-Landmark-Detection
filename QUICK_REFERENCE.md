# Cobb Angle Analysis - Quick Reference Guide

## Quick Commands

### Basic Analysis

```bash
# Analyze single image
python cobb_angle_inference.py --image_path xray.jpg

# Use CPU instead of GPU
python cobb_angle_inference.py --image_path xray.jpg --device cpu

# Don't save files (display only)
python cobb_angle_inference.py --image_path xray.jpg --no_save
```

### Advanced Analysis

```bash
# Custom model and output directory
python cobb_angle_inference.py \
    --image_path xray.jpg \
    --model_path my_model.pth \
    --output_dir my_results

# Basic landmark detection only
python evaluate_single_image.py --image_path xray.jpg
```

## Color Coding Reference

| Color       | Meaning            | Description                         |
| ----------- | ------------------ | ----------------------------------- |
| **Blue**    | Detected Vertebrae | All AI-detected vertebral landmarks |
| **Green**   | Primary Angle      | Largest Cobb angle (main curve)     |
| **Magenta** | Secondary Angle    | Second largest Cobb angle           |
| **Cyan**    | Tertiary Angle     | Third largest Cobb angle            |

## Severity Classification

| Angle Range | Severity | Action                    |
| ----------- | -------- | ------------------------- |
| < 10°       | Normal   | No treatment needed       |
| 10° - 25°   | Mild     | Monitor, possible bracing |
| 25° - 40°   | Moderate | Bracing, physical therapy |
| > 40°       | Severe   | Surgical consultation     |

## Common Issues & Solutions

### Problem: "No landmarks detected!"

**Solution**:

- Check image quality (high contrast, clear spine)
- Ensure spine is centered and visible
- Try different image format

### Problem: CUDA out of memory

**Solution**:

```bash
python cobb_angle_inference.py --image_path xray.jpg --device cpu
```

### Problem: Model not found

**Solution**:

- Verify `pretrained_model/model_last.pth` exists
- Check file permissions
- Download model file again

### Problem: Poor detection accuracy

**Solution**:

- Use high-resolution images (1024x512+)
- Ensure good contrast
- Use standard AP radiographic views

## Output Files

| File                  | Description                          |
| --------------------- | ------------------------------------ |
| `*_cobb_analysis.jpg` | Visualized results with measurements |
| `*_landmarks.npy`     | Detected landmark coordinates        |
| `*_cobb_angles.npy`   | Calculated Cobb angles               |
| `*_report.txt`        | Detailed analysis report             |

## Understanding Results

### Example Output

```
Primary: 34.5°     ← Main curve (moderate scoliosis)
Secondary: 29.3°   ← Compensatory curve
Tertiary: 5.6°     ← Minor curve (normal variation)
Maximum: 34.5°     ← Highest angle
Diagnosis: Moderate Scoliosis
```

### Key Points

- **Primary angle** determines treatment approach
- **Secondary angle** helps assess spinal balance
- **Tertiary angle** usually not clinically significant
- **Maximum angle** = Primary angle (largest curve)

## Performance Tips

### For Speed

- Use GPU (`--device cuda`)
- Close other applications
- Use SSD storage

### For Accuracy

- High-resolution images (1024x512 minimum)
- Good image contrast
- Standard radiographic positioning
- Patient still during imaging

## Medical Notes

### When to Measure Cobb Angles

- Initial scoliosis diagnosis
- Monitoring curve progression
- Pre/post treatment assessment
- Surgical planning

### Clinical Thresholds

- **< 10°**: Normal variation
- **10-25°**: Monitor for progression
- **25-40°**: Consider bracing
- **> 40°**: Surgical evaluation

### Important Considerations

- Results should be verified by medical professionals
- Tool is for research/educational purposes
- Clinical decisions require qualified healthcare providers
- Compare angles over time for progression assessment

## Emergency Contacts

### Technical Support

- Check documentation first
- Report issues with detailed information
- Include error messages and system specs

### Medical Questions

- Consult qualified healthcare providers
- This tool is not for clinical diagnosis
- Always seek professional medical advice

---

**Remember**: This tool is for research and educational purposes. Clinical decisions should be made by qualified healthcare providers.
