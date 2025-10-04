# Ocelot Re-Identification System - User Guide

## 📋 Overview
This system uses computer vision and pattern matching to identify individual ocelots based on their unique spot patterns. It works with both images and videos.

## 🔧 Installation

### Required Libraries
```bash
pip install opencv-python numpy matplotlib scipy scikit-image
```

### Verify Installation
```bash
python -c "import cv2; import numpy; import matplotlib; print('All libraries installed!')"
```

## 🚀 Quick Start

### Basic Usage (Interactive Mode)
```bash
python ocelot_reid_interactive.py
```

The system will prompt you to:
1. Choose between image or video mode
2. Enter paths to your files
3. Optionally save visualizations

### Image Comparison
```bash
# When prompted, choose option 1 (Image Mode)
# Then enter your image paths:
Image 1: path/to/ocelot1.jpg
Image 2: path/to/ocelot2.jpg
```

### Video Comparison
```bash
# When prompted, choose option 2 (Video Mode)
# Then enter your video paths:
Video 1: path/to/ocelot_video1.mp4
Video 2: path/to/ocelot_video2.mp4
```

## 📊 Understanding Results

### Confidence Levels
- **75-100%**: VERY HIGH - Same individual
- **60-74%**: HIGH - Likely same individual
- **45-59%**: MODERATE - Possibly same individual
- **30-44%**: LOW - Unlikely same individual
- **0-29%**: VERY LOW - Different individual

### Component Scores
1. **SIFT Features** (35% weight): Unique spot pattern keypoints
2. **ORB Features** (20% weight): Fast pattern descriptors
3. **Histogram Similarity** (25% weight): Overall color/texture distribution
4. **Pattern Statistics** (20% weight): Spot count, size, and density

## 🎯 Best Practices

### Image Quality
- ✅ Clear, well-lit photos
- ✅ Focus on coat pattern areas (body, legs)
- ✅ Minimum resolution: 800x600 pixels
- ❌ Avoid blurry or dark images
- ❌ Avoid extreme angles

### Video Tips
- The system samples 10 frames evenly distributed throughout the video
- Longer videos with varied angles produce better results
- Ensure ocelot is visible in most frames

## 📁 Output Files

### Image Mode
- `ocelot_reid_analysis.png`: Detailed visual comparison

### Video Mode
- `video_reid_analysis.png`: Frame-by-frame comparison grid
- `best_match_frames.png`: Side-by-side of best matching frames

## 🔍 Advanced Usage

### Database Search
To identify an ocelot against multiple known individuals:

```python
from ocelot_reid_interactive import OcelotReID

reid = OcelotReID()
database = ['known_ocelot1.jpg', 'known_ocelot2.jpg', 'known_ocelot3.jpg']
results = reid.identify_individual('unknown_ocelot.jpg', database)
```

### Custom Thresholds
Modify confidence thresholds in `_get_confidence_level()` method to adjust sensitivity.

## ⚠️ Troubleshooting

### "Could not load image" error
- Check file path is correct
- Ensure image format is supported (JPG, PNG, BMP)
- Verify file is not corrupted

### Low scores on same individual
- Images may have different lighting conditions
- Try images with similar angles/perspectives
- Ensure both images show clear spot patterns

### Video processing is slow
- Normal for long videos (processing takes ~2-3 seconds per frame)
- Consider extracting key frames manually for faster processing

## 📖 Example Workflow

### Scenario: Field Research
1. Collect camera trap footage of ocelots
2. Extract videos or images of individual sightings
3. Run comparison against existing database
4. Review confidence scores and visualizations
5. Update database with confirmed new individuals

### Sample Command Flow
```
$ python ocelot_reid_interactive.py

=== OCELOT RE-IDENTIFICATION SYSTEM ===
1. Compare Images
2. Compare Videos
3. Exit
Choice: 1

Enter path to first image: field_data/ocelot_trap1_day5.jpg
Enter path to second image: field_data/ocelot_trap2_day12.jpg
Save visualization? (y/n): y

Processing...
Overall Score: 78.3%
Confidence: VERY HIGH - Same individual

Visualization saved to: ocelot_reid_analysis.png
```

## 🆘 Support

For issues or questions:
- Check image/video file paths are correct
- Ensure all dependencies are installed
- Verify Python version is 3.7+
- Review output visualizations for quality issues

## 📝 Notes

- First run may take longer as OpenCV initializes
- Video processing requires ffmpeg (usually included with opencv-python)
- Large video files may require significant processing time
- System works best with dorsal (back) and lateral (side) views of ocelots