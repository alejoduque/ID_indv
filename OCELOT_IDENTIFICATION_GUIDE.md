# Ocelot Individual Identification System

## Quick Start Guide

### Prerequisites & Setup
1. **Create and activate a virtual environment** (recommended):
```bash
python3 -m venv ocelot_env
source ocelot_env/bin/activate  # On Windows: ocelot_env\Scripts\activate
```

2. **Install required Python packages**:
```bash
pip install opencv-python numpy matplotlib
```

### Basic Usage

#### 1. Compare Two Video Files
```bash
# Remember to activate the virtual environment first
source ocelot_env/bin/activate
python3 ocelot_video_comparison.py video1.mov video2.mov
```

**Example with test files:**
```bash
source ocelot_env/bin/activate
python3 ocelot_video_comparison.py 1.mov 2.mov
```

**Expected output:** `MODERATE-HIGH CONFIDENCE - LIKELY SAME INDIVIDUAL (75-85%)`

#### 2. Compare Two Static Images
```bash
python3 ocelot_pattern_comparison.py image1.jpg image2.jpg
```

### What the System Does

1. **Extracts Frames**: Takes 12 evenly distributed frames from each video
2. **Detects Patterns**: Identifies ocelot-specific features:
   - **Rosettes** (circular spotted patterns)
   - **Spots** (solid dark markings)
   - **Traditional features** (SIFT/ORB for general texture)

3. **Compares Patterns**: Analyzes all frame combinations (144 total comparisons)
4. **Generates Analysis**: Creates visual reports and confidence scores

### Output Files Generated

After running the comparison, you'll get:

#### Visual Reports:
- `ocelot_pattern_profile_analysis.png` - **6-panel comprehensive analysis**
  - Pattern similarity heatmaps
  - Profile lines showing match trends
  - Consistency analysis across frames
  - Summary confidence dashboard

- `ocelot_enhanced_pattern_matches.png` - **Best matching frame pairs**
  - Shows top 3 frame matches
  - Pattern scores and feature counts
  - Color-coded confidence borders

- `ocelot_best_matches.png` - **Traditional feature matching visualization**

#### Data Files:
- `ocelot_analysis_results.json` - **Detailed metrics and statistics**
- `frames/` directory - **Extracted frames from both videos**

### Interpreting Results

#### Confidence Levels:
- **90-95%**: HIGH CONFIDENCE - Same individual
- **75-85%**: MODERATE-HIGH CONFIDENCE - Likely same individual  
- **55-70%**: MODERATE CONFIDENCE - Possibly same individual
- **35-50%**: LOW-MODERATE CONFIDENCE - Uncertain
- **15-35%**: LOW CONFIDENCE - Likely different individuals

#### Key Metrics to Look For:
- **Max Pattern Similarity**: Overall best pattern match score
- **Rosette Matches**: Number of matching rosette patterns
- **Spot Matches**: Number of matching spot patterns
- **Weighted Score**: Combined confidence taking pattern priority into account

### Understanding the Visual Analysis

#### Pattern Profile Analysis (6-panel view):
1. **Top Left**: Overall pattern similarity heatmap
2. **Top Middle**: Rosette-specific matches (red colormap)
3. **Top Right**: Spot-specific matches (blue colormap)
4. **Bottom Left**: Profile lines showing similarity trends
5. **Bottom Middle**: Pattern consistency across frames
6. **Bottom Right**: Summary bar chart with confidence assessment

#### Enhanced Pattern Matches:
- **Green borders**: High confidence matches (>50%)
- **Orange borders**: Moderate confidence matches (30-50%)
- **Red borders**: Low confidence matches (<30%)

### Tips for Best Results

1. **Video Quality**: Use videos with clear, well-lit ocelot footage
2. **Multiple Angles**: Videos showing side profiles work best for pattern comparison
3. **Stable Footage**: Less motion blur = better pattern detection
4. **Duration**: Longer videos provide more frames for analysis

### Troubleshooting

#### If you get errors:
- Check that video files exist in the current directory
- Ensure OpenCV can read your video format
- Verify Python packages are installed correctly

#### If confidence seems low:
- Try videos with better lighting
- Ensure the ocelot is clearly visible in most frames
- Check that both videos show similar body regions

### Example Command
```bash
# Compare two ocelot videos
python3 ocelot_video_comparison.py 1.mov 2.mov

# The system will:
# 1. Extract 12 frames from each video
# 2. Detect rosettes, spots, and features
# 3. Compare all 144 frame combinations  
# 4. Generate visual analysis reports
# 5. Output confidence assessment
```

### Advanced Usage

The system automatically handles:
- Frame extraction at optimal intervals
- Image enhancement for pattern visibility
- Multi-algorithm pattern matching
- Statistical confidence assessment
- Comprehensive visualization generation

No additional parameters needed - just provide the two video file paths!