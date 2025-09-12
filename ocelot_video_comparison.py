import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
import sys

def extract_frames_from_video(video_path, num_frames=15, output_dir="frames"):
    """Extract frames from video at regular intervals"""
    print(f"Processing video: {video_path}")
    
    # Create output directory
    video_name = Path(video_path).stem
    frame_dir = Path(output_dir) / video_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
    
    # Calculate frame intervals
    if total_frames < num_frames:
        frame_indices = list(range(total_frames))
    else:
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    extracted_frames = []
    frame_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            # Save frame
            frame_path = frame_dir / f"frame_{i:03d}_{frame_idx:05d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            
            extracted_frames.append(frame)
            frame_paths.append(str(frame_path))
            
            print(f"Extracted frame {i+1}/{len(frame_indices)} (frame #{frame_idx})")
        else:
            print(f"Could not read frame {frame_idx}")
    
    cap.release()
    print(f"Extracted {len(extracted_frames)} frames from {video_name}")
    return extracted_frames, frame_paths

def enhance_frame_patterns(frame):
    """Enhanced pattern processing for ocelot identification"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Noise reduction
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Edge detection for pattern enhancement
    edges = cv2.Canny(denoised, 30, 100)
    
    # Morphological operations to enhance patterns
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    return enhanced, denoised, edges, morphed

def detect_ocelot_patterns(enhanced_frame):
    """Detect ocelot-specific patterns: rosettes, spots, and stripes"""
    pattern_data = {}
    
    # 1. Rosette detection using circular/elliptical pattern detection
    rosettes = detect_rosette_patterns(enhanced_frame)
    pattern_data['rosettes'] = rosettes
    
    # 2. Spot detection using blob detection
    spots = detect_spot_patterns(enhanced_frame)
    pattern_data['spots'] = spots
    
    # 3. Traditional feature extraction for overall matching
    sift = cv2.SIFT_create(nfeatures=500)
    kp_sift, desc_sift = sift.detectAndCompute(enhanced_frame, None)
    pattern_data['sift'] = (kp_sift, desc_sift)
    
    orb = cv2.ORB_create(nfeatures=500)
    kp_orb, desc_orb = orb.detectAndCompute(enhanced_frame, None)
    pattern_data['orb'] = (kp_orb, desc_orb)
    
    return pattern_data

def detect_rosette_patterns(enhanced_frame):
    """Detect circular/elliptical rosette patterns characteristic of ocelots"""
    # Create mask for pattern detection
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Apply morphological operations to enhance rosette-like patterns
    opened = cv2.morphologyEx(enhanced_frame, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # Use HoughCircles to detect circular patterns
    circles = cv2.HoughCircles(
        closed,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,  # Minimum distance between rosette centers
        param1=50,   # Upper threshold for edge detection
        param2=25,   # Accumulator threshold for center detection
        minRadius=8,  # Minimum rosette radius
        maxRadius=40  # Maximum rosette radius
    )
    
    rosettes = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Validate rosette by checking if it has dark center and lighter rim
            if is_valid_rosette(enhanced_frame, x, y, r):
                rosettes.append({
                    'center': (x, y),
                    'radius': r,
                    'confidence': calculate_rosette_confidence(enhanced_frame, x, y, r)
                })
    
    return rosettes

def detect_spot_patterns(enhanced_frame):
    """Detect solid spot patterns using blob detection"""
    # Set up blob detector parameters for spots
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Area
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 800
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3
    
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6
    
    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    # Create detector
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Invert image for dark spot detection
    inverted = cv2.bitwise_not(enhanced_frame)
    
    # Detect blobs
    keypoints = detector.detect(inverted)
    
    spots = []
    for kp in keypoints:
        spots.append({
            'center': (int(kp.pt[0]), int(kp.pt[1])),
            'size': kp.size,
            'confidence': kp.response
        })
    
    return spots

def is_valid_rosette(image, x, y, radius):
    """Validate if detected circle is actually a rosette pattern"""
    h, w = image.shape
    if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
        return False
    
    # Extract region of interest
    roi = image[y-radius:y+radius, x-radius:x+radius]
    
    # Calculate center and rim intensities
    center_region = roi[radius//2:3*radius//2, radius//2:3*radius//2]
    center_intensity = np.mean(center_region)
    
    # Create ring mask for rim
    mask = np.zeros_like(roi)
    cv2.circle(mask, (radius, radius), radius, 255, 3)
    rim_intensity = np.mean(roi[mask > 0])
    
    # Rosette should have darker center than rim
    return center_intensity < rim_intensity * 0.8

def calculate_rosette_confidence(image, x, y, radius):
    """Calculate confidence score for rosette detection"""
    h, w = image.shape
    if x - radius < 0 or x + radius >= w or y - radius < 0 or y + radius >= h:
        return 0.0
    
    roi = image[y-radius:y+radius, x-radius:x+radius]
    
    # Calculate various metrics
    center_region = roi[radius//2:3*radius//2, radius//2:3*radius//2]
    center_intensity = np.mean(center_region)
    
    mask = np.zeros_like(roi)
    cv2.circle(mask, (radius, radius), radius, 255, 3)
    rim_intensity = np.mean(roi[mask > 0])
    
    # Confidence based on contrast ratio and pattern regularity
    contrast_ratio = rim_intensity / (center_intensity + 1)
    regularity = 1.0 - (np.std(roi) / (np.mean(roi) + 1))
    
    confidence = min(1.0, (contrast_ratio * 0.6 + regularity * 0.4))
    return confidence

def extract_robust_features(enhanced_frame):
    """Legacy function - maintained for backward compatibility"""
    return detect_ocelot_patterns(enhanced_frame)

def match_ocelot_patterns(patterns1, patterns2):
    """Match ocelot-specific patterns between frames"""
    pattern_matches = {
        'rosette_matches': [],
        'spot_matches': [],
        'rosette_score': 0,
        'spot_score': 0,
        'pattern_confidence': 0
    }
    
    # Match rosettes
    rosettes1 = patterns1.get('rosettes', [])
    rosettes2 = patterns2.get('rosettes', [])
    
    if rosettes1 and rosettes2:
        rosette_matches = match_rosettes(rosettes1, rosettes2)
        pattern_matches['rosette_matches'] = rosette_matches
        pattern_matches['rosette_score'] = calculate_rosette_match_score(rosette_matches, len(rosettes1), len(rosettes2))
    
    # Match spots
    spots1 = patterns1.get('spots', [])
    spots2 = patterns2.get('spots', [])
    
    if spots1 and spots2:
        spot_matches = match_spots(spots1, spots2)
        pattern_matches['spot_matches'] = spot_matches
        pattern_matches['spot_score'] = calculate_spot_match_score(spot_matches, len(spots1), len(spots2))
    
    # Calculate overall pattern confidence
    pattern_matches['pattern_confidence'] = (
        pattern_matches['rosette_score'] * 0.7 + 
        pattern_matches['spot_score'] * 0.3
    )
    
    return pattern_matches

def match_rosettes(rosettes1, rosettes2):
    """Match rosettes between two frames based on spatial relationship and size"""
    matches = []
    distance_threshold = 50  # Maximum pixel distance for matching
    size_threshold = 0.3     # Maximum relative size difference
    
    for r1 in rosettes1:
        best_match = None
        best_score = 0
        
        for r2 in rosettes2:
            # Calculate spatial distance
            dx = r1['center'][0] - r2['center'][0]
            dy = r1['center'][1] - r2['center'][1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Calculate size similarity
            size_diff = abs(r1['radius'] - r2['radius']) / max(r1['radius'], r2['radius'])
            
            # Calculate match score
            if distance < distance_threshold and size_diff < size_threshold:
                spatial_score = 1.0 - (distance / distance_threshold)
                size_score = 1.0 - (size_diff / size_threshold)
                confidence_score = (r1['confidence'] + r2['confidence']) / 2
                
                match_score = spatial_score * 0.4 + size_score * 0.3 + confidence_score * 0.3
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = {
                        'rosette1': r1,
                        'rosette2': r2,
                        'score': match_score,
                        'distance': distance
                    }
        
        if best_match and best_score > 0.5:
            matches.append(best_match)
    
    return matches

def match_spots(spots1, spots2):
    """Match spots between two frames"""
    matches = []
    distance_threshold = 40
    size_threshold = 0.4
    
    for s1 in spots1:
        best_match = None
        best_score = 0
        
        for s2 in spots2:
            dx = s1['center'][0] - s2['center'][0]
            dy = s1['center'][1] - s2['center'][1]
            distance = np.sqrt(dx*dx + dy*dy)
            
            size_diff = abs(s1['size'] - s2['size']) / max(s1['size'], s2['size'])
            
            if distance < distance_threshold and size_diff < size_threshold:
                spatial_score = 1.0 - (distance / distance_threshold)
                size_score = 1.0 - (size_diff / size_threshold)
                confidence_score = (s1['confidence'] + s2['confidence']) / 2
                
                match_score = spatial_score * 0.5 + size_score * 0.3 + confidence_score * 0.2
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = {
                        'spot1': s1,
                        'spot2': s2,
                        'score': match_score,
                        'distance': distance
                    }
        
        if best_match and best_score > 0.4:
            matches.append(best_match)
    
    return matches

def calculate_rosette_match_score(matches, count1, count2):
    """Calculate rosette matching score"""
    if not matches or (count1 == 0 and count2 == 0):
        return 0
    
    match_quality = np.mean([m['score'] for m in matches]) if matches else 0
    match_ratio = len(matches) / max(count1, count2, 1)
    
    return min(100, (match_quality * 0.6 + match_ratio * 0.4) * 100)

def calculate_spot_match_score(matches, count1, count2):
    """Calculate spot matching score"""
    if not matches or (count1 == 0 and count2 == 0):
        return 0
    
    match_quality = np.mean([m['score'] for m in matches]) if matches else 0
    match_ratio = len(matches) / max(count1, count2, 1)
    
    return min(100, (match_quality * 0.7 + match_ratio * 0.3) * 100)

def match_features_between_frames(features1, features2, method='sift'):
    """Match features between two frames - enhanced for pattern analysis"""
    # First check for pattern matching
    if method == 'patterns':
        return match_ocelot_patterns(features1, features2)
    
    # Traditional feature matching
    kp1, desc1 = features1[method]
    kp2, desc2 = features2[method]
    
    if desc1 is None or desc2 is None or len(desc1) < 10 or len(desc2) < 10:
        return [], 0
    
    if method == 'sift':
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.65 * n.distance:
                    good_matches.append(m)
    
    elif method == 'orb':
        # Brute force matcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    similarity_score = min(100, (len(good_matches) / 30) * 100)
    
    return good_matches, similarity_score

def analyze_video_comparison(video1_path, video2_path):
    """Main function to compare ocelot videos"""
    
    print("=== OCELOT VIDEO COMPARISON ANALYSIS ===")
    print(f"Video 1: {Path(video1_path).name}")
    print(f"Video 2: {Path(video2_path).name}")
    print("="*50)
    
    # Extract frames from both videos
    print("\n1. EXTRACTING FRAMES...")
    frames1, paths1 = extract_frames_from_video(video1_path, num_frames=12)
    frames2, paths2 = extract_frames_from_video(video2_path, num_frames=12)
    
    if not frames1 or not frames2:
        print("Error: Could not extract frames from one or both videos")
        return
    
    print(f"\nExtracted {len(frames1)} frames from video 1")
    print(f"Extracted {len(frames2)} frames from video 2")
    
    # Process frames and extract features
    print("\n2. PROCESSING FRAMES AND EXTRACTING FEATURES...")
    
    features1_list = []
    features2_list = []
    enhanced1_list = []
    enhanced2_list = []
    
    for i, frame in enumerate(frames1):
        enhanced, denoised, edges, morphed = enhance_frame_patterns(frame)
        features = extract_robust_features(denoised)
        features1_list.append(features)
        enhanced1_list.append(denoised)
        print(f"Processed frame {i+1}/{len(frames1)} from video 1")
    
    for i, frame in enumerate(frames2):
        enhanced, denoised, edges, morphed = enhance_frame_patterns(frame)
        features = extract_robust_features(denoised)
        features2_list.append(features)
        enhanced2_list.append(denoised)
        print(f"Processed frame {i+1}/{len(frames2)} from video 2")
    
    # Compare all frame combinations with pattern analysis
    print("\n3. ANALYZING OCELOT PATTERNS ACROSS ALL FRAME COMBINATIONS...")
    
    comparison_results = {
        'sift_scores': [],
        'orb_scores': [],
        'pattern_scores': [],
        'rosette_scores': [],
        'spot_scores': [],
        'best_matches': [],
        'pattern_matrix': []
    }
    
    # Create matrices for visualization
    pattern_matrix = np.zeros((len(features1_list), len(features2_list)))
    rosette_matrix = np.zeros((len(features1_list), len(features2_list)))
    spot_matrix = np.zeros((len(features1_list), len(features2_list)))
    
    for i, feat1 in enumerate(features1_list):
        for j, feat2 in enumerate(features2_list):
            # Traditional feature matching
            sift_matches, sift_score = match_features_between_frames(feat1, feat2, 'sift')
            orb_matches, orb_score = match_features_between_frames(feat1, feat2, 'orb')
            
            # Pattern-specific matching
            pattern_match_data = match_ocelot_patterns(feat1, feat2)
            pattern_score = pattern_match_data['pattern_confidence']
            rosette_score = pattern_match_data['rosette_score']
            spot_score = pattern_match_data['spot_score']
            
            # Store results
            comparison_results['sift_scores'].append(sift_score)
            comparison_results['orb_scores'].append(orb_score)
            comparison_results['pattern_scores'].append(pattern_score)
            comparison_results['rosette_scores'].append(rosette_score)
            comparison_results['spot_scores'].append(spot_score)
            
            # Fill matrices for visualization
            pattern_matrix[i, j] = pattern_score
            rosette_matrix[i, j] = rosette_score
            spot_matrix[i, j] = spot_score
            
            # Store significant matches
            if pattern_score > 20 or sift_score > 25 or orb_score > 25:
                comparison_results['best_matches'].append({
                    'frame1_idx': i,
                    'frame2_idx': j,
                    'sift_score': sift_score,
                    'orb_score': orb_score,
                    'pattern_score': pattern_score,
                    'rosette_score': rosette_score,
                    'spot_score': spot_score,
                    'sift_matches': len(sift_matches),
                    'orb_matches': len(orb_matches),
                    'rosette_matches': len(pattern_match_data['rosette_matches']),
                    'spot_matches': len(pattern_match_data['spot_matches'])
                })
    
    # Store matrices for visualization
    comparison_results['pattern_matrix'] = pattern_matrix
    comparison_results['rosette_matrix'] = rosette_matrix
    comparison_results['spot_matrix'] = spot_matrix
    
    # Statistical analysis with pattern metrics
    print("\n4. OCELOT PATTERN STATISTICAL ANALYSIS...")
    
    sift_scores = comparison_results['sift_scores']
    orb_scores = comparison_results['orb_scores']
    pattern_scores = comparison_results['pattern_scores']
    rosette_scores = comparison_results['rosette_scores']
    spot_scores = comparison_results['spot_scores']
    
    sift_stats = {
        'max': max(sift_scores) if sift_scores else 0,
        'mean': np.mean(sift_scores) if sift_scores else 0,
        'std': np.std(sift_scores) if sift_scores else 0,
        'above_threshold': sum(1 for s in sift_scores if s > 25)
    }
    
    orb_stats = {
        'max': max(orb_scores) if orb_scores else 0,
        'mean': np.mean(orb_scores) if orb_scores else 0,
        'std': np.std(orb_scores) if orb_scores else 0,
        'above_threshold': sum(1 for s in orb_scores if s > 25)
    }
    
    pattern_stats = {
        'max': max(pattern_scores) if pattern_scores else 0,
        'mean': np.mean(pattern_scores) if pattern_scores else 0,
        'std': np.std(pattern_scores) if pattern_scores else 0,
        'above_threshold': sum(1 for s in pattern_scores if s > 30)
    }
    
    rosette_stats = {
        'max': max(rosette_scores) if rosette_scores else 0,
        'mean': np.mean(rosette_scores) if rosette_scores else 0,
        'std': np.std(rosette_scores) if rosette_scores else 0,
        'above_threshold': sum(1 for s in rosette_scores if s > 25)
    }
    
    spot_stats = {
        'max': max(spot_scores) if spot_scores else 0,
        'mean': np.mean(spot_scores) if spot_scores else 0,
        'std': np.std(spot_scores) if spot_scores else 0,
        'above_threshold': sum(1 for s in spot_scores if s > 25)
    }
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("OCELOT IDENTIFICATION ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\nTOTAL FRAME COMPARISONS: {len(sift_scores)}")
    print(f"Video 1 frames: {len(frames1)}")
    print(f"Video 2 frames: {len(frames2)}")
    
    print(f"\nSIFT FEATURE ANALYSIS:")
    print(f"  Max similarity: {sift_stats['max']:.1f}%")
    print(f"  Average similarity: {sift_stats['mean']:.1f}%")
    print(f"  Standard deviation: {sift_stats['std']:.1f}%")
    print(f"  Comparisons above 25% threshold: {sift_stats['above_threshold']}")
    
    print(f"\nORB FEATURE ANALYSIS:")
    print(f"  Max similarity: {orb_stats['max']:.1f}%")
    print(f"  Average similarity: {orb_stats['mean']:.1f}%")
    print(f"  Standard deviation: {orb_stats['std']:.1f}%")
    print(f"  Comparisons above 25% threshold: {orb_stats['above_threshold']}")
    
    print(f"\nOCELOT PATTERN ANALYSIS:")
    print(f"  Max pattern similarity: {pattern_stats['max']:.1f}%")
    print(f"  Average pattern similarity: {pattern_stats['mean']:.1f}%")
    print(f"  Pattern consistency (std dev): {pattern_stats['std']:.1f}%")
    print(f"  Strong pattern matches (>30%): {pattern_stats['above_threshold']}")
    
    print(f"\nROSETTE PATTERN ANALYSIS:")
    print(f"  Max rosette similarity: {rosette_stats['max']:.1f}%")
    print(f"  Average rosette similarity: {rosette_stats['mean']:.1f}%")
    print(f"  Rosette match consistency: {rosette_stats['std']:.1f}%")
    print(f"  Strong rosette matches: {rosette_stats['above_threshold']}")
    
    print(f"\nSPOT PATTERN ANALYSIS:")
    print(f"  Max spot similarity: {spot_stats['max']:.1f}%")
    print(f"  Average spot similarity: {spot_stats['mean']:.1f}%")
    print(f"  Spot match consistency: {spot_stats['std']:.1f}%")
    print(f"  Strong spot matches: {spot_stats['above_threshold']}")
    
    # Determine overall assessment with pattern weighting
    max_pattern_similarity = pattern_stats['max']
    max_traditional_similarity = max(sift_stats['max'], orb_stats['max'])
    max_similarity = max(max_pattern_similarity, max_traditional_similarity)
    
    # Weight pattern matches higher for ocelot identification
    pattern_weight = 2.0  # Pattern matches are more reliable for individual ID
    weighted_score = (
        pattern_stats['above_threshold'] * pattern_weight + 
        rosette_stats['above_threshold'] * 1.5 +
        spot_stats['above_threshold'] * 1.0 +
        sift_stats['above_threshold'] * 0.5 +
        orb_stats['above_threshold'] * 0.5
    )
    
    print(f"\n" + "="*60)
    print("INDIVIDUAL IDENTIFICATION ASSESSMENT")
    print("="*60)
    
    # Enhanced confidence assessment based on pattern analysis
    if max_pattern_similarity > 50 and weighted_score > 8:
        conclusion = "HIGH CONFIDENCE - SAME INDIVIDUAL"
        confidence = "90-95%"
    elif max_pattern_similarity > 35 and weighted_score > 5:
        conclusion = "MODERATE-HIGH CONFIDENCE - LIKELY SAME INDIVIDUAL"
        confidence = "75-85%"
    elif max_pattern_similarity > 20 and weighted_score > 3:
        conclusion = "MODERATE CONFIDENCE - POSSIBLY SAME INDIVIDUAL"
        confidence = "55-70%"
    elif max_similarity > 30 and weighted_score > 2:
        conclusion = "LOW-MODERATE CONFIDENCE - UNCERTAIN"
        confidence = "35-50%"
    else:
        conclusion = "LOW CONFIDENCE - LIKELY DIFFERENT INDIVIDUALS"
        confidence = "15-35%"
    
    print(f"CONCLUSION: {conclusion}")
    print(f"CONFIDENCE LEVEL: {confidence}")
    print(f"MAX PATTERN SIMILARITY: {max_pattern_similarity:.1f}%")
    print(f"MAX OVERALL SIMILARITY: {max_similarity:.1f}%")
    print(f"WEIGHTED PATTERN SCORE: {weighted_score:.1f}")
    
    # Create comprehensive visualizations
    print(f"\n5. CREATING PATTERN ANALYSIS VISUALIZATIONS...")
    
    # Pattern profile visualization
    create_pattern_profile_visualization(comparison_results, len(frames1), len(frames2))
    
    # Best matches visualization
    if comparison_results['best_matches']:
        create_enhanced_matches_visualization(
            frames1, frames2, enhanced1_list, enhanced2_list, 
            comparison_results['best_matches'][:6]
        )
    
    # Save detailed results with pattern analysis
    results = {
        'video1': Path(video1_path).name,
        'video2': Path(video2_path).name,
        'conclusion': conclusion,
        'confidence': confidence,
        'max_similarity': max_similarity,
        'max_pattern_similarity': max_pattern_similarity,
        'weighted_score': weighted_score,
        'sift_stats': sift_stats,
        'orb_stats': orb_stats,
        'pattern_stats': pattern_stats,
        'rosette_stats': rosette_stats,
        'spot_stats': spot_stats,
        'total_comparisons': len(sift_scores),
        'best_matches': comparison_results['best_matches'][:10]
    }
    
    with open('ocelot_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to 'ocelot_analysis_results.json'")

def create_best_matches_visualization(frames1, frames2, enhanced1, enhanced2, best_matches):
    """Create visualization of the best matching frame pairs"""
    if not best_matches:
        print("No significant matches found for visualization")
        return
    
    # Sort by best overall score
    best_matches.sort(key=lambda x: max(x['sift_score'], x['orb_score']), reverse=True)
    
    num_pairs = min(3, len(best_matches))  # Show top 3 pairs
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(16, 6*num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Best Matching Frame Pairs - Ocelot Pattern Comparison', 
                 fontsize=16, fontweight='bold')
    
    for i, match in enumerate(best_matches[:num_pairs]):
        frame1_idx = match['frame1_idx']
        frame2_idx = match['frame2_idx']
        
        # Display enhanced frames
        axes[i, 0].imshow(enhanced1[frame1_idx], cmap='gray')
        axes[i, 0].set_title(f'Video 1 - Frame {frame1_idx+1}\n'
                            f'SIFT: {match["sift_score"]:.1f}% | ORB: {match["orb_score"]:.1f}%')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(enhanced2[frame2_idx], cmap='gray')
        axes[i, 1].set_title(f'Video 2 - Frame {frame2_idx+1}\n'
                            f'SIFT Matches: {match["sift_matches"]} | ORB Matches: {match["orb_matches"]}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('ocelot_best_matches.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Best matches visualization saved as 'ocelot_best_matches.png'")

def create_pattern_profile_visualization(comparison_results, num_frames1, num_frames2):
    """Create comprehensive pattern analysis profile visualization"""
    
    # Extract matrices
    pattern_matrix = comparison_results['pattern_matrix']
    rosette_matrix = comparison_results['rosette_matrix'] 
    spot_matrix = comparison_results['spot_matrix']
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Main title
    fig.suptitle('Ocelot Pattern Analysis Profile - Individual Identification', 
                 fontsize=16, fontweight='bold')
    
    # 1. Pattern Similarity Heatmap
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(pattern_matrix, cmap='hot', interpolation='nearest', aspect='auto')
    ax1.set_title('Overall Pattern Similarity Matrix')
    ax1.set_xlabel('Video 2 Frames')
    ax1.set_ylabel('Video 1 Frames') 
    plt.colorbar(im1, ax=ax1, label='Similarity %')
    
    # 2. Rosette Pattern Heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(rosette_matrix, cmap='Reds', interpolation='nearest', aspect='auto')
    ax2.set_title('Rosette Pattern Matches')
    ax2.set_xlabel('Video 2 Frames')
    ax2.set_ylabel('Video 1 Frames')
    plt.colorbar(im2, ax=ax2, label='Rosette Score %')
    
    # 3. Spot Pattern Heatmap  
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(spot_matrix, cmap='Blues', interpolation='nearest', aspect='auto')
    ax3.set_title('Spot Pattern Matches')
    ax3.set_xlabel('Video 2 Frames')
    ax3.set_ylabel('Video 1 Frames')
    plt.colorbar(im3, ax=ax3, label='Spot Score %')
    
    # 4. Pattern Profile Lines
    ax4 = plt.subplot(2, 3, 4)
    max_pattern_per_frame1 = np.max(pattern_matrix, axis=1)
    max_pattern_per_frame2 = np.max(pattern_matrix, axis=0)
    
    ax4.plot(range(num_frames1), max_pattern_per_frame1, 'ro-', 
             label='Video 1 max pattern match per frame', markersize=6)
    ax4.plot(range(num_frames2), max_pattern_per_frame2, 'bs-',
             label='Video 2 max pattern match per frame', markersize=6)
    ax4.set_title('Pattern Similarity Profile Lines')
    ax4.set_xlabel('Frame Number')
    ax4.set_ylabel('Max Pattern Similarity %')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Pattern Consistency Analysis
    ax5 = plt.subplot(2, 3, 5)
    pattern_variance_v1 = np.var(pattern_matrix, axis=1)
    pattern_variance_v2 = np.var(pattern_matrix, axis=0)
    
    ax5.plot(range(num_frames1), pattern_variance_v1, 'go-', 
             label='Video 1 pattern consistency', markersize=5)
    ax5.plot(range(num_frames2), pattern_variance_v2, 'mo-',
             label='Video 2 pattern consistency', markersize=5)
    ax5.set_title('Pattern Consistency Across Frames')
    ax5.set_xlabel('Frame Number')
    ax5.set_ylabel('Pattern Variance (lower = more consistent)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Combined Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    
    # Calculate summary statistics
    overall_pattern_score = np.max(pattern_matrix)
    overall_rosette_score = np.max(rosette_matrix)
    overall_spot_score = np.max(spot_matrix)
    pattern_consistency = 100 - np.mean([np.std(pattern_matrix), np.std(rosette_matrix), np.std(spot_matrix)])
    
    metrics = ['Overall Pattern', 'Best Rosette', 'Best Spot', 'Consistency']
    values = [overall_pattern_score, overall_rosette_score, overall_spot_score, max(0, pattern_consistency)]
    colors = ['red', 'orange', 'blue', 'green']
    
    bars = ax6.bar(metrics, values, color=colors, alpha=0.7)
    ax6.set_title('Pattern Analysis Summary')
    ax6.set_ylabel('Score %')
    ax6.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Add confidence assessment text
    max_score = max(overall_pattern_score, overall_rosette_score, overall_spot_score)
    if max_score > 50:
        confidence_text = "HIGH CONFIDENCE MATCH"
        text_color = 'green'
    elif max_score > 30:
        confidence_text = "MODERATE CONFIDENCE"  
        text_color = 'orange'
    else:
        confidence_text = "LOW CONFIDENCE"
        text_color = 'red'
    
    ax6.text(0.5, 0.95, confidence_text, transform=ax6.transAxes, 
             ha='center', va='top', fontsize=12, fontweight='bold', color=text_color,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('ocelot_pattern_profile_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Pattern profile analysis saved as 'ocelot_pattern_profile_analysis.png'")

def create_enhanced_matches_visualization(frames1, frames2, enhanced1, enhanced2, best_matches):
    """Create enhanced visualization with pattern overlay"""
    if not best_matches:
        print("No significant matches found for enhanced visualization")
        return
    
    # Sort by pattern score first, then overall score
    best_matches.sort(key=lambda x: (x.get('pattern_score', 0), max(x['sift_score'], x['orb_score'])), reverse=True)
    
    num_pairs = min(3, len(best_matches))
    
    fig, axes = plt.subplots(num_pairs, 2, figsize=(16, 6*num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Enhanced Ocelot Pattern Matching - Best Frame Pairs', 
                 fontsize=16, fontweight='bold')
    
    for i, match in enumerate(best_matches[:num_pairs]):
        frame1_idx = match['frame1_idx']
        frame2_idx = match['frame2_idx']
        
        # Display enhanced frames
        axes[i, 0].imshow(enhanced1[frame1_idx], cmap='gray')
        pattern_info = f"Pattern: {match.get('pattern_score', 0):.1f}%"
        rosette_info = f"Rosettes: {match.get('rosette_matches', 0)}"
        spot_info = f"Spots: {match.get('spot_matches', 0)}"
        title1 = f'Video 1 - Frame {frame1_idx+1}\n{pattern_info}\nSIFT: {match["sift_score"]:.1f}% | ORB: {match["orb_score"]:.1f}%'
        axes[i, 0].set_title(title1, fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(enhanced2[frame2_idx], cmap='gray')
        title2 = f'Video 2 - Frame {frame2_idx+1}\n{rosette_info} | {spot_info}\nFeature Matches: SIFT={match["sift_matches"]} | ORB={match["orb_matches"]}'
        axes[i, 1].set_title(title2, fontsize=10)
        axes[i, 1].axis('off')
        
        # Add confidence indicator
        confidence_score = max(match.get('pattern_score', 0), match['sift_score'], match['orb_score'])
        if confidence_score > 50:
            border_color = 'green'
        elif confidence_score > 30:
            border_color = 'orange'
        else:
            border_color = 'red'
        
        # Add colored border
        for ax in [axes[i, 0], axes[i, 1]]:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('ocelot_enhanced_pattern_matches.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Enhanced pattern matches saved as 'ocelot_enhanced_pattern_matches.png'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 ocelot_video_comparison.py <video1_path> <video2_path>")
        print("Example: python3 ocelot_video_comparison.py 1.mov 2.mov")
        sys.exit(1)
    
    video1_path = sys.argv[1]
    video2_path = sys.argv[2]
    
    # Check if files exist
    if not os.path.exists(video1_path):
        print(f"Error: Video file '{video1_path}' not found")
        sys.exit(1)
    
    if not os.path.exists(video2_path):
        print(f"Error: Video file '{video2_path}' not found")
        sys.exit(1)
    
    analyze_video_comparison(video1_path, video2_path)