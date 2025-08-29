import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

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

def extract_robust_features(enhanced_frame):
    """Extract multiple types of features for robust matching"""
    features_data = {}
    
    # SIFT features
    sift = cv2.SIFT_create(nfeatures=500)
    kp_sift, desc_sift = sift.detectAndCompute(enhanced_frame, None)
    features_data['sift'] = (kp_sift, desc_sift)
    
    # ORB features (more robust to lighting changes)
    orb = cv2.ORB_create(nfeatures=500)
    kp_orb, desc_orb = orb.detectAndCompute(enhanced_frame, None)
    features_data['orb'] = (kp_orb, desc_orb)
    
    return features_data

def match_features_between_frames(features1, features2, method='sift'):
    """Match features between two frames"""
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
                if m.distance < 0.65 * n.distance:  # Slightly more strict
                    good_matches.append(m)
    
    elif method == 'orb':
        # Brute force matcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        good_matches = sorted(matches, key=lambda x: x.distance)[:50]  # Top 50 matches
    
    # Calculate similarity score
    similarity_score = min(100, (len(good_matches) / 30) * 100)
    
    return good_matches, similarity_score

def analyze_video_comparison():
    """Main function to compare ocelot videos"""
    # Video paths
    video1_path = "/Users/a/Documents/p r o y e c t o s/M A N A K A I/Camaras Trampa/selected camera trap videos/02062025_IMAG0033_adjaramillo.AVI"
    video2_path = "/Users/a/Documents/p r o y e c t o s/M A N A K A I/Camaras Trampa/2024 Octubre camara #2/DCIM/100MEDIA/IMAG0059.mp4"
    
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
    
    # Compare all frame combinations
    print("\n3. COMPARING PATTERNS ACROSS ALL FRAME COMBINATIONS...")
    
    comparison_results = {
        'sift_scores': [],
        'orb_scores': [],
        'best_matches': []
    }
    
    for i, feat1 in enumerate(features1_list):
        for j, feat2 in enumerate(features2_list):
            # SIFT matching
            sift_matches, sift_score = match_features_between_frames(feat1, feat2, 'sift')
            
            # ORB matching
            orb_matches, orb_score = match_features_between_frames(feat1, feat2, 'orb')
            
            comparison_results['sift_scores'].append(sift_score)
            comparison_results['orb_scores'].append(orb_score)
            
            # Store best matches for visualization
            if sift_score > 30 or orb_score > 30:  # Significant matches
                comparison_results['best_matches'].append({
                    'frame1_idx': i,
                    'frame2_idx': j,
                    'sift_score': sift_score,
                    'orb_score': orb_score,
                    'sift_matches': len(sift_matches),
                    'orb_matches': len(orb_matches)
                })
    
    # Statistical analysis
    print("\n4. STATISTICAL ANALYSIS...")
    
    sift_scores = comparison_results['sift_scores']
    orb_scores = comparison_results['orb_scores']
    
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
    
    # Determine overall assessment
    max_similarity = max(sift_stats['max'], orb_stats['max'])
    strong_matches = sift_stats['above_threshold'] + orb_stats['above_threshold']
    
    print(f"\n" + "="*60)
    print("INDIVIDUAL IDENTIFICATION ASSESSMENT")
    print("="*60)
    
    if max_similarity > 60 and strong_matches > 5:
        conclusion = "HIGH CONFIDENCE - SAME INDIVIDUAL"
        confidence = "85-95%"
    elif max_similarity > 40 and strong_matches > 3:
        conclusion = "MODERATE CONFIDENCE - LIKELY SAME INDIVIDUAL"
        confidence = "65-80%"
    elif max_similarity > 25 and strong_matches > 1:
        conclusion = "LOW-MODERATE CONFIDENCE - POSSIBLY SAME INDIVIDUAL"
        confidence = "45-65%"
    else:
        conclusion = "LOW CONFIDENCE - LIKELY DIFFERENT INDIVIDUALS"
        confidence = "20-45%"
    
    print(f"CONCLUSION: {conclusion}")
    print(f"CONFIDENCE LEVEL: {confidence}")
    print(f"MAX SIMILARITY SCORE: {max_similarity:.1f}%")
    print(f"STRONG PATTERN MATCHES: {strong_matches}")
    
    # Create visualization of best matches
    if comparison_results['best_matches']:
        print(f"\n5. CREATING VISUALIZATION...")
        create_best_matches_visualization(
            frames1, frames2, enhanced1_list, enhanced2_list, 
            comparison_results['best_matches'][:6]  # Top 6 matches
        )
    
    # Save detailed results
    results = {
        'video1': Path(video1_path).name,
        'video2': Path(video2_path).name,
        'conclusion': conclusion,
        'confidence': confidence,
        'max_similarity': max_similarity,
        'sift_stats': sift_stats,
        'orb_stats': orb_stats,
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

if __name__ == "__main__":
    analyze_video_comparison()