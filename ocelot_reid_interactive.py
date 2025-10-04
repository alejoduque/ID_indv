import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from skimage.feature import local_binary_pattern
import os
from datetime import datetime

class OcelotReID:
    """Ocelot Re-identification System using multiple pattern matching techniques"""
    
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=500)
        self.orb = cv2.ORB_create(nfeatures=500)
        self.matcher_flann = self._setup_flann_matcher()
        self.matcher_bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
    def _setup_flann_matcher(self):
        """Setup FLANN matcher for SIFT features"""
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        return cv2.FlannBasedMatcher(index_params, search_params)
    
    def load_and_preprocess(self, image_path):
        """Load and preprocess image for re-identification"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize for consistent processing (maintain aspect ratio)
        height, width = gray.shape
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            gray = cv2.resize(gray, (new_width, new_height))
            img = cv2.resize(img, (new_width, new_height))
        
        return img, gray
    
    def enhance_patterns(self, gray_img):
        """Enhanced pattern extraction for ocelot spots"""
        results = {}
        
        # 1. CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        results['enhanced'] = clahe.apply(gray_img)
        
        # 2. Multi-scale edge detection
        edges_fine = cv2.Canny(results['enhanced'], 30, 100)
        edges_coarse = cv2.Canny(results['enhanced'], 50, 150)
        results['edges'] = cv2.addWeighted(edges_fine, 0.5, edges_coarse, 0.5, 0)
        
        # 3. Morphological operations to enhance spot patterns
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        results['morph'] = cv2.morphologyEx(results['enhanced'], cv2.MORPH_GRADIENT, kernel)
        
        # 4. Local Binary Pattern for texture
        radius = 3
        n_points = 8 * radius
        results['lbp'] = local_binary_pattern(results['enhanced'], n_points, radius, method='uniform')
        results['lbp'] = (results['lbp'] / results['lbp'].max() * 255).astype(np.uint8)
        
        return results
    
    def extract_features(self, img_dict):
        """Extract multiple types of features for robust matching"""
        features = {}
        
        # SIFT features on enhanced image
        kp_sift, desc_sift = self.sift.detectAndCompute(img_dict['enhanced'], None)
        features['sift'] = {'kp': kp_sift, 'desc': desc_sift}
        
        # ORB features for faster matching
        kp_orb, desc_orb = self.orb.detectAndCompute(img_dict['enhanced'], None)
        features['orb'] = {'kp': kp_orb, 'desc': desc_orb}
        
        # Global histogram features
        hist_enhanced = cv2.calcHist([img_dict['enhanced']], [0], None, [256], [0,256])
        hist_lbp = cv2.calcHist([img_dict['lbp']], [0], None, [256], [0,256])
        features['hist'] = {'enhanced': hist_enhanced.flatten(), 'lbp': hist_lbp.flatten()}
        
        # Spot pattern statistics
        features['pattern_stats'] = self._analyze_spot_patterns(img_dict['morph'])
        
        return features
    
    def _analyze_spot_patterns(self, morph_img):
        """Analyze spot patterns for additional features"""
        # Find contours (spots)
        contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        min_area = 20
        spots = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Calculate statistics
        if len(spots) > 0:
            areas = [cv2.contourArea(c) for c in spots]
            perimeters = [cv2.arcLength(c, True) for c in spots]
            
            stats = {
                'num_spots': len(spots),
                'mean_area': np.mean(areas),
                'std_area': np.std(areas),
                'mean_perimeter': np.mean(perimeters),
                'total_spot_area': np.sum(areas),
                'spot_density': len(spots) / (morph_img.shape[0] * morph_img.shape[1])
            }
        else:
            stats = {
                'num_spots': 0,
                'mean_area': 0,
                'std_area': 0,
                'mean_perimeter': 0,
                'total_spot_area': 0,
                'spot_density': 0
            }
        
        return stats
    
    def match_features(self, features1, features2):
        """Match features between two images"""
        scores = {}
        
        # 1. SIFT matching
        if features1['sift']['desc'] is not None and features2['sift']['desc'] is not None:
            if len(features1['sift']['desc']) > 10 and len(features2['sift']['desc']) > 10:
                matches_sift = self.matcher_flann.knnMatch(
                    features1['sift']['desc'], 
                    features2['sift']['desc'], 
                    k=2
                )
                
                # Lowe's ratio test
                good_sift = []
                for match_pair in matches_sift:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:
                            good_sift.append(m)
                
                scores['sift_matches'] = len(good_sift)
                scores['sift_score'] = min(100, (len(good_sift) / 30) * 100)
            else:
                scores['sift_matches'] = 0
                scores['sift_score'] = 0
        else:
            scores['sift_matches'] = 0
            scores['sift_score'] = 0
        
        # 2. ORB matching
        if features1['orb']['desc'] is not None and features2['orb']['desc'] is not None:
            matches_orb = self.matcher_bf.match(features1['orb']['desc'], features2['orb']['desc'])
            scores['orb_matches'] = len(matches_orb)
            scores['orb_score'] = min(100, (len(matches_orb) / 50) * 100)
        else:
            scores['orb_matches'] = 0
            scores['orb_score'] = 0
        
        # 3. Histogram correlation
        hist_corr_enhanced = cv2.compareHist(
            features1['hist']['enhanced'], 
            features2['hist']['enhanced'],
            cv2.HISTCMP_CORREL
        )
        hist_corr_lbp = cv2.compareHist(
            features1['hist']['lbp'],
            features2['hist']['lbp'],
            cv2.HISTCMP_CORREL
        )
        scores['hist_score'] = ((hist_corr_enhanced + hist_corr_lbp) / 2) * 100
        
        # 4. Pattern statistics similarity
        stats1 = features1['pattern_stats']
        stats2 = features2['pattern_stats']
        
        pattern_similarities = []
        if stats1['num_spots'] > 0 and stats2['num_spots'] > 0:
            # Normalize differences
            spot_diff = 1 - abs(stats1['num_spots'] - stats2['num_spots']) / max(stats1['num_spots'], stats2['num_spots'])
            area_diff = 1 - abs(stats1['mean_area'] - stats2['mean_area']) / max(stats1['mean_area'], stats2['mean_area'], 1)
            density_diff = 1 - abs(stats1['spot_density'] - stats2['spot_density']) / max(stats1['spot_density'], stats2['spot_density'], 0.001)
            
            pattern_similarities = [spot_diff, area_diff, density_diff]
            scores['pattern_score'] = np.mean(pattern_similarities) * 100
        else:
            scores['pattern_score'] = 0
        
        return scores
    
    def calculate_reid_score(self, scores):
        """Calculate weighted re-identification score"""
        # Weight different matching methods
        weights = {
            'sift': 0.35,
            'orb': 0.20,
            'hist': 0.25,
            'pattern': 0.20
        }
        
        weighted_score = (
            scores['sift_score'] * weights['sift'] +
            scores['orb_score'] * weights['orb'] +
            scores['hist_score'] * weights['hist'] +
            scores['pattern_score'] * weights['pattern']
        )
        
        return weighted_score
    
    def identify_individual(self, image_path, database_paths):
        """Identify an individual against a database of known ocelots"""
        print(f"\n{'='*60}")
        print(f"Re-identifying: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Process query image
        query_img, query_gray = self.load_and_preprocess(image_path)
        query_enhanced = self.enhance_patterns(query_gray)
        query_features = self.extract_features(query_enhanced)
        
        # Compare against database
        results = []
        for db_path in database_paths:
            if db_path == image_path:  # Skip self-comparison
                continue
                
            try:
                db_img, db_gray = self.load_and_preprocess(db_path)
                db_enhanced = self.enhance_patterns(db_gray)
                db_features = self.extract_features(db_enhanced)
                
                scores = self.match_features(query_features, db_features)
                reid_score = self.calculate_reid_score(scores)
                
                results.append({
                    'path': db_path,
                    'name': os.path.basename(db_path),
                    'score': reid_score,
                    'details': scores
                })
                
            except Exception as e:
                print(f"Error processing {db_path}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Print results
        print("\nTop Matches:")
        print("-" * 60)
        for i, result in enumerate(results[:5], 1):
            confidence = self._get_confidence_level(result['score'])
            print(f"{i}. {result['name']}")
            print(f"   Overall Score: {result['score']:.1f}% - {confidence}")
            print(f"   SIFT matches: {result['details']['sift_matches']}")
            print(f"   ORB matches: {result['details']['orb_matches']}")
            print(f"   Histogram similarity: {result['details']['hist_score']:.1f}%")
            print(f"   Pattern similarity: {result['details']['pattern_score']:.1f}%")
            print()
        
        return results
    
    def _get_confidence_level(self, score):
        """Convert score to confidence level"""
        if score >= 75:
            return "VERY HIGH - Same individual"
        elif score >= 60:
            return "HIGH - Likely same individual"
        elif score >= 45:
            return "MODERATE - Possibly same individual"
        elif score >= 30:
            return "LOW - Unlikely same individual"
        else:
            return "VERY LOW - Different individual"
    
    def compare_pair(self, image_path1, image_path2, save_visualization=True):
        """Detailed comparison of two ocelot images"""
        print(f"\n{'='*60}")
        print("Pairwise Re-identification Analysis")
        print(f"{'='*60}")
        print(f"Image 1: {os.path.basename(image_path1)}")
        print(f"Image 2: {os.path.basename(image_path2)}")
        print("-" * 60)
        
        # Process both images
        img1, gray1 = self.load_and_preprocess(image_path1)
        img2, gray2 = self.load_and_preprocess(image_path2)
        
        enhanced1 = self.enhance_patterns(gray1)
        enhanced2 = self.enhance_patterns(gray2)
        
        features1 = self.extract_features(enhanced1)
        features2 = self.extract_features(enhanced2)
        
        # Calculate scores
        scores = self.match_features(features1, features2)
        reid_score = self.calculate_reid_score(scores)
        
        # Print detailed results
        print(f"\n📊 RE-IDENTIFICATION RESULTS:")
        print(f"Overall Score: {reid_score:.1f}%")
        print(f"Confidence: {self._get_confidence_level(reid_score)}")
        print(f"\n📈 Detailed Scores:")
        print(f"  • SIFT Feature Matches: {scores['sift_matches']} (Score: {scores['sift_score']:.1f}%)")
        print(f"  • ORB Feature Matches: {scores['orb_matches']} (Score: {scores['orb_score']:.1f}%)")
        print(f"  • Histogram Correlation: {scores['hist_score']:.1f}%")
        print(f"  • Pattern Statistics: {scores['pattern_score']:.1f}%")
        
        if save_visualization:
            self._create_visualization(
                img1, img2, 
                enhanced1, enhanced2, 
                features1, features2, 
                scores, reid_score
            )
            print(f"\n📁 Visualization saved to: ocelot_reid_analysis.png")
        
        return reid_score, scores
    
    def _create_visualization(self, img1, img2, enhanced1, enhanced2, 
                            features1, features2, scores, reid_score):
        """Create comprehensive visualization of re-identification analysis"""
        fig = plt.figure(figsize=(20, 12))
        
        # Main title with score
        confidence = self._get_confidence_level(reid_score)
        fig.suptitle(
            f'Ocelot Re-Identification Analysis\nOverall Score: {reid_score:.1f}% - {confidence}',
            fontsize=16, fontweight='bold'
        )
        
        # Create grid
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Original images
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        ax1.set_title('Image 1 (Original)')
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        ax2.set_title('Image 2 (Original)')
        ax2.axis('off')
        
        # Row 1: Enhanced patterns
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(enhanced1['enhanced'], cmap='gray')
        ax3.set_title('Enhanced Pattern 1')
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(enhanced2['enhanced'], cmap='gray')
        ax4.set_title('Enhanced Pattern 2')
        ax4.axis('off')
        
        # Row 2: Edge and morph
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(enhanced1['edges'], cmap='gray')
        ax5.set_title('Edge Detection 1')
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(enhanced2['edges'], cmap='gray')
        ax6.set_title('Edge Detection 2')
        ax6.axis('off')
        
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.imshow(enhanced1['morph'], cmap='gray')
        ax7.set_title('Spot Patterns 1')
        ax7.axis('off')
        
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.imshow(enhanced2['morph'], cmap='gray')
        ax8.set_title('Spot Patterns 2')
        ax8.axis('off')
        
        # Row 3: LBP and scores
        ax9 = fig.add_subplot(gs[2, 0])
        ax9.imshow(enhanced1['lbp'], cmap='gray')
        ax9.set_title('Texture (LBP) 1')
        ax9.axis('off')
        
        ax10 = fig.add_subplot(gs[2, 1])
        ax10.imshow(enhanced2['lbp'], cmap='gray')
        ax10.set_title('Texture (LBP) 2')
        ax10.axis('off')
        
        # Score visualization
        ax11 = fig.add_subplot(gs[2, 2:])
        categories = ['SIFT\nFeatures', 'ORB\nFeatures', 'Histogram\nSimilarity', 'Pattern\nStatistics']
        values = [scores['sift_score'], scores['orb_score'], scores['hist_score'], scores['pattern_score']]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c']
        
        bars = ax11.bar(categories, values, color=colors, alpha=0.8)
        ax11.set_ylabel('Similarity Score (%)', fontsize=12)
        ax11.set_title('Component Scores', fontsize=12, fontweight='bold')
        ax11.set_ylim(0, 100)
        ax11.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax11.text(bar.get_x() + bar.get_width()/2., height,
                     f'{value:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Add threshold line
        ax11.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='High Confidence Threshold')
        ax11.legend()
        
        plt.tight_layout()
        plt.savefig('ocelot_reid_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def extract_video_frames(self, video_path, num_frames=10):
        """Extract evenly distributed frames from video"""
        print(f"📹 Extracting frames from: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"   Total frames: {total_frames}, Duration: {duration:.1f}s, FPS: {fps:.1f}")
        
        # Calculate frame indices to extract
        if total_frames < num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        print(f"   Extracted {len(frames)} frames")
        return frames
    
    def compare_videos(self, video_path1, video_path2, num_frames=10, save_visualization=True):
        """Compare two videos by analyzing multiple frames"""
        print(f"\n{'='*60}")
        print("Video Re-identification Analysis")
        print(f"{'='*60}")
        print(f"Video 1: {os.path.basename(video_path1)}")
        print(f"Video 2: {os.path.basename(video_path2)}")
        print("-" * 60)
        
        # Extract frames
        frames1 = self.extract_video_frames(video_path1, num_frames)
        frames2 = self.extract_video_frames(video_path2, num_frames)
        
        if not frames1 or not frames2:
            raise ValueError("Could not extract frames from one or both videos")
        
        print(f"\n🔍 Analyzing frame pairs...")
        
        # Compare all frame pairs
        all_scores = []
        best_score = 0
        best_pair = (None, None)
        best_pair_indices = (0, 0)
        
        for i, frame1 in enumerate(frames1):
            for j, frame2 in enumerate(frames2):
                # Save frames temporarily
                temp_path1 = f"temp_frame1_{i}.jpg"
                temp_path2 = f"temp_frame2_{j}.jpg"
                cv2.imwrite(temp_path1, frame1)
                cv2.imwrite(temp_path2, frame2)
                
                try:
                    # Process frames
                    img1, gray1 = self.load_and_preprocess(temp_path1)
                    img2, gray2 = self.load_and_preprocess(temp_path2)
                    
                    enhanced1 = self.enhance_patterns(gray1)
                    enhanced2 = self.enhance_patterns(gray2)
                    
                    features1 = self.extract_features(enhanced1)
                    features2 = self.extract_features(enhanced2)
                    
                    scores = self.match_features(features1, features2)
                    reid_score = self.calculate_reid_score(scores)
                    
                    all_scores.append(reid_score)
                    
                    if reid_score > best_score:
                        best_score = reid_score
                        best_pair = (frame1.copy(), frame2.copy())
                        best_pair_indices = (i, j)
                    
                except Exception as e:
                    print(f"   Error processing frame pair ({i}, {j}): {e}")
                    continue
                finally:
                    # Clean up temp files
                    if os.path.exists(temp_path1):
                        os.remove(temp_path1)
                    if os.path.exists(temp_path2):
                        os.remove(temp_path2)
        
        # Calculate statistics
        mean_score = np.mean(all_scores) if all_scores else 0
        std_score = np.std(all_scores) if all_scores else 0
        median_score = np.median(all_scores) if all_scores else 0
        
        # Print results
        print(f"\n📊 VIDEO RE-IDENTIFICATION RESULTS:")
        print(f"Frames analyzed: {len(frames1)} x {len(frames2)} = {len(all_scores)} comparisons")
        print(f"\nScore Statistics:")
        print(f"  • Best Match Score: {best_score:.1f}% - {self._get_confidence_level(best_score)}")
        print(f"  • Mean Score: {mean_score:.1f}%")
        print(f"  • Median Score: {median_score:.1f}%")
        print(f"  • Std Deviation: {std_score:.1f}%")
        print(f"\nBest matching frames: Video1[{best_pair_indices[0]}] - Video2[{best_pair_indices[1]}]")
        
        if save_visualization and best_pair[0] is not None:
            self._create_video_visualization(
                frames1, frames2, all_scores, best_pair, best_pair_indices, 
                mean_score, best_score
            )
            print(f"\n📁 Visualizations saved:")
            print(f"   - video_reid_analysis.png")
            print(f"   - best_match_frames.png")
        
        return mean_score, best_score, all_scores
    
    def _create_video_visualization(self, frames1, frames2, scores, best_pair, 
                                   best_indices, mean_score, best_score):
        """Create visualization for video comparison"""
        # Create main visualization with frame grid
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(
            f'Video Re-Identification Analysis\n'
            f'Best Score: {best_score:.1f}% | Mean Score: {mean_score:.1f}%',
            fontsize=16, fontweight='bold'
        )
        
        # Calculate grid size
        n_frames1 = min(5, len(frames1))
        n_frames2 = min(5, len(frames2))
        
        # Show sample frames from each video
        for i in range(n_frames1):
            ax = plt.subplot(3, 5, i + 1)
            frame_idx = int(i * len(frames1) / n_frames1)
            ax.imshow(cv2.cvtColor(frames1[frame_idx], cv2.COLOR_BGR2RGB))
            ax.set_title(f'Video 1 - Frame {frame_idx}')
            ax.axis('off')
        
        for i in range(n_frames2):
            ax = plt.subplot(3, 5, i + 6)
            frame_idx = int(i * len(frames2) / n_frames2)
            ax.imshow(cv2.cvtColor(frames2[frame_idx], cv2.COLOR_BGR2RGB))
            ax.set_title(f'Video 2 - Frame {frame_idx}')
            ax.axis('off')
        
        # Score distribution
        ax_scores = plt.subplot(3, 5, 11)
        ax_scores.hist(scores, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
        ax_scores.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}%')
        ax_scores.axvline(best_score, color='green', linestyle='--', linewidth=2, label=f'Best: {best_score:.1f}%')
        ax_scores.set_xlabel('Similarity Score (%)')
        ax_scores.set_ylabel('Frequency')
        ax_scores.set_title('Score Distribution')
        ax_scores.legend()
        ax_scores.grid(alpha=0.3)
        
        # Score heatmap
        ax_heat = plt.subplot(3, 5, (12, 15))
        score_matrix = np.array(scores).reshape(len(frames1), len(frames2))
        im = ax_heat.imshow(score_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
        ax_heat.set_xlabel('Video 2 Frames')
        ax_heat.set_ylabel('Video 1 Frames')
        ax_heat.set_title('Frame-by-Frame Similarity Heatmap')
        plt.colorbar(im, ax=ax_heat, label='Similarity Score (%)')
        
        # Mark best match
        ax_heat.plot(best_indices[1], best_indices[0], 'r*', markersize=20, 
                    markeredgecolor='white', markeredgewidth=2)
        
        plt.tight_layout()
        plt.savefig('video_reid_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate visualization for best matching frames
        fig2 = plt.figure(figsize=(16, 8))
        fig2.suptitle(
            f'Best Matching Frames\n'
            f'Video 1 Frame {best_indices[0]} vs Video 2 Frame {best_indices[1]} | '
            f'Score: {best_score:.1f}% - {self._get_confidence_level(best_score)}',
            fontsize=14, fontweight='bold'
        )
        
        ax1 = plt.subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(best_pair[0], cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Video 1 - Frame {best_indices[0]}')
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 2, 2)
        ax2.imshow(cv2.cvtColor(best_pair[1], cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Video 2 - Frame {best_indices[1]}')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('best_match_frames.png', dpi=300, bbox_inches='tight')
        plt.close()


def interactive_menu():
    """Interactive command-line interface"""
    reid_system = OcelotReID()
    
    print("\n" + "="*60)
    print("🐆 OCELOT RE-IDENTIFICATION SYSTEM 🐆")
    print("="*60)
    print("\nThis system compares ocelot images or videos based on")
    print("their unique spot patterns to identify individuals.")
    print("="*60)
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Compare Two Images")
        print("2. Compare Two Videos")
        print("3. Identify Against Database (Images)")
        print("4. Help & Information")
        print("5. Exit")
        print("-"*60)
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            compare_images_interactive(reid_system)
        elif choice == '2':
            compare_videos_interactive(reid_system)
        elif choice == '3':
            database_search_interactive(reid_system)
        elif choice == '4':
            show_help()
        elif choice == '5':
            print("\n👋 Thank you for using Ocelot Re-ID System!")
            print("="*60)
            break
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def compare_images_interactive(reid_system):
    """Interactive image comparison"""
    print("\n" + "="*60)
    print("IMAGE COMPARISON MODE")
    print("="*60)
    
    while True:
        path1 = input("\nEnter path to first image (or 'back' to return): ").strip()
        if path1.lower() == 'back':
            return
        
        if not os.path.exists(path1):
            print(f"❌ Error: File not found: {path1}")
            continue
        
        path2 = input("Enter path to second image: ").strip()
        if not os.path.exists(path2):
            print(f"❌ Error: File not found: {path2}")
            continue
        
        save_viz = input("\nSave visualization? (y/n, default=y): ").strip().lower()
        save_viz = save_viz != 'n'
        
        try:
            print("\n🔄 Processing images...")
            score, details = reid_system.compare_pair(path1, path2, save_visualization=save_viz)
            
            print("\n✅ Analysis complete!")
            
            another = input("\nCompare another pair? (y/n): ").strip().lower()
            if another != 'y':
                break
                
        except Exception as e:
            print(f"\n❌ Error during comparison: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                break


def compare_videos_interactive(reid_system):
    """Interactive video comparison"""
    print("\n" + "="*60)
    print("VIDEO COMPARISON MODE")
    print("="*60)
    print("\nNote: Video processing may take several minutes")
    print("depending on video length and frame rate.")
    
    while True:
        path1 = input("\nEnter path to first video (or 'back' to return): ").strip()
        if path1.lower() == 'back':
            return
        
        if not os.path.exists(path1):
            print(f"❌ Error: File not found: {path1}")
            continue
        
        path2 = input("Enter path to second video: ").strip()
        if not os.path.exists(path2):
            print(f"❌ Error: File not found: {path2}")
            continue
        
        num_frames = input("\nNumber of frames to extract per video (default=10): ").strip()
        num_frames = int(num_frames) if num_frames.isdigit() else 10
        
        save_viz = input("Save visualization? (y/n, default=y): ").strip().lower()
        save_viz = save_viz != 'n'
        
        try:
            print("\n🔄 Processing videos (this may take a while)...")
            mean_score, best_score, all_scores = reid_system.compare_videos(
                path1, path2, num_frames=num_frames, save_visualization=save_viz
            )
            
            print("\n✅ Analysis complete!")
            
            another = input("\nCompare another pair? (y/n): ").strip().lower()
            if another != 'y':
                break
                
        except Exception as e:
            print(f"\n❌ Error during comparison: {e}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                break


def database_search_interactive(reid_system):
    """Interactive database search"""
    print("\n" + "="*60)
    print("DATABASE SEARCH MODE")
    print("="*60)
    print("\nThis mode compares one query image against multiple")
    print("database images to find the best match.")
    
    query_path = input("\nEnter path to query image (or 'back' to return): ").strip()
    if query_path.lower() == 'back':
        return
    
    if not os.path.exists(query_path):
        print(f"❌ Error: File not found: {query_path}")
        return
    
    print("\nEnter database image paths (one per line).")
    print("Enter 'done' when finished, or 'folder' to add all images from a folder:")
    
    database_paths = []
    while True:
        path = input("> ").strip()
        
        if path.lower() == 'done':
            break
        elif path.lower() == 'folder':
            folder = input("Enter folder path: ").strip()
            if os.path.isdir(folder):
                extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                for file in os.listdir(folder):
                    if any(file.lower().endswith(ext) for ext in extensions):
                        database_paths.append(os.path.join(folder, file))
                print(f"Added {len([f for f in os.listdir(folder) if any(f.lower().endswith(ext) for ext in extensions)])} images from folder")
            else:
                print(f"❌ Folder not found: {folder}")
        elif os.path.exists(path):
            database_paths.append(path)
            print(f"✓ Added: {os.path.basename(path)}")
        else:
            print(f"❌ File not found: {path}")
    
    if not database_paths:
        print("❌ No database images provided.")
        return
    
    try:
        print(f"\n🔄 Searching against {len(database_paths)} database images...")
        results = reid_system.identify_individual(query_path, database_paths)
        print("\n✅ Search complete!")
        
    except Exception as e:
        print(f"\n❌ Error during search: {e}")


def show_help():
    """Display help information"""
    print("\n" + "="*60)
    print("HELP & INFORMATION")
    print("="*60)
    
    help_text = """
📖 HOW IT WORKS:
The system analyzes unique spot patterns on ocelots using:
- SIFT & ORB feature detection
- Histogram comparison
- Pattern statistics (spot count, size, distribution)

🎯 CONFIDENCE LEVELS:
- 75-100%: VERY HIGH - Same individual
- 60-74%:  HIGH - Likely same individual  
- 45-59%:  MODERATE - Possibly same individual
- 30-44%:  LOW - Unlikely same individual
- 0-29%:   VERY LOW - Different individual

💡 TIPS FOR BEST RESULTS:
✓ Use clear, well-lit images/videos
✓ Focus on body/leg areas with visible spots
✓ Minimum resolution: 800x600 pixels
✓ Similar angles/perspectives work best
✗ Avoid blurry or dark media
✗ Avoid extreme angles

📁 SUPPORTED FORMATS:
Images: JPG, JPEG, PNG, BMP
Videos: MP4, AVI, MOV (any OpenCV-supported format)

📊 OUTPUT FILES:
- ocelot_reid_analysis.png (image comparison)
- video_reid_analysis.png (video frame grid & heatmap)
- best_match_frames.png (best matching video frames)

⚠️  VIDEO PROCESSING:
- Extracts 10 frames by default (adjustable)
- Processing time: ~2-3 seconds per frame pair
- Example: 10 frames each = 100 comparisons ≈ 3-5 minutes

🔧 REQUIREMENTS:
- Python 3.7+
- OpenCV, NumPy, Matplotlib, SciPy, scikit-image

For detailed documentation, see the User Guide.
"""
    
    print(help_text)
    input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted by user.")
        print("👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your inputs and try again.")


if __name__ == "__main__":
    main()