import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    """Load image and convert to grayscale for processing"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def enhance_patterns(gray_img):
    """Enhance pattern visibility using various techniques"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray_img)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Edge detection to highlight patterns
    edges = cv2.Canny(blurred, 50, 150)
    
    return enhanced, edges

def extract_key_features(gray_img):
    """Extract key features using SIFT detector"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors

def compare_patterns():
    """Main function to compare ocelot patterns"""
    # Load images
    img1, gray1 = load_and_preprocess_image('Ocelote_compare_1.jpg')
    img2, gray2 = load_and_preprocess_image('Ocelote_compare_2.jpg')
    
    # Enhance patterns
    enhanced1, edges1 = enhance_patterns(gray1)
    enhanced2, edges2 = enhance_patterns(gray2)
    
    # Extract features
    kp1, desc1 = extract_key_features(enhanced1)
    kp2, desc2 = extract_key_features(enhanced2)
    
    # Match features using FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    if desc1 is not None and desc2 is not None and len(desc1) > 10 and len(desc2) > 10:
        matches = flann.knnMatch(desc1, desc2, k=2)
        
        # Apply Lowe's ratio test to find good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
    else:
        good_matches = []
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ocelot Pattern Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Row 1: Original images and enhanced versions
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Ocelote_compare_1 (Original)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(enhanced1, cmap='gray')
    axes[0, 1].set_title('Enhanced Pattern 1')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(edges1, cmap='gray')
    axes[0, 2].set_title('Edge Detection 1')
    axes[0, 2].axis('off')
    
    # Row 2: Second image versions
    axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Ocelote_compare_2 (Original)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(enhanced2, cmap='gray')
    axes[1, 1].set_title('Enhanced Pattern 2')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(edges2, cmap='gray')
    axes[1, 2].set_title('Edge Detection 2')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('ocelot_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature matching visualization
    if len(good_matches) > 10:
        # Draw matches
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'Feature Matches Found: {len(good_matches)} (Showing top 50)')
        plt.axis('off')
        plt.savefig('ocelot_feature_matches.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        similarity_score = min(100, (len(good_matches) / 50) * 100)
        print(f"Pattern similarity analysis complete!")
        print(f"Good feature matches found: {len(good_matches)}")
        print(f"Estimated similarity score: {similarity_score:.1f}%")
        
        if similarity_score > 70:
            print("HIGH similarity - Likely the SAME individual")
        elif similarity_score > 40:
            print("MODERATE similarity - Possibly the same individual")
        else:
            print("LOW similarity - Likely DIFFERENT individuals")
    else:
        print("Insufficient feature matches found for reliable comparison")
    
    # Create side-by-side enhanced comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Enhanced Pattern Comparison for Individual Identification', fontsize=14, fontweight='bold')
    
    ax1.imshow(enhanced1, cmap='gray')
    ax1.set_title('Ocelote_compare_1 (Enhanced)')
    ax1.axis('off')
    
    ax2.imshow(enhanced2, cmap='gray')
    ax2.set_title('Ocelote_compare_2 (Enhanced)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('ocelot_enhanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Analysis images saved:")
    print("- ocelot_pattern_analysis.png: Detailed pattern analysis")
    print("- ocelot_enhanced_comparison.png: Side-by-side enhanced comparison")
    if len(good_matches) > 10:
        print("- ocelot_feature_matches.png: Feature matching visualization")

if __name__ == "__main__":
    compare_patterns()