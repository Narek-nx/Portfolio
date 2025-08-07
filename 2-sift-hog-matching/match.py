import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import time

# --- ROI Selector using matplotlib ---
roi_coords = []

def line_select_callback(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    roi_coords.append((min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

def select_roi_with_matplotlib(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    ax.set_title("Select ROI (drag with mouse), then close the window")
    toggle_selector = RectangleSelector(ax, line_select_callback,
                                        drawtype='box', useblit=True,
                                        button=[1], minspanx=5, minspany=5,
                                        spancoords='pixels', interactive=True)
    plt.show()
    return roi_coords[0]

# --- Your feature computation functions here ---
def compute_hog_descriptor(img, keypoints, patch_size=16):
    hog_features = []
    half = patch_size // 2
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = img[max(0, y - half):y + half, max(0, x - half):x + half]
        print(patch.shape)
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            hog_features.append(np.zeros(36))
            continue
        feat = hog(patch, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(feat)
    return np.array(hog_features)

def compute_lbp_descriptor(img, keypoints, patch_size=15, radius=1, n_points=8):
    lbp_features = []
    half = patch_size // 2
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        patch = img[max(0, y - half):y + half, max(0, x - half):x + half]
        #print(patch.shape)
        if patch.shape[0] >= patch_size or patch.shape[1] >= patch_size:
            lbp_features.append(np.zeros(59))
            continue
        lbp = local_binary_pattern(patch, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(60), range=(0, 59))
        lbp_hist = lbp_hist.astype(np.float32)
        lbp_hist /= (lbp_hist.sum() + 1e-6)
        lbp_features.append(lbp_hist)
    return np.array(lbp_features)

# Load your image
img1 = cv2.imread('DSC01568.JPG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('DSC01568.JPG', cv2.IMREAD_GRAYSCALE)

# Select ROI via matplotlib
x, y, w, h = select_roi_with_matplotlib(img1)
# Hardcoded ROI for testing (x, y, width, height)
#x, y, w, h = 2700, 1000, 200, 200

roi_img = img1[y:y+h, x:x+w]

# Feature flags
use_hog = True
use_lbp = True

# ORB
orb = cv2.ORB_create(
    nfeatures=100000, scaleFactor=1.2, nlevels=12,
    edgeThreshold=16, firstLevel=0, WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE, patchSize=16, fastThreshold=20
)

# Keypoints and descriptors
kp1, des1 = orb.detectAndCompute(roi_img, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Compute HOG and LBP
hog1 = compute_hog_descriptor(roi_img, kp1) if use_hog else None
hog2 = compute_hog_descriptor(img2, kp2) if use_hog else None
lbp1 = compute_lbp_descriptor(roi_img, kp1) if use_lbp else None
lbp2 = compute_lbp_descriptor(img2, kp2) if use_lbp else None
print(f"hog1                                        {hog1}")
print(f"lbp1                                        {lbp1}")
# Feature concatenation
if des1 is not None:
    des1_aug = des1.astype(np.float32)
    if hog1 is not None: des1_aug = np.hstack((des1_aug, hog1.astype(np.float32)))
    if lbp1 is not None: des1_aug = np.hstack((des1_aug, lbp1.astype(np.float32)))

if des2 is not None:
    des2_aug = des2.astype(np.float32)
    if hog2 is not None: des2_aug = np.hstack((des2_aug, hog2.astype(np.float32)))
    if lbp2 is not None: des2_aug = np.hstack((des2_aug, lbp2.astype(np.float32)))

# Matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1_aug, des2_aug, k=2)

# Lowe's Ratio Test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# RANSAC Filtering
if len(good_matches) > 4:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    print(f"Found {len(inliers)} inlier matches after RANSAC.")
else:
    inliers = []
    print("Not enough matches for RANSAC.")

# Draw and show
matched_img = cv2.drawMatches(roi_img, kp1, img2, kp2, inliers, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(matched_img, cmap='gray')
plt.title("ORB + HOG + LBP Matching with RANSAC")
plt.axis('off')
plt.show()

