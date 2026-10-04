#!/usr/bin/env python

##  hw5_skeleton.py

__version__   = '1.0.0'
__author__    = "Aditya Abhang ([your login]@purdue.edu)"
__date__      = '2026-September-23'


__doc__ = '''

    hw5_skeleton.py

    This is the skeleton for Homework 5 of ECE 661, Computer Vision, Fall 2026.

    Rename this file to hw5_<FirstName><LastName>.py before you submit it.

    The thread running through every task is measurement.  Three of your
    images are views of one flat mural, so the map between any two of them is
    exactly a homography and you are given that homography.  Every match your
    code produces there therefore has a right answer.

        Task 2    Harris corner detection, then NCC and SSD correspondence,
                  all of it your own code
        Task 3    OpenCV SIFT, run as provided, scored the same way
        Task 4    SuperPoint, run as provided through the supplied wrapper
        Task 5    the repetitive facade, and your own photographs
        Bonus     orientation-normalised windows

    What you may and may not use:

        Task 2    your own code.  numpy, scipy.ndimage, scipy.signal and a
                  generic correlation or convolution routine are all fine.
                  An off-the-shelf corner detector or matcher is not: no
                  cv2.cornerHarris, no cv2.goodFeaturesToTrack, no
                  skimage.feature.corner_harris, no cv2.matchTemplate.
        Task 3    the opposite.  Use OpenCV's SIFT and its matcher.  The
                  point of that task is the comparison, not a
                  reimplementation.
        Task 4    the provided SuperPoint and SuperGlue wrapper, as it is.

    Vectorise.  A doubly nested Python loop over pixels takes minutes where a
    numpy expression takes under a second, and the correspondence search in
    particular is one matrix product once you have set it up properly.

    evaluate() below is GIVEN to you, already written.  It is the measuring
    instrument rather than part of the exercise, and everyone using the same
    one is what makes the numbers in your report mean the same thing as the
    numbers in everybody else's.  Do not modify it.

    Note that the work is all inside functions and that the calls at the
    bottom sit behind the usual __main__ guard.  Keep it that way.  A grader
    may import your file rather than run it, and a file that reads an image
    or opens a window while it is being imported will fail on a machine that
    has neither a display nor the data.

    Fill in every place marked TODO, and delete the NotImplementedError when
    you do.

    Usage:      python hw5_skeleton.py
    Requires:   numpy, scipy, opencv-python, imageio   (versions in the handout)

    Course policy on AI assistants applies to this file.  Your report must
    open with an "AI Usage: XX%" line, and any function here that was
    generated entirely by an AI assistant must say so in its own docstring.
'''

import os
import numpy as np
import cv2
import imageio.v3 as iio
import scipy.ndimage as ndimage
from scipy.spatial import cKDTree


##  --------------------------------------------------------------------------
##  Paths.  These live here, at the top, not buried inside a function:  a hard
##  coded C:\Users\... path cannot be run by anyone else, your grader included.
##  --------------------------------------------------------------------------

DATA_DIR     = "."
OUT_DIR      = "output"

PLANAR       = {"a": "planar_a.jpg", "b": "planar_b.jpg", "c": "planar_c.jpg"}
PURDUE       = ["purdue_1.jpg", "purdue_2.jpg"]
REPEAT       = ["repeat_1.jpg", "repeat_2.jpg"]
TRUTH_PLANAR = "truth_hw5_planar.npz"
TRUTH_REPEAT = "truth_hw5_repeat.npz"


def load_image(name):
    """Read an image from DATA_DIR as an (H, W, 3) uint8 RGB array."""
    return iio.imread(os.path.join(DATA_DIR, name))


def load_truth(name):
    """Load one of the two .npz files of known homographies."""
    return np.load(os.path.join(DATA_DIR, name))


def save_figure(fig, name):
    """Write a figure into OUT_DIR, creating it on first use."""
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")


##  ============================================================================
##  Given helpers.  evaluate() in particular is not yours to change.
##  ============================================================================

def to_gray(img):
    """Luminance, float64 in [0, 255]."""
    if img.ndim == 2:
        return img.astype(np.float64)
    return (0.2989 * img[:, :, 0]
            + 0.5870 * img[:, :, 1]
            + 0.1140 * img[:, :, 2]).astype(np.float64)


def apply_H(H, pts):
    """pts: (N,2) -> (N,2), through the 3x3 homography H."""
    pts = np.asarray(pts, float)
    q = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


def evaluate(p1, p2, H21, H12, shape2, shape1, thresh=3.0, margin=0.0):
    """GIVEN -- do not modify.  Scores putative matches against the known
    homography of the controlled planar set.

    A match (p1, p2) is correct if p2 is where H21 says it should be, to
    within `thresh` pixels of symmetric transfer error.

    Only 'evaluable' matches count towards the inlier rate: those whose true
    correspondence actually lands inside the other image.  The three views do
    not overlap completely, so a corner with no counterpart in the other view
    has no correct answer available and it would be meaningless to score it.
    """
    out = dict(n_matches=int(len(p1)), n_evaluable=0, n_inliers=0,
               inlier_rate=float("nan"), mean_err=float("nan"),
               median_err=float("nan"))
    if len(p1) == 0:
        return out
    h2, w2 = shape2[:2]
    true2 = apply_H(H21, p1)
    true1 = apply_H(H12, p2)
    ev = ((true2[:, 0] >= margin) & (true2[:, 0] < w2 - margin)
          & (true2[:, 1] >= margin) & (true2[:, 1] < h2 - margin))
    out["n_evaluable"] = int(ev.sum())
    if ev.sum() == 0:
        return out
    err = 0.5 * (np.linalg.norm(true2 - p2, axis=1)
                 + np.linalg.norm(true1 - p1, axis=1))
    ee = err[ev]
    inl = ee < thresh
    out["n_inliers"] = int(inl.sum())
    out["inlier_rate"] = float(inl.mean())
    if inl.any():
        out["mean_err"] = float(ee[inl].mean())
        out["median_err"] = float(np.median(ee[inl]))
    return out


##  ============================================================================
##  Task 2: Harris corner detection, with NCC and SSD
##  ============================================================================

def haar_filters(sigma):
    """The x- and y-oriented Haar wavelet filters at scale sigma.

    Returns:
        hx, hy: two 2-D float arrays, the x- and the y-oriented filter
    """
    M = 2 * (int(np.floor(2.0 * sigma)) + 1)
    half = M // 2
    hx = np.ones((M, M))
    hx[:, :half] = -1.0
    hy = hx.T.copy()
    return hx, hy


def harris_corners(gray, sigma=1.2, k_win=5.0, max_corners=600,
                   nms_radius=6, ratio_thresh=0.10):
    """Harris corners at the single scale sigma.

    Args:
        gray: (H, W) float array
    Returns:
        (N, 2) array of (x, y) = (column, row), strongest first
    """
    ##  TODO, following Lecture 9, pages 9-11 and 9-12:
    ##    1. dx, dy from the Haar filters at scale sigma.
    ##    2. Accumulate C = [[sum dx^2, sum dx dy], [sum dx dy, sum dy^2]]
    ##       over a square neighbourhood roughly k_win*sigma across.
    ##    3. Score each pixel. The report asks you to justify using
    ##       det(C) / trace(C)^2 rather than computing the two eigenvalues.
    ##       Work out what that ratio equals in terms of lambda2/lambda1
    ##       first, and threshold on it accordingly.
    ##    4. Non-maximum suppression over a (2*nms_radius+1) square, so that
    ##       one corner does not come back as a cluster of forty.
    ##    5. Discard responses within the filter's own support of the border,
    ##       then keep the strongest max_corners.

    hx, hy = haar_filters(sigma)
    M = hx.shape[0]

    dx = ndimage.correlate(gray, hx, mode="reflect")
    dy = ndimage.correlate(gray, hy, mode="reflect")

    w = 2 * int(round(k_win * sigma / 2.0)) + 1
    sxx = ndimage.uniform_filter(dx * dx, size=w, mode="reflect")
    syy = ndimage.uniform_filter(dy * dy, size=w, mode="reflect")
    sxy = ndimage.uniform_filter(dx * dy, size=w, mode="reflect")

    det = sxx * syy - sxy * sxy
    tr = sxx + syy
    
    ratio = np.zeros_like(tr)
    nz = tr > 0
    ratio[nz] = det[nz] / (tr[nz] ** 2)

    ratio_cut = ratio_thresh / (1.0 + ratio_thresh) ** 2
    tr_floor = 1e-3 * tr.max()
    candidate = (ratio >= ratio_cut) & (tr > tr_floor)

    R = max(int(nms_radius), w // 2)
    local_max = det == ndimage.maximum_filter(det, size=2 * R + 1, mode="constant", cval=-np.inf)

    H, W = gray.shape
    border = M // 2 + w // 2
    inside = np.zeros_like(candidate)
    inside[border:H - border, border:W - border] = True

    keep = candidate & local_max & inside
    rows, cols = np.nonzero(keep)
    
    order = np.argsort(-det[rows, cols])[:max_corners]
    pts = np.stack([cols[order], rows[order]], axis=1).astype(float)
    return pts


def match_patches(gray1, gray2, pts1, pts2, metric="ncc", win=21,
                  mutual=True, ratio=None):
    """Window-based correspondence search between two corner sets.

    Args:
        metric: 'ssd', where a smaller score is better, or 'ncc', where a
            larger one is
        mutual: keep a pair only when each is the other's best match
    Returns:
        (matched_pts1, matched_pts2, scores)
    """
    ##  TODO: extract a win x win window around each corner and compare every
    ##  window in image 1 against every window in image 2 under `metric`:
    ##
    ##      'ssd'   sum over the window of (a - b)^2
    ##      'ncc'   sum (a - abar)(b - bbar)
    ##              / sqrt( sum (a - abar)^2 * sum (b - bbar)^2 )
    ##
    ##  Do it as a matrix operation, not with nested loops. For SSD,
    ##  expanding ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b turns the whole
    ##  thing into one matrix product; for NCC, mean-centre and unit-normalise
    ##  each window first and the product IS the score matrix.
    ##
    ##  `mutual` is a cheap outlier filter. Report what it does to your
    ##  numbers.
    #raise NotImplementedError

    h = win // 2
    H1, W1 = gray1.shape
    H2, W2 = gray2.shape

    ip1 = np.round(np.asarray(pts1, float)).astype(int).reshape(-1, 2)
    keep1 = ((ip1[:, 0] - h >= 0) & (ip1[:, 0] + h < W1) &
             (ip1[:, 1] - h >= 0) & (ip1[:, 1] + h < H1))
    ip1 = ip1[keep1]
    P1 = np.asarray(pts1)[keep1]

    ip2 = np.round(np.asarray(pts2, float)).astype(int).reshape(-1, 2)
    keep2 = ((ip2[:, 0] - h >= 0) & (ip2[:, 0] + h < W2) &
             (ip2[:, 1] - h >= 0) & (ip2[:, 1] + h < H2))
    ip2 = ip2[keep2]
    P2 = np.asarray(pts2)[keep2]

    if len(P1) == 0 or len(P2) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    off = np.arange(-h, h + 1)
    
    rows1 = ip1[:, 1, None, None] + off[None, :, None]
    cols1 = ip1[:, 0, None, None] + off[None, None, :]
    D1 = gray1[rows1, cols1].reshape(len(ip1), win * win)

    rows2 = ip2[:, 1, None, None] + off[None, :, None]
    cols2 = ip2[:, 0, None, None] + off[None, None, :]
    D2 = gray2[rows2, cols2].reshape(len(ip2), win * win)

    if metric == "ncc":
        A = D1 - D1.mean(axis=1, keepdims=True)
        B = D2 - D2.mean(axis=1, keepdims=True)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
        S = A @ B.T
        larger_is_better = True
    elif metric == "ssd":
        n1 = (D1 * D1).sum(axis=1)[:, None]
        n2 = (D2 * D2).sum(axis=1)[None, :]
        S = np.maximum(n1 + n2 - 2.0 * (D1 @ D2.T), 0.0)
        larger_is_better = False
    else:
        raise ValueError("metric must be 'ncc' or 'ssd'")

    score_mat = S if larger_is_better else -S
    j = np.argmax(score_mat, axis=1)
    i = np.arange(score_mat.shape[0])
    ok = np.ones(len(i), bool)

    if mutual:
        ok &= np.argmax(score_mat, axis=0)[j] == i

    if ratio is not None and score_mat.shape[1] >= 2:
        top2 = -np.sort(-score_mat, axis=1)[:, :2]
        d = (1.0 - top2) if larger_is_better else -top2
        ok &= d[:, 0] < ratio * d[:, 1]

    i, j = i[ok], j[ok]
    scores = S[i, j] if larger_is_better else -S[i, j]

    return P1[i], P2[j], scores


##  ============================================================================
##  Detector repeatability, used by Task 3(b)
##  ============================================================================

def detector_repeatability(ptsA, ptsX, H_ax, shapeX, thresh=3.0):
    """Fraction of view-a corners whose mapped position has a view-x corner
    within `thresh` pixels.

    Returns:
        (fraction, n_inside): the fraction, and how many mapped points landed
        inside view x at all
    """
    ##  TODO: map every point of ptsA through H_ax, keep the ones landing
    ##  inside view x, and for each of those ask whether ANY point of ptsX
    ##  lies within thresh pixels of it.
    ##
    ##  Note what this does and does not involve: the detector's output and
    ##  the known homography, and no descriptor at all. That is the whole
    ##  point. Task 3(b) uses it to separate detector behaviour from
    ##  descriptor behaviour instead of assuming which one is responsible.
    #raise NotImplementedError

    h, w = shapeX[:2]
    mapped = apply_H(H_ax, ptsA)
    inside = ((mapped[:, 0] >= 0) & (mapped[:, 0] < w) &
              (mapped[:, 1] >= 0) & (mapped[:, 1] < h))
    n_inside = int(inside.sum())

    if n_inside == 0 or len(ptsX) == 0:
        return float("nan"), n_inside

    dist, _ = cKDTree(np.asarray(ptsX, float)).query(mapped[inside], k=1)
    fraction = float((dist <= thresh).mean())
    return fraction, n_inside


##  ============================================================================
##  Task 3: SIFT
##  ============================================================================

def sift_match(img1, img2, ratio=0.75):
    """OpenCV SIFT keypoints and descriptors on both images, matched by
    Euclidean descriptor distance and Lowe's ratio test.

    Returns:
        (pts1, pts2, n_keypoints1, n_keypoints2), pts as (N, 2) arrays, so
        that the result goes straight into evaluate() like your Harris output
    """
    ##  TODO: cv2.SIFT_create().detectAndCompute() on each image, then
    ##  cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2) and keep a pair
    ##  only when d1 < ratio * d2. SURF is non-free and is not in a standard
    ##  OpenCV build, so SIFT is what you run.
    #raise NotImplementedError

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if img2.ndim == 3 else img2

    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(gray1, None)
    k2, d2 = sift.detectAndCompute(gray2, None)

    pts1 = np.array([k.pt for k in k1], float).reshape(-1, 2)
    pts2 = np.array([k.pt for k in k2], float).reshape(-1, 2)

    if d1 is None or d2 is None or len(d2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2)), len(k1), len(k2)

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]

    p1 = np.array([k1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    p2 = np.array([k2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    return p1, p2, len(k1), len(k2)


##  ============================================================================
##  Task 4: SuperPoint, matched by the provided SuperGlue wrapper
##  ============================================================================

def superglue_match(img_path1, img_path2, weights="outdoor", resize=None):
    """Thin wrapper around the provided superglue_ece661.SuperGlue class.

    Note that class is built by SuperGlue.create(...) rather than by calling
    it, and that its detectAndCompute and match methods take a path to an
    image file rather than an array.

    Returns:
        (pts1, pts2) in ORIGINAL image coordinates
    """
    ##  TODO: two of the wrapper's defaults are deliberately left for you to
    ##  think about, and Task 4 asks you to measure both rather than take
    ##  either on faith:
    ##    * superglue_wts defaults to 'indoor';
    ##    * resize defaults to [640, 480], which downsamples the image before
    ##      SuperPoint ever sees it.
    ##  Check what the wrapper's `scales` are doing before you trust the
    ##  coordinates it hands back.
    #raise NotImplementedError

    try:
        from superglue_ece661 import SuperGlue
    except ImportError:
        import sys
        # Attempt to inject path to SuperGlue network if it exists locally
        repo_path = os.environ.get("SUPERGLUE_REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SuperGluePretrainedNetwork")))
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from superglue_ece661 import SuperGlue

    rs = [640, 480] if resize is None else resize
    sg = SuperGlue.create(superglue_wts=weights, resize=rs)
    p1, p2, _ = sg.match(img_path1, img_path2)
    return p1, p2


##  ============================================================================
##  Bonus: orientation-normalised windows
##  ============================================================================

def dominant_orientation(gray, pts, sigma):
    """BONUS. A dominant orientation at each point, in degrees.

    Returns:
        (N,) array of angles, one per point of `pts`
    """
    ##  TODO: a gradient-magnitude-weighted histogram of gradient directions
    ##  in the neighbourhood of each point, as in step 4 of the SIFT
    ##  construction in Lecture 9, page 9-7. Say in your report how many bins
    ##  you used and how you handled the histogram peak.
    #raise NotImplementedError

    ff = ndimage.gaussian_filter(gray, sigma)
    gx = np.zeros_like(ff)
    gy = np.zeros_like(ff)
    gx[:, :-1] = ff[:, 1:] - ff[:, :-1]
    gy[:-1, :] = ff[1:, :] - ff[:-1, :]
    mag = np.hypot(gx, gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 360.0

    r = int(round(3 * 1.5 * sigma))
    H, W = gray.shape
    ip = np.round(np.asarray(pts, float)).astype(int).reshape(-1, 2)
    off = np.arange(-r, r + 1)

    rows = np.clip(ip[:, 1, None, None] + off[None, :, None], 0, H - 1)
    cols = np.clip(ip[:, 0, None, None] + off[None, None, :], 0, W - 1)

    g = np.exp(-(off[:, None] ** 2 + off[None, :] ** 2) / (2 * (1.5 * sigma) ** 2))
    wts = (mag[rows, cols] * g).reshape(len(ip), -1)

    N_ORI_BINS = 36
    bins = (ang[rows, cols] // (360.0 / N_ORI_BINS)).astype(int)
    bins = np.minimum(bins, N_ORI_BINS - 1).reshape(len(ip), -1)

    flat = (np.arange(len(ip))[:, None] * N_ORI_BINS + bins).ravel()
    hist = np.bincount(flat, weights=wts.ravel(), minlength=len(ip) * N_ORI_BINS)
    hist = hist.reshape(len(ip), N_ORI_BINS)

    k = np.argmax(hist, axis=1)
    idx = np.arange(len(ip))
    left = hist[idx, (k - 1) % N_ORI_BINS]
    mid = hist[idx, k]
    right = hist[idx, (k + 1) % N_ORI_BINS]
    denom = left - 2 * mid + right

    delta = np.zeros(len(ip))
    ok = np.abs(denom) > 1e-12
    delta[ok] = 0.5 * (left[ok] - right[ok]) / denom[ok]

    bw = 360.0 / N_ORI_BINS
    return ((k + 0.5 + delta) * bw) % 360.0


def match_patches_oriented(gray1, gray2, pts1, pts2, ang1, ang2, **kw):
    """BONUS. As match_patches, but each window is sampled in a frame rotated
    to that point's dominant orientation before the comparison is made.
    """
    ##  TODO: the comparison should no longer depend on how the camera was
    ##  rolled. Everything else about the search is unchanged.
    #raise NotImplementedError
    
    metric = kw.get("metric", "ncc")
    win = kw.get("win", 21)
    mutual = kw.get("mutual", True)
    ratio = kw.get("ratio", None)

    h = win // 2
    H1, W1 = gray1.shape
    H2, W2 = gray2.shape
    rad = h * np.sqrt(2) + 1

    def extract(gray, pts, angles):
        P = np.asarray(pts, float).reshape(-1, 2)
        keep = ((P[:, 0] - rad >= 0) & (P[:, 0] + rad < W1 if gray is gray1 else W2) &
                (P[:, 1] - rad >= 0) & (P[:, 1] + rad < H1 if gray is gray1 else H2))
        P_k = P[keep]
        t = np.radians(np.asarray(angles, float)[keep])

        off = np.arange(-h, h + 1, dtype=float)
        v, u = np.meshgrid(off, off, indexing="ij")
        c, s = np.cos(t)[:, None, None], np.sin(t)[:, None, None]

        xs = P_k[:, 0, None, None] + u * c - v * s
        ys = P_k[:, 1, None, None] + u * s + v * c

        D = ndimage.map_coordinates(gray, [ys.ravel(), xs.ravel()], order=1)
        return D.reshape(len(P_k), win * win), keep, P_k

    D1, k1, P1 = extract(gray1, pts1, ang1)
    D2, k2, P2 = extract(gray2, pts2, ang2)

    if len(D1) == 0 or len(D2) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), np.zeros(0)

    if metric == "ncc":
        A = D1 - D1.mean(axis=1, keepdims=True)
        B = D2 - D2.mean(axis=1, keepdims=True)
        A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-8
        B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-8
        S = A @ B.T
        larger_is_better = True
    elif metric == "ssd":
        n1 = (D1 * D1).sum(axis=1)[:, None]
        n2 = (D2 * D2).sum(axis=1)[None, :]
        S = np.maximum(n1 + n2 - 2.0 * (D1 @ D2.T), 0.0)
        larger_is_better = False
    else:
        raise ValueError("metric must be 'ncc' or 'ssd'")

    score_mat = S if larger_is_better else -S
    j = np.argmax(score_mat, axis=1)
    i = np.arange(score_mat.shape[0])
    ok = np.ones(len(i), bool)

    if mutual:
        ok &= np.argmax(score_mat, axis=0)[j] == i

    if ratio is not None and score_mat.shape[1] >= 2:
        top2 = -np.sort(-score_mat, axis=1)[:, :2]
        d = (1.0 - top2) if larger_is_better else -top2
        ok &= d[:, 0] < ratio * d[:, 1]

    i, j = i[ok], j[ok]
    scores = S[i, j] if larger_is_better else -S[i, j]

    return P1[i], P2[j], scores


##  ============================================================================
##  Driver
##  ============================================================================
import matplotlib
matplotlib.use('Agg') # Fixes the Wayland/Qt display error for Linux
import matplotlib.pyplot as plt

def main():
    truth = load_truth(TRUTH_PLANAR)
    imgA  = load_image(PLANAR["a"])
    gA    = to_gray(imgA)

    for tag in ("b", "c"):
        imgX = load_image(PLANAR[tag])
        gX   = to_gray(imgX)
        Hf, Hb = truth["H_a" + tag], truth["H_" + tag + "a"]
        print("\n=====================================")
        print(f"--- pair a -> {tag} ---")
        print("=====================================")

        # Setup side-by-side canvas for drawing matches
        hA, wA = imgA.shape[:2]
        hX, wX = imgX.shape[:2]
        canvas = np.zeros((max(hA, hX), wA + wX, 3), dtype=np.uint8)
        canvas[:hA, :wA] = imgA
        canvas[:hX, wA:] = imgX

        ##  Task 2: Harris Corners, NCC, SSD
        print("\n[Task 2] Harris Corners:")
        for sigma in (1.8, 4.5, 6.0):
            cA = harris_corners(gA, sigma=sigma)
            cX = harris_corners(gX, sigma=sigma)
            
            # Save corner plots for image A (only need to do it once per sigma)
            if tag == "b":
                fig, ax = plt.subplots()
                ax.imshow(imgA)
                ax.plot(cA[:, 0], cA[:, 1], 'r+', markersize=3) 
                ax.axis('off')
                save_figure(fig, f"t2a_corners_planar_a_s{sigma}.png")
                plt.close(fig)

            metrics = ["ncc"]
            if tag == "b" and sigma == 4.5:
                metrics.append("ssd")
            
            for metric in metrics:
                m1, m2, _ = match_patches(gA, gX, cA, cX, metric=metric)
                r = evaluate(m1, m2, Hf, Hb, imgX.shape, imgA.shape)
                
                # Updated print to match the Table 4 requirements
                print(f"  Harris s={sigma:4.1f} {metric.upper()}  "
                      f"corners {len(cA)}/{len(cX)}  "
                      f"matches {r['n_matches']:5d}  "
                      f"evaluable {r['n_evaluable']:5d}  "
                      f"inliers {r['n_inliers']:5d}  "
                      f"rate {100*r['inlier_rate']:5.1f}%  "
                      f"med err {r['median_err']:.2f}")

                # Plot Harris matches
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(canvas)
                n_plot = min(150, len(m1)) # Limit to 150 lines
                for i in range(n_plot):
                    ax.plot([m1[i, 0], m2[i, 0] + wA], [m1[i, 1], m2[i, 1]], 'y-', lw=0.5)
                    ax.plot(m1[i, 0], m1[i, 1], 'r+', markersize=2)
                    ax.plot(m2[i, 0] + wA, m2[i, 1], 'r+', markersize=2)
                ax.axis('off')
                save_figure(fig, f"t2{tag}_matches_{metric}_a_{tag}_s{sigma}.png")
                plt.close(fig)

        ##  Task 3(b): Detector Repeatability
        print("\n[Task 3(b)] Detector Repeatability:")
        cA_45 = harris_corners(gA, sigma=4.5)
        cX_45 = harris_corners(gX, sigma=4.5)
        rep_frac, rep_inside = detector_repeatability(cA_45, cX_45, Hf, imgX.shape)
        print(f"  Repeatability (s=4.5): {100*rep_frac:.1f}% (of {rep_inside} inside)")

        ##  Task 3: SIFT
        print("\n[Task 3(a)] SIFT Base (rho=0.75):")
        s1, s2, n1, n2 = sift_match(imgA, imgX)
        r = evaluate(s1, s2, Hf, Hb, imgX.shape, imgA.shape)
        print(f"  SIFT  kps {n1}/{n2}  matches {r['n_matches']}  evaluable {r['n_evaluable']}  "
              f"inliers {r['n_inliers']}  rate {100*r['inlier_rate']:.1f}%  med err {r['median_err']:.2f}")
              
        # Plot SIFT matches
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas)
        n_plot = min(150, len(s1))
        for i in range(n_plot):
            ax.plot([s1[i, 0], s2[i, 0] + wA], [s1[i, 1], s2[i, 1]], 'c-', lw=0.5)
            ax.plot(s1[i, 0], s1[i, 1], 'b+', markersize=2)
            ax.plot(s2[i, 0] + wA, s2[i, 1], 'b+', markersize=2)
        ax.axis('off')
        save_figure(fig, f"t3a_sift_a_{tag}.png")
        plt.close(fig)

        # Task 3(a): Ratio sweep only on pair a->b
        if tag == "b":
            print("\n[Task 3(a)] SIFT Ratio Sweep (a -> b):")
            for rho in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
                s1_r, s2_r, _, _ = sift_match(imgA, imgX, ratio=rho)
                r_rho = evaluate(s1_r, s2_r, Hf, Hb, imgX.shape, imgA.shape)
                print(f"  rho={rho}: matches {r_rho['n_matches']:3d}  evaluable {r_rho['n_evaluable']:3d}  "
                      f"inliers {r_rho['n_inliers']:3d}  rate {100*r_rho['inlier_rate']:5.1f}%")

##  Task 4: SuperPoint
        print("\n[Task 4] SuperPoint + SuperGlue:")
        try:
            # Default
            p1, p2 = superglue_match(os.path.join(DATA_DIR, PLANAR["a"]),
                                     os.path.join(DATA_DIR, PLANAR[tag]))
            r = evaluate(p1, p2, Hf, Hb, imgX.shape, imgA.shape)
            print(f"  SP+SG (default) matches {r['n_matches']}  evaluable {r['n_evaluable']}  "
                  f"inliers {r['n_inliers']}  rate {100*r['inlier_rate']:.1f}%  med err {r['median_err']:.2f}")
                  
            # Plot SP+SG matches (Default)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(canvas)
            n_plot = min(150, len(p1))
            for i in range(n_plot):
                ax.plot([p1[i, 0], p2[i, 0] + wA], [p1[i, 1], p2[i, 1]], 'm-', lw=0.5)
                ax.plot(p1[i, 0], p1[i, 1], 'g+', markersize=2)
                ax.plot(p2[i, 0] + wA, p2[i, 1], 'g+', markersize=2)
            ax.axis('off')
            save_figure(fig, f"t4a_spsg_a_{tag}.png")
            plt.close(fig)
            
            # Tasks 4(b) and 4(d) configurations run only on a->b
            if tag == "b":
                print("\n[Task 4(b) & 4(d)] SP+SG Configurations (a -> b):")
                
                # 4(b) Native size
                p1_nat, p2_nat = superglue_match(os.path.join(DATA_DIR, PLANAR["a"]),
                                                 os.path.join(DATA_DIR, PLANAR[tag]),
                                                 resize=[-1])
                r_nat = evaluate(p1_nat, p2_nat, Hf, Hb, imgX.shape, imgA.shape)
                print(f"  SP+SG (native size): matches {r_nat['n_matches']}  evaluable {r_nat['n_evaluable']}  "
                      f"inliers {r_nat['n_inliers']}  rate {100*r_nat['inlier_rate']:.1f}%  med err {r_nat['median_err']:.2f}")
                
                # Plot 4(b) Native Size
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(canvas)
                n_plot = min(150, len(p1_nat))
                for i in range(n_plot):
                    ax.plot([p1_nat[i, 0], p2_nat[i, 0] + wA], [p1_nat[i, 1], p2_nat[i, 1]], 'm-', lw=0.5)
                    ax.plot(p1_nat[i, 0], p1_nat[i, 1], 'g+', markersize=2)
                    ax.plot(p2_nat[i, 0] + wA, p2_nat[i, 1], 'g+', markersize=2)
                ax.axis('off')
                save_figure(fig, f"t4b_spsg_native_a_{tag}.png")
                plt.close(fig)

                # 4(d) Indoor weights
                p1_ind, p2_ind = superglue_match(os.path.join(DATA_DIR, PLANAR["a"]),
                                                 os.path.join(DATA_DIR, PLANAR[tag]),
                                                 weights="indoor")
                r_ind = evaluate(p1_ind, p2_ind, Hf, Hb, imgX.shape, imgA.shape)
                print(f"  SP+SG (indoor weights): matches {r_ind['n_matches']}  evaluable {r_ind['n_evaluable']}  "
                      f"inliers {r_ind['n_inliers']}  rate {100*r_ind['inlier_rate']:.1f}%  med err {r_ind['median_err']:.2f}")
                
                # Plot 4(d) Indoor Weights
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(canvas)
                n_plot = min(150, len(p1_ind))
                for i in range(n_plot):
                    ax.plot([p1_ind[i, 0], p2_ind[i, 0] + wA], [p1_ind[i, 1], p2_ind[i, 1]], 'm-', lw=0.5)
                    ax.plot(p1_ind[i, 0], p1_ind[i, 1], 'g+', markersize=2)
                    ax.plot(p2_ind[i, 0] + wA, p2_ind[i, 1], 'g+', markersize=2)
                ax.axis('off')
                save_figure(fig, f"t4d_spsg_indoor_a_{tag}.png")
                plt.close(fig)

        except Exception as e:
            print(f"  SP+SG skipped/failed: {e}")

##  Task 5(a): The Repetitive Facade
    print("\n=====================================")
    print("--- Task 5(a): Repetitive Facade ---")
    print("=====================================")
    truth_rep = load_truth(TRUTH_REPEAT)
    Hf_rep, Hb_rep = truth_rep["H_12"], truth_rep["H_21"]
    imgR1 = load_image(REPEAT[0])
    imgR2 = load_image(REPEAT[1])
    gR1, gR2 = to_gray(imgR1), to_gray(imgR2)

    # Setup side-by-side canvas for Task 5(a)
    hR1, wR1 = imgR1.shape[:2]
    hR2, wR2 = imgR2.shape[:2]
    canvas_rep = np.zeros((max(hR1, hR2), wR1 + wR2, 3), dtype=np.uint8)
    canvas_rep[:hR1, :wR1] = imgR1
    canvas_rep[:hR2, wR2:] = imgR2

    # 1. Harris + NCC, s=4.5
    cR1 = harris_corners(gR1, sigma=4.5)
    cR2 = harris_corners(gR2, sigma=4.5)
    m1_rep, m2_rep, _ = match_patches(gR1, gR2, cR1, cR2, metric="ncc")
    r_rep = evaluate(m1_rep, m2_rep, Hf_rep, Hb_rep, imgR2.shape, imgR1.shape)
    print(f"  Harris s=4.5 NCC: corners {len(cR1)}/{len(cR2)}  matches {r_rep['n_matches']}  "
          f"evaluable {r_rep['n_evaluable']}  inliers {r_rep['n_inliers']}  "
          f"rate {100*r_rep['inlier_rate']:.1f}%  med err {r_rep['median_err']:.2f}")

    # Plot Harris + NCC Facade
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas_rep)
    n_plot = min(150, len(m1_rep))
    for i in range(n_plot):
        ax.plot([m1_rep[i, 0], m2_rep[i, 0] + wR1], [m1_rep[i, 1], m2_rep[i, 1]], 'y-', lw=0.5)
        ax.plot(m1_rep[i, 0], m1_rep[i, 1], 'r+', markersize=2)
        ax.plot(m2_rep[i, 0] + wR1, m2_rep[i, 1], 'r+', markersize=2)
    ax.axis('off')
    save_figure(fig, "t5a_ncc_repeat.png")
    plt.close(fig)

    # 2. SIFT (For reference in Table 12)
    s1_rep, s2_rep, n1_rep, n2_rep = sift_match(imgR1, imgR2)
    rs_rep = evaluate(s1_rep, s2_rep, Hf_rep, Hb_rep, imgR2.shape, imgR1.shape)
    print(f"  SIFT rho=0.75: matches {rs_rep['n_matches']}  evaluable {rs_rep['n_evaluable']}  "
          f"inliers {rs_rep['n_inliers']}  rate {100*rs_rep['inlier_rate']:.1f}%  med err {rs_rep['median_err']:.2f}")

    # 3. SuperPoint + SuperGlue
    try:
        p1_rep, p2_rep = superglue_match(os.path.join(DATA_DIR, REPEAT[0]),
                                         os.path.join(DATA_DIR, REPEAT[1]),
                                         weights="outdoor")
        rsp_rep = evaluate(p1_rep, p2_rep, Hf_rep, Hb_rep, imgR2.shape, imgR1.shape)
        print(f"  SP+SG outdoor 640x480: matches {rsp_rep['n_matches']}  evaluable {rsp_rep['n_evaluable']}  "
              f"inliers {rsp_rep['n_inliers']}  rate {100*rsp_rep['inlier_rate']:.1f}%  med err {rsp_rep['median_err']:.2f}")
              
        # Plot SP+SG Facade
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas_rep)
        n_plot = min(150, len(p1_rep))
        for i in range(n_plot):
            ax.plot([p1_rep[i, 0], p2_rep[i, 0] + wR1], [p1_rep[i, 1], p2_rep[i, 1]], 'm-', lw=0.5)
            ax.plot(p1_rep[i, 0], p1_rep[i, 1], 'g+', markersize=2)
            ax.plot(p2_rep[i, 0] + wR1, p2_rep[i, 1], 'g+', markersize=2)
        ax.axis('off')
        save_figure(fig, "t5a_spsg_repeat.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"  SP+SG repeat skipped/failed: {e}")

## Task 5(b): Custom Photographs
    print("\n=====================================")
    print("--- Task 5(b): Custom Photographs ---")
    print("=====================================")
    try:
        img_own1 = load_image("own_1.jpg")
        img_own2 = load_image("own_2.jpg")
        gO1, gO2 = to_gray(img_own1), to_gray(img_own2)
        
        # Setup side-by-side canvas
        hO1, wO1 = img_own1.shape[:2]
        hO2, wO2 = img_own2.shape[:2]
        canvas_own = np.zeros((max(hO1, hO2), wO1 + wO2, 3), dtype=np.uint8)
        canvas_own[:hO1, :wO1] = img_own1
        canvas_own[:hO2, wO2:] = img_own2

        # 1 & 2. Harris + NCC & Harris + SSD (sigma = 4.5)
        cO1 = harris_corners(gO1, sigma=4.5)
        cO2 = harris_corners(gO2, sigma=4.5)
        
        for metric in ["ncc", "ssd"]:
            m1_own, m2_own, _ = match_patches(gO1, gO2, cO1, cO2, metric=metric)
            print(f"  Harris s=4.5 {metric.upper()} on custom pair: matches {len(m1_own)}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(canvas_own)
            n_plot = min(150, len(m1_own))
            for i in range(n_plot):
                ax.plot([m1_own[i, 0], m2_own[i, 0] + wO1], [m1_own[i, 1], m2_own[i, 1]], 'y-', lw=0.5)
                ax.plot(m1_own[i, 0], m1_own[i, 1], 'r+', markersize=2)
                ax.plot(m2_own[i, 0] + wO1, m2_own[i, 1], 'r+', markersize=2)
            ax.axis('off')
            save_figure(fig, f"t5b_{metric}_own.png")
            plt.close(fig)

        # 3. SIFT
        s1_own, s2_own, _, _ = sift_match(img_own1, img_own2)
        print(f"  SIFT rho=0.75 on custom pair: matches {len(s1_own)}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas_own)
        n_plot = min(150, len(s1_own))
        for i in range(n_plot):
            ax.plot([s1_own[i, 0], s2_own[i, 0] + wO1], [s1_own[i, 1], s2_own[i, 1]], 'c-', lw=0.5)
            ax.plot(s1_own[i, 0], s1_own[i, 1], 'b+', markersize=2)
            ax.plot(s2_own[i, 0] + wO1, s2_own[i, 1], 'b+', markersize=2)
        ax.axis('off')
        save_figure(fig, "t5b_sift_own.png")
        plt.close(fig)

        # 4. SuperPoint + SuperGlue (Indoor Weights)
        p1_own, p2_own = superglue_match(os.path.join(DATA_DIR, "own_1.jpg"),
                                         os.path.join(DATA_DIR, "own_2.jpg"),
                                         weights="indoor")
        print(f"  SP+SG indoor on custom pair: matches {len(p1_own)}")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(canvas_own)
        n_plot = min(150, len(p1_own))
        for i in range(n_plot):
            ax.plot([p1_own[i, 0], p2_own[i, 0] + wO1], [p1_own[i, 1], p2_own[i, 1]], 'm-', lw=0.5)
            ax.plot(p1_own[i, 0], p1_own[i, 1], 'g+', markersize=2)
            ax.plot(p2_own[i, 0] + wO1, p2_own[i, 1], 'g+', markersize=2)
        ax.axis('off')
        save_figure(fig, "t5b_spsg_own.png")
        plt.close(fig)

    except Exception as e:
        print(f"  Task 5(b) skipped or failed: {e}")

    # Generate Purdue overlays
    for i in (1, 2):
        imgP = load_image(PURDUE[i-1])
        cP = harris_corners(to_gray(imgP), sigma=4.5)
        fig, ax = plt.subplots(); ax.imshow(imgP); ax.plot(cP[:, 0], cP[:, 1], 'r+', markersize=3); ax.axis('off')
        save_figure(fig, f"t2a_corners_purdue_{i}.png"); plt.close(fig)

if __name__ == "__main__":
    main()