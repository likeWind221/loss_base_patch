"""
Shape metrics for binary masks.

All inputs are expected to be numpy arrays of dtype bool or {0,1}.
"""

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

try:
	import cv2  # Optional for more precise CC stats
	_HAS_CV = True
except Exception:
	_HAS_CV = False


def _structure(connectivity: int) -> np.ndarray:
	"""Return the structuring element for 4- or 8-neighborhood."""
	if connectivity == 1:
		# 4-neighbor (cross)
		return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
	# 8-neighbor (full 3x3)
	return np.ones((3, 3), dtype=np.uint8)


def shape_stats(
	mask: np.ndarray,
	connectivity: int = 2,
	min_area: int = 0,
	use_cv: bool = True,
	mode: str = "auto",
) -> Dict[str, float]:
	"""Compute basic shape metrics of a binary mask.

	Metrics:
		ncc: number of connected components (<=3 is preferred).
		holes: number of holes (0 is preferred).
		lcc_ratio: area of the largest component divided by total area.
		mean_cc: mean component area (area_total / ncc).
		compactness: (perimeter^2) / area for the largest component; lower is tighter.

	Args:
		mask: HxW binary mask (bool or {0,1}).
		connectivity: 1 for 4-neighbor, 2 for 8-neighbor.
		min_area: discard connected components smaller than this area (pixel count) before stats.
		use_cv: if True and OpenCV is available, use connectedComponentsWithStats for robust counting.
		mode: "auto"/"cc"/"contour". "contour" uses cv2.findContours (external) to count blobs.

	Returns:
		Dictionary with the metrics above and area_total.
	"""

	mask_bool = mask.astype(bool)
	area_total = float(mask_bool.sum())

	if area_total == 0:
		return {
			"ncc": 0,
			"holes": 0,
			"lcc_ratio": 0.0,
			"mean_cc": 0.0,
			"compactness": 0.0,
			"area_total": 0.0,
		}

	# Contour-based external blob counting (optional, better for分裂小块)
	if mode == "contour" and _HAS_CV:
		mask_u8 = mask_bool.astype(np.uint8)
		contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		areas = [cv2.contourArea(c) for c in contours]
		kept = [(c, a) for c, a in zip(contours, areas) if a >= max(1, min_area)]
		if len(kept) == 0:
			return {
				"ncc": 0,
				"holes": 0,
				"lcc_ratio": 0.0,
				"mean_cc": 0.0,
				"compactness": 0.0,
				"area_total": 0.0,
			}

		areas_kept = [a for _, a in kept]
		area_kept_total = float(np.sum(areas_kept))
		ncc = len(areas_kept)
		lcc_area = float(np.max(areas_kept))
		lcc_ratio = lcc_area / (area_total + 1e-6)
		mean_cc = area_kept_total / float(ncc)

		lcc_idx = int(np.argmax(areas_kept))
		lcc_contour = kept[lcc_idx][0]
		perimeter = float(cv2.arcLength(lcc_contour, True)) if lcc_contour is not None else 0.0
		compactness = (perimeter * perimeter) / (lcc_area + 1e-6) if lcc_area > 0 else 0.0

		return {
			"ncc": int(ncc),
			"holes": 0,
			"lcc_ratio": lcc_ratio,
			"mean_cc": mean_cc,
			"compactness": compactness,
			"area_total": float(area_kept_total),
		}

	if use_cv and _HAS_CV:
		# OpenCV connected components with stats allows min-area filtering easily
		cv_conn = 4 if connectivity == 1 else 8
		num_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8), connectivity=cv_conn)
		areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
		labels_keep = [i for i, a in enumerate(areas, start=1) if a >= max(1, min_area)]
		if len(labels_keep) == 0:
			return {
				"ncc": 0,
				"holes": 0,
				"lcc_ratio": 0.0,
				"mean_cc": 0.0,
				"compactness": 0.0,
				"area_total": area_total,
			}
		mask_keep = np.isin(labeled, labels_keep)
		areas_kept = [areas[i - 1] for i in labels_keep]
		area_kept_total = float(np.sum(areas_kept))
		ncc = len(labels_keep)
		lcc_area = float(np.max(areas_kept))
		lcc_ratio = lcc_area / (area_kept_total + 1e-6)
		mean_cc = area_kept_total / float(ncc) if ncc > 0 else 0.0

		# Compactness on largest component (approx perimeter via contour length)
		lcc_label = labels_keep[int(np.argmax(areas_kept))]
		lcc_mask = (labeled == lcc_label).astype(np.uint8)
		contours, _ = cv2.findContours(lcc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			perimeter = float(cv2.arcLength(contours[0], True))
		else:
			perimeter = 0.0
		compactness = (perimeter * perimeter) / (lcc_area + 1e-6) if lcc_area > 0 else 0.0

		# Hole count using binary fill on the kept mask
		filled = ndi.binary_fill_holes(mask_keep)
		holes = int(filled.sum() - mask_keep.sum())

		return {
			"ncc": int(ncc),
			"holes": holes,
			"lcc_ratio": lcc_ratio,
			"mean_cc": mean_cc,
			"compactness": compactness,
			"area_total": float(area_kept_total),
		}

	# SciPy fallback
	structure = _structure(connectivity)
	labeled, ncc = ndi.label(mask_bool, structure=structure)

	# Hole count via fill-holes difference
	filled = ndi.binary_fill_holes(mask_bool)
	holes = int(filled.sum() - mask_bool.sum())

	# Largest component stats
	areas = np.bincount(labeled.ravel())[1:]  # skip background
	if min_area > 0:
		keep_idx = np.where(areas >= min_area)[0]
		mask_keep = np.isin(labeled, keep_idx + 1)
		labeled, ncc = ndi.label(mask_keep, structure=structure)
		areas = np.bincount(labeled.ravel())[1:]

	lcc_area = float(areas.max()) if len(areas) else 0.0
	lcc_ratio = lcc_area / (float(mask_bool.sum()) + 1e-6)

	mean_cc = (float(mask_bool.sum()) / float(ncc)) if ncc > 0 else 0.0

	# Compactness on the largest component only
	if len(areas) > 0:
		lcc_label = 1 + int(np.argmax(areas))
		lcc_mask = labeled == lcc_label
		eroded = ndi.binary_erosion(lcc_mask, structure=structure, border_value=0)
		perimeter = float(np.logical_xor(lcc_mask, eroded).sum())
		compactness = (perimeter * perimeter) / (lcc_area + 1e-6)
	else:
		compactness = 0.0

	return {
		"ncc": int(ncc),
		"holes": holes,
		"lcc_ratio": lcc_ratio,
		"mean_cc": mean_cc,
		"compactness": compactness,
		"area_total": area_total,
	}


__all__ = ["shape_stats"]
