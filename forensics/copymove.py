"""Copy-move forgery detection: ORB keypoints matched against themselves.
Clusters of near-identical descriptors that are spatially far apart
indicate a region was duplicated elsewhere in the same image."""

import cv2
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from core.schemas import Signal


def run_copymove(image_path: str) -> Signal:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(image, None)

    if descriptors is None or len(descriptors) < 2:
        return Signal(
            name="copymove", score=0.0, weight=0.0,
            note="not enough keypoints to check", extras={"clusters": 0},
        )

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors, descriptors, k=2)

    good_matches = []
    points = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pt1 = keypoints[m.queryIdx].pt
            pt2 = keypoints[m.trainIdx].pt
            spatial_dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
            if spatial_dist > 50:  # must be far apart, not the same keypoint twice
                good_matches.append(m)
                points.append(pt1)
                points.append(pt2)

    cluster_count = 0
    score = 0.0
    if len(points) > 4:
        clusters = fcluster(
            linkage(np.array(points), method="single", metric="euclidean"),
            t=40, criterion="distance",
        )
        unique_clusters = len(set(clusters))
        if unique_clusters >= 2:
            cluster_count = unique_clusters
            score = min(1.0, float(len(good_matches)) / 50.0)

    return Signal(
        name="copymove", score=score, weight=0.0,
        note=f"found {len(good_matches)} suspicious self-matches",
        extras={"cluster_count": cluster_count, "matches": len(good_matches)},
    )
