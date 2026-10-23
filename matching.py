"""
=============================================================================
MODULE 4: Matching & Identification
=============================================================================
Implements face identification by comparing query feature templates against
all stored database templates using classical distance/similarity measures.

Distance Measures Implemented:
  - Euclidean Distance (L2)       — primary measure
  - Cosine Similarity             — angle-based comparison
  - Chi-Squared Distance          — histogram-specific
  - Manhattan Distance (L1)       — robust to outliers
  - Correlation Coefficient       — normalized cross-correlation
=============================================================================
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

from database import FaceDatabase
from preprocessing import full_preprocess_pipeline
from feature_extraction import extract_feature_template


# ---------------------------------------------------------------------------
# 4.1  Data Classes for Results
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Container for a single face matching result.

    Attributes:
        matched:     True if identity was found within threshold.
        user_id:     Database ID of the best match (None if unmatched).
        name:        Name of the best match ("Unknown" if unmatched).
        distance:    Euclidean distance to the best match template.
        similarity:  Cosine similarity score [0, 1] (higher = more similar).
        rank:        Rank in the sorted match list (1 = closest).
        all_scores:  Full sorted list of (user_id, name, distance) tuples.
    """
    matched: bool
    user_id: Optional[int]
    name: str
    distance: float
    similarity: float
    rank: int
    all_scores: List[Tuple[int, str, float]]


# ---------------------------------------------------------------------------
# 4.2  Distance & Similarity Functions
# ---------------------------------------------------------------------------

def euclidean_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute the Euclidean (L2) distance between two feature vectors.

    Definition:
        d(a, b) = sqrt(Σ (a_i - b_i)²)

    Supports multi-representation galleries (returns minimum distance across gallery templates).

    Args:
        vec_a: Query feature vector (float32, length N).
        vec_b: Database feature vector (float32, length N or M*N).

    Returns:
        Scalar Euclidean distance (float).
    """
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)

    # Multi-template gallery support (M representations of length N)
    if len(b) > len(a) and len(b) % len(a) == 0:
        n_gallery = len(b) // len(a)
        dists = [np.linalg.norm(a - b[i * len(a) : (i + 1) * len(a)]) for i in range(n_gallery)]
        return float(min(dists))
    elif len(a) > len(b) and len(a) % len(b) == 0:
        n_gallery = len(a) // len(b)
        dists = [np.linalg.norm(a[i * len(b) : (i + 1) * len(b)] - b) for i in range(n_gallery)]
        return float(min(dists))
    elif a.shape != b.shape:
        raise ValueError(
            f"Template length mismatch: query={vec_a.shape}, "
            f"stored={vec_b.shape}. Re-enroll the user."
        )

    return float(np.linalg.norm(a - b))


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two feature vectors.

    Definition:
        sim(a, b) = (a · b) / (||a|| × ||b||)

    Supports multi-representation galleries (returns maximum similarity).

    Args:
        vec_a: Query feature vector (float32).
        vec_b: Database feature vector (float32).

    Returns:
        Cosine similarity score in [0, 1].
    """
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)

    if len(b) > len(a) and len(b) % len(a) == 0:
        n_gallery = len(b) // len(a)
        sims = []
        norm_a = np.linalg.norm(a) + 1e-10
        for i in range(n_gallery):
            sub_b = b[i * len(a) : (i + 1) * len(a)]
            sims.append(np.dot(a, sub_b) / (norm_a * np.linalg.norm(sub_b) + 1e-10))
        return float(max(sims))

    dot = np.dot(a, b)
    norm_product = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(dot / norm_product)


def chi_squared_distance(hist_a: np.ndarray, hist_b: np.ndarray) -> float:
    """
    Compute Chi-Squared (χ²) distance between two normalized histograms.

    Definition:
        χ²(a, b) = Σ (a_i - b_i)² / (a_i + b_i + ε)

    Particularly well-suited for comparing histogram-type feature vectors
    (LBP histograms, gradient orientation histograms).
    Chi-squared penalizes relative differences, making it more sensitive
    to changes in histogram shape than Euclidean distance.

    Args:
        hist_a: Query histogram vector (float32, sum ≈ 1).
        hist_b: Database histogram vector (float32, sum ≈ 1).

    Returns:
        Chi-squared distance (float). Range: [0, +∞). Lower = more similar.
    """
    a = hist_a.astype(np.float64)
    b = hist_b.astype(np.float64)
    num = (a - b) ** 2
    den = a + b + 1e-10
    return float(np.sum(num / den))


def manhattan_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute Manhattan (L1) distance between two feature vectors.

    Definition:
        d(a, b) = Σ |a_i - b_i|

    Also called 'city block' or 'taxicab' distance.
    More robust to outlier feature dimensions than L2 distance because
    it doesn't square the differences.

    Args:
        vec_a: Query feature vector (float32).
        vec_b: Database feature vector (float32).

    Returns:
        Manhattan distance (float). Range: [0, +∞).
    """
    return float(np.sum(np.abs(vec_a.astype(np.float64) -
                                vec_b.astype(np.float64))))


def normalized_correlation(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute normalized cross-correlation (NCC) between two feature vectors.

    Definition:
        NCC(a, b) = Σ [(a_i - μ_a)(b_i - μ_b)] / (σ_a × σ_b × N)

    Equivalent to Pearson correlation coefficient.
    Range: [-1, 1]. Values near +1 indicate strong positive correlation
    (similar patterns), near 0 means uncorrelated (different faces).

    Advantages: Invariant to both additive and multiplicative illumination.

    Args:
        vec_a: Query feature vector (float32).
        vec_b: Database feature vector (float32).

    Returns:
        NCC score in [-1, 1]. Higher = more similar.
    """
    a = vec_a.astype(np.float64)
    b = vec_b.astype(np.float64)
    a_norm = a - a.mean()
    b_norm = b - b.mean()
    num = np.dot(a_norm, b_norm)
    den = (np.linalg.norm(a_norm) * np.linalg.norm(b_norm)) + 1e-10
    return float(num / den)


# ---------------------------------------------------------------------------
# 4.3  Core Matching Engine
# ---------------------------------------------------------------------------

class FaceMatcher:
    """
    Face identification engine using template matching against SQLite database.

    Workflow:
        1. Load all user templates from the database.
        2. For a query face image, run the full preprocessing + feature
           extraction pipeline (identical to enrollment).
        3. Compare the query template against each stored template using
           Euclidean distance as the primary metric.
        4. Sort results by distance (ascending).
        5. If the minimum distance < threshold → matched user.
           Otherwise → "Unknown".

    Attributes:
        db         (FaceDatabase): The database manager instance.
        threshold  (float):       Euclidean distance threshold for acceptance.
                                  Tune this for your dataset.
    """

    # Calibrated Euclidean distance threshold for Unit-Normalized Anatomical LBPH
    # Intra-class (same person, diff photo / tilted / bearded): ~0.30 - 0.39
    # Inter-class (different individuals)                    : ~0.52 - 0.58
    # Threshold 0.46 provides a wide, robust decision boundary.
    DEFAULT_THRESHOLD = 0.46

    def __init__(self, db: FaceDatabase,
                 threshold: float = DEFAULT_THRESHOLD):
        """
        Initialize the FaceMatcher.

        Args:
            db:         FaceDatabase instance for template retrieval.
            threshold:  Euclidean distance threshold (lower = stricter).
                        Faces with distance < threshold are accepted.
        """
        self.db = db
        self.threshold = threshold

    def identify(self, query_image: np.ndarray) -> MatchResult:
        """
        Identify the person in a query face image.

        Full pipeline:
            1. Preprocess query image (grayscale, CLAHE, Gaussian blur, resize)
            2. Extract feature template (LBP + gradient hist + HOG)
            3. Retrieve all stored templates from database
            4. Compute Euclidean distance to each template
            5. Sort by distance and apply threshold decision

        Args:
            query_image: Raw BGR image from webcam or file (uint8, H×W×3).

        Returns:
            MatchResult object with identity, distance, similarity, and ranking.
        """
        # --- Step 1: Preprocessing ---
        face_roi = full_preprocess_pipeline(query_image)
        if face_roi is None:
            return MatchResult(
                matched=False, user_id=None, name="No Face Detected",
                distance=float("inf"), similarity=0.0, rank=-1, all_scores=[]
            )

        # --- Step 2: Feature Extraction ---
        query_template = extract_feature_template(face_roi)

        return self.identify_from_template(query_template)

    def identify_from_template(self,
                               query_template: np.ndarray) -> MatchResult:
        """
        Identify a person given a pre-extracted feature template.

        This method is exposed separately so that callers who have already
        run the preprocessing + feature extraction pipeline can skip those
        steps and go directly to matching.

        Args:
            query_template: 1D float32 feature vector (already extracted).

        Returns:
            MatchResult object with identity and scoring information.
        """
        # --- Step 3: Retrieve all database templates ---
        all_users = self.db.get_all_users()

        if not all_users:
            return MatchResult(
                matched=False, user_id=None, name="Database Empty",
                distance=float("inf"), similarity=0.0, rank=-1, all_scores=[]
            )

        # --- Step 4: Compute distances ---
        scores = []
        for (user_id, name, stored_template) in all_users:
            try:
                dist = euclidean_distance(query_template, stored_template)
                sim = cosine_similarity(query_template, stored_template)
                scores.append((user_id, name, dist, sim))
            except ValueError as e:
                # Template dimension mismatch — skip this user
                print(f"[MATCHER] Skipping user '{name}' (ID={user_id}): {e}")
                continue

        if not scores:
            return MatchResult(
                matched=False, user_id=None, name="Unknown",
                distance=float("inf"), similarity=0.0, rank=-1, all_scores=[]
            )

        # --- Step 5: Sort by Euclidean distance (ascending) ---
        scores.sort(key=lambda x: x[2])

        # Best match is the first entry (smallest distance)
        best_id, best_name, best_dist, best_sim = scores[0]

        # Apply threshold decision
        matched = best_dist < self.threshold
        identity = best_name if matched else "Unknown"

        # Package all scores for the caller (without similarity for brevity)
        all_scores_summary = [(uid, name, dist)
                              for uid, name, dist, sim in scores]

        return MatchResult(
            matched=matched,
            user_id=best_id if matched else None,
            name=identity,
            distance=best_dist,
            similarity=best_sim,
            rank=1,
            all_scores=all_scores_summary
        )

    def identify_top_k(self, query_image: np.ndarray,
                       k: int = 3) -> List[MatchResult]:
        """
        Return top-K matching candidates for a query image.

        Useful for displaying ranked results in a UI or for analysis.

        Args:
            query_image: Raw BGR image (uint8, H×W×3).
            k:           Number of top candidates to return.

        Returns:
            List of up to k MatchResult objects, sorted by distance.
        """
        face_roi = full_preprocess_pipeline(query_image)
        if face_roi is None:
            return []

        query_template = extract_feature_template(face_roi)
        all_users = self.db.get_all_users()

        if not all_users:
            return []

        scores = []
        for (user_id, name, stored_template) in all_users:
            try:
                dist = euclidean_distance(query_template, stored_template)
                sim = cosine_similarity(query_template, stored_template)
                scores.append((user_id, name, dist, sim))
            except ValueError:
                continue

        scores.sort(key=lambda x: x[2])
        results = []
        for rank, (uid, name, dist, sim) in enumerate(scores[:k], start=1):
            matched = dist < self.threshold
            results.append(MatchResult(
                matched=matched,
                user_id=uid if matched else None,
                name=name if matched else f"Unknown (closest: {name})",
                distance=dist,
                similarity=sim,
                rank=rank,
                all_scores=[(u, n, d) for u, n, d, _ in scores]
            ))
        return results

    def set_threshold(self, new_threshold: float) -> None:
        """
        Update the matching threshold at runtime.

        Lower threshold = stricter matching (fewer false positives).
        Higher threshold = more lenient matching (fewer false negatives).

        Args:
            new_threshold: New Euclidean distance threshold.
        """
        self.threshold = new_threshold
        print(f"[MATCHER] Threshold updated to {new_threshold:.4f}.")
