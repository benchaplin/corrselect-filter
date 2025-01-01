import numpy as np

def measure_point_biserial_correlation(scores, labels):
    """
    Compute the point-biserial correlation between:
      - scores: a 1D numpy array of shape (n,)
      - labels:    a 1D numpy array of shape (n,) of 0/1
    Returns r_pb in [-1,1].
    """
    scores = scores.astype(float)
    labels = labels.astype(float)
    
    s = labels.mean() # fraction labeled 1
    if s == 0 or s == 1:
        # Degenerate case: all labels are 0 or all are 1
        # correlation is undefined, return 0 or sign that indicates a corner
        return 0.0
    
    s_d = scores.std(ddof=1)
    if s_d == 0:
        # All scores identical -> correlation is zero
        return 0.0
    
    # Means of scores in each group
    d1_mean = scores[labels == 1].mean()
    d0_mean = scores[labels == 0].mean()
    
    # Point-biserial formula
    r_pb = ((d1_mean - d0_mean) / s_d) * np.sqrt(s * (1.0 - s))
    return r_pb


def create_filter_threshold_random_flips(scores, 
                                         target_corr, # c in [-1, 1]
                                         selectivity, # s in [0, 1]
                                         max_iter=100,
                                         flip_batch_size=10,
                                         corr_tolerance=0.02,
                                         random_seed=None):
    """
    Create a 0/1 filter over 'scores' to achieve:
      1) fraction = selectivity  (s)
      2) point-biserial correlation ~ target_corr  (c)
    
    Returns:
      labels: a numpy array of 0/1 of length len(scores)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n = len(scores)
    # Sort by score (ascending)
    sorted_idx = np.argsort(scores)
    
    # Number of items to label 1
    k = int(np.floor(selectivity * n))
    
    labels = np.zeros(n, dtype=int)
    if target_corr < 0:
        # Label the worst k scores as 1
        ones_indices = sorted_idx[:k]
    else:
        # Label the best k scores as 1
        ones_indices = sorted_idx[-k:]
    
    labels[ones_indices] = 1
    
    # Measure initial correlation
    current_corr = measure_point_biserial_correlation(scores, labels)

    sorted_idx = np.argsort(scores)
    
    # If correlation is off, we'll do random flips to move correlation toward target.
    
    def correlation_error(r_now, r_target):
        return abs(r_now - r_target)
    
    current_err = correlation_error(current_corr, target_corr)
    
    # Early exit if already good
    if current_err < corr_tolerance:
        return labels
    
    # We'll attempt up to max_iter "rounds" of flipping
    for _ in range(max_iter):
        labels_copy = labels.copy()
        
        # We'll do a small batch of flips each iteration
        for _batch in range(flip_batch_size):
            if current_corr < target_corr:
                # Random flip of 0 -> 1 from the worst scores
                candidate_0 = np.where(labels_copy == 0)[0]
                candidates = candidate_0[np.argsort(scores[candidate_0])[:10]] # bottom 10
                if len(candidates) > 0:
                    flip_idx = np.random.choice(candidates)
                    labels_copy[flip_idx] = 1
            else:
                # Random flip of 1 -> 0 from the best scores
                candidate_1 = np.where(labels_copy == 1)[0]
                candidates = candidate_1[np.argsort(scores[candidate_1])[-10:]] # top 10
                if len(candidates) > 0:
                    flip_idx = np.random.choice(candidates)
                    labels_copy[flip_idx] = 0
        
        # measure new correlation
        new_corr = measure_point_biserial_correlation(scores, labels_copy)
        new_error = correlation_error(new_corr, target_corr)
        
        # If better, accept the flips
        if new_error < current_err:
            labels = labels_copy
            current_corr = new_corr
            current_err = new_error
        
        # Check if within tolerance
        if current_err < corr_tolerance:
            break
    
    return labels

if __name__ == "__main__":
    np.random.seed(42)
    n = 1000
    # Create some random scores in range [0, 1]
    scores = np.random.rand(n)
    
    # Desired selectivity s and correlation c
    s = 0.2
    c = -0.5
    
    labels = create_filter_threshold_random_flips(
        scores,
        target_corr=c,
        selectivity=s,
        max_iter=50,
        flip_batch_size=20,
        corr_tolerance=0.01,
        random_seed=123
    )
    
    # Check results
    final_corr = measure_point_biserial_correlation(scores, labels)
    fraction = labels.mean()
    
    print(f"Final correlation: {final_corr:.3f} (target was {c})")
    print(f"Final fraction of 1s: {fraction} (target was {s})")