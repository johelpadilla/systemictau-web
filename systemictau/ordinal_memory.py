import numpy as np
import warnings

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator(args[0]) if len(args) == 1 and callable(args[0]) else decorator


@njit
def _generate_ordinal_patterns(x: np.ndarray, m: int, delay: int):
    """
    Generates ordinal patterns (permutation symbols) for a 1D time series.
    """
    n_patterns = len(x) - (m - 1) * delay
    if n_patterns <= 0:
        return np.empty(0, dtype=np.int64)

    symbols = np.zeros(n_patterns, dtype=np.int64)
    factorials = np.zeros(m, dtype=np.int64)
    factorials[0] = 1
    for i in range(1, m):
        factorials[i] = factorials[i - 1] * i

    for i in range(n_patterns):
        word = np.zeros(m, dtype=np.float64)
        for j in range(m):
            word[j] = x[i + j * delay]

        perm = np.argsort(word)

        symbol = 0
        for j in range(m - 1):
            count = 0
            for k in range(j + 1, m):
                if perm[k] < perm[j]:
                    count += 1
            symbol += count * factorials[m - 1 - j]

        symbols[i] = symbol

    return symbols


@njit
def _entropy_from_counts(counts: np.ndarray) -> float:
    """Calculates Shannon entropy (in bits) with numerical protection."""
    total = np.sum(counts)
    if total == 0:
        return 0.0
    p = counts / total
    p = np.clip(p, 1e-15, 1.0)
    return -np.sum(p * np.log2(p))


def symbolic_transfer_entropy_pair(x: np.ndarray, y: np.ndarray, m: int = 3, delay: int = 1) -> float:
    """
    Computes Symbolic Transfer Entropy from x to y (T_{X→Y}) using ordinal patterns.

    This implementation follows the standard formulation of Symbolic Transfer Entropy.
    Returns a non-negative value in bits.
    """
    sym_x = _generate_ordinal_patterns(x, m, delay)
    sym_y = _generate_ordinal_patterns(y, m, delay)

    min_len = min(len(sym_x), len(sym_y))
    if min_len < 2:
        return 0.0

    # Use aligned past and future symbols
    x_past = sym_x[:min_len - 1]
    y_past = sym_y[:min_len - 1]
    y_fut = sym_y[1:min_len]

    # Build joint distributions
    joint3 = np.column_stack((y_fut, y_past, x_past))
    joint_yfut_ypast = np.column_stack((y_fut, y_past))
    joint_ypast_xpast = np.column_stack((y_past, x_past))

    # Count occurrences
    _, counts3 = np.unique(joint3, axis=0, return_counts=True)
    _, counts_yfut_ypast = np.unique(joint_yfut_ypast, axis=0, return_counts=True)
    _, counts_ypast_xpast = np.unique(joint_ypast_xpast, axis=0, return_counts=True)
    _, counts_ypast = np.unique(y_past, return_counts=True)

    # Entropies
    H3 = _entropy_from_counts(counts3)
    H_yfut_ypast = _entropy_from_counts(counts_yfut_ypast)
    H_ypast_xpast = _entropy_from_counts(counts_ypast_xpast)
    H_ypast = _entropy_from_counts(counts_ypast)

    # Transfer Entropy: H(Yfut|Ypast) - H(Yfut|Ypast, Xpast)
    te = (H_yfut_ypast - H_ypast) - (H3 - H_ypast_xpast)
    return max(0.0, te)


def rank_mutual_information(X: np.ndarray) -> float:
    """
    Lite Mode: Average pairwise Rank-based Mutual Information.

    This is a fast approximation using 2D histograms on rank-transformed data.
    Suitable for quick estimation of non-linear coupling strength.
    Not recommended for high-precision analysis.
    """
    T_len, N_vars = X.shape
    if N_vars < 2:
        return 0.0

    ranks = np.argsort(np.argsort(X, axis=0), axis=0)
    B = max(2, int(np.sqrt(T_len)))

    total_mi = 0.0
    pairs = 0

    for i in range(N_vars):
        for j in range(i + 1, N_vars):
            hist, _, _ = np.histogram2d(ranks[:, i], ranks[:, j], bins=B)
            pxy = hist / np.sum(hist)
            px = np.sum(pxy, axis=1)
            py = np.sum(pxy, axis=0)
            px_py = np.outer(px, py)

            pxy_clip = np.clip(pxy, 1e-15, 1.0)
            px_py_clip = np.clip(px_py, 1e-15, 1.0)

            mask = pxy > 0
            mi = np.sum(pxy[mask] * np.log2(pxy_clip[mask] / px_py_clip[mask]))
            total_mi += max(0.0, mi)
            pairs += 1

    return total_mi / pairs if pairs > 0 else 0.0


def compute_ordinal_features(
    X: np.ndarray,
    window_size: int,
    stride: int = 5,
    mode: str = "lite",
    m: int = 3,
    delay: int = 1
) -> dict:
    """
    Main function to compute Ordinal Memory features over sliding windows.

    Parameters
    ----------
    mode : {"lite", "full"}
        - "lite": Rank Mutual Information (fast, symmetric)
        - "full": Symbolic Transfer Entropy (directional, slower)
    """
    T_len, N_vars = X.shape

    # Parameter validation
    if m < 2:
        raise ValueError("Embedding dimension 'm' must be >= 2.")
    if window_size <= m * delay:
        raise ValueError(f"window_size must be greater than m * delay ({m * delay}).")

    if mode == "full" and not HAS_NUMBA:
        warnings.warn(
            "Numba is not available. 'full' mode (Symbolic Transfer Entropy) will be very slow.",
            RuntimeWarning
        )

    calc_indices = list(range(window_size - 1, T_len, stride))

    total_flow = np.full(T_len, np.nan)
    net_asymmetry = np.full(T_len, np.nan)
    mean_te = np.full(T_len, np.nan)

    for t in calc_indices:
        window = X[t - window_size + 1 : t + 1, :]

        if mode == "lite":
            mi = rank_mutual_information(window)
            total_flow[t] = mi
            net_asymmetry[t] = 0.0
            mean_te[t] = mi

        elif mode == "full":
            ste_matrix = np.zeros((N_vars, N_vars))
            for i in range(N_vars):
                for j in range(N_vars):
                    if i != j:
                        ste_matrix[i, j] = symbolic_transfer_entropy_pair(
                            window[:, i], window[:, j], m, delay
                        )

            total_flow[t] = np.sum(ste_matrix)

            asym = 0.0
            for i in range(N_vars):
                for j in range(i + 1, N_vars):
                    asym += abs(ste_matrix[i, j] - ste_matrix[j, i])
            net_asymmetry[t] = asym

            positive_te = ste_matrix[ste_matrix > 0]
            mean_te[t] = np.mean(positive_te) if len(positive_te) > 0 else 0.0

    return {
        "total_flow": total_flow,
        "net_asymmetry": net_asymmetry,
        "mean_te": mean_te,
        "computation_windows": calc_indices,
        "mode": mode,
        "m": m,
        "delay": delay
    }
