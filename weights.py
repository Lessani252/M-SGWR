"""Shared weighting for selection, fitting, and inference."""
import numpy as np


def composite_weights(coords, i, bw, alpha, data, fixed):
    coords = np.asarray(coords, dtype=float)
    if not 0 <= alpha <= 1:
        raise ValueError('alpha must be between zero and one.')
    distances = np.linalg.norm(coords - coords[i], axis=1)
    if np.isinf(bw) and bw > 0:
        spatial = np.ones(len(coords))
    elif fixed:
        if not np.isfinite(bw) or bw <= 0:
            raise ValueError('Fixed bandwidth must be positive.')
        spatial = np.exp(-0.5 * (distances / bw)**2)
    else:
        if not np.isfinite(bw) or int(bw) != bw or not 2 <= bw <= len(coords):
            raise ValueError('Adaptive bandwidth must be an integer from 2 through n.')
        radius = np.partition(distances, int(bw) - 1)[int(bw) - 1] * 1.0000001
        if radius <= 0:
            raise ValueError('Adaptive bandwidth has zero radius; increase the bandwidth.')
        spatial = (1 - np.minimum(distances / radius, 1)**2)**2
    if alpha == 1:
        return spatial
    attribute = np.asarray(data, dtype=float).reshape(-1)
    if attribute.size != len(coords):
        raise ValueError('Similarity weighting requires one attribute per observation.')
    neighbors = spatial > 0
    sd = np.std(attribute[neighbors])
    sd = sd if sd > 0 else 1e-5
    similarity = np.zeros(len(coords))
    similarity[neighbors] = np.exp(np.log(0.5) * ((attribute[neighbors] - attribute[i]) / sd)**2)
    return np.minimum(alpha * spatial + (1 - alpha) * similarity, 1)
