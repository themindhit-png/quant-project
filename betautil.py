"""Memory-light rolling statistics on (T, N) return panels, computed in column chunks (float64 inside, float32 out)."""
import numpy as np


def rolling_std(ret, W=168, min_periods=72, chunk=100):
    """Rolling std (ddof=1) of hourly returns ignoring NaN, NaN where fewer than min_periods valid — matches
    pandas DataFrame.rolling(W, min_periods).std() to ~1e-6 relative."""
    T, N = ret.shape; out = np.full((T, N), np.nan, dtype=np.float32)
    for c0 in range(0, N, chunk):
        R = ret[:, c0:c0 + chunk].astype(np.float64); ok = np.isfinite(R); X = np.where(ok, R, 0.0)
        cs = np.cumsum(X, axis=0); css = np.cumsum(X * X, axis=0); cn = np.cumsum(ok, axis=0)
        z = np.zeros((1, X.shape[1]))
        S = cs[W - 1:] - np.vstack([z, cs[:-W]]); SS = css[W - 1:] - np.vstack([z, css[:-W]]); n = cn[W - 1:] - np.vstack([z, cn[:-W]])
        with np.errstate(all='ignore'):
            var = (SS - S * S / n) / (n - 1); sd = np.sqrt(np.maximum(var, 0.0))
        sd[n < min_periods] = np.nan
        out[W - 1:, c0:c0 + chunk] = sd.astype(np.float32)
        del R, ok, X, cs, css, cn, S, SS, n, var, sd
    return out


def rolling_beta(ret, jb, W=720, chunk=100):
    T, N = ret.shape
    y = np.where(np.isfinite(ret[:, jb]), ret[:, jb], 0.0).astype(np.float64)
    cy = np.cumsum(y); cyy = np.cumsum(y * y)
    Sy = np.full(T, np.nan); Syy = np.full(T, np.nan)
    Sy[W - 1:] = cy[W - 1:] - np.concatenate([[0.0], cy[:-W]]); Syy[W - 1:] = cyy[W - 1:] - np.concatenate([[0.0], cyy[:-W]])
    var = Syy / W - (Sy / W) ** 2
    B = np.full((T, N), np.nan, dtype=np.float32)
    for c0 in range(0, N, chunk):
        X = np.where(np.isfinite(ret[:, c0:c0 + chunk]), ret[:, c0:c0 + chunk], 0.0).astype(np.float64)
        cx = np.cumsum(X, axis=0); cxy = np.cumsum(X * y[:, None], axis=0)
        z = np.zeros((1, X.shape[1]))
        Sx = np.full_like(X, np.nan); Sxy = np.full_like(X, np.nan)
        Sx[W - 1:] = cx[W - 1:] - np.vstack([z, cx[:-W]]); Sxy[W - 1:] = cxy[W - 1:] - np.vstack([z, cxy[:-W]])
        cov = Sxy / W - (Sx / W) * (Sy[:, None] / W)
        with np.errstate(all='ignore'):
            B[:, c0:c0 + chunk] = (cov / var[:, None]).astype(np.float32)
        del X, cx, cxy, Sx, Sxy, cov
    return B
