from ptuio.reader import TTTRReader
from ptuio.decoder import T3OverflowCorrector
from ptuio.reconstructor import ScanConfig
from ptuio.reconstructor import ImageReconstructor
import numpy as np
from matplotlib import cm
import copy
from typing import Optional
from scipy.optimize import curve_fit
from typing import Dict
import xarray as xr


def create_FLIM_image(mean_photon_arrival_time, intensity, colormap=cm.rainbow, 
                      lt_min=None, lt_max=None,
                      int_min=None, int_max=None):
    """
    Create an RGB FLIM image from lifetime and intensity data.

    Parameters:
    - mean_photon_arrival_time: 2D numpy array of lifetimes
    - intensity: 2D numpy array of photon counts
    - colormap: Matplotlib colormap (default: cm.rainbow)
    - lt_min: optional float, min lifetime for normalization
    - lt_max: optional float, max lifetime for normalization

    Returns:
    - FLIM_image: 3D numpy array (H, W, 3) with RGB values
    """

    # Validate shape
    if mean_photon_arrival_time.shape != intensity.shape:
        raise ValueError("Lifetime and intensity arrays must have the same shape")

    # Lifetime normalization
    if lt_min is None or lt_max is None:
        lt_min = np.nanmin(mean_photon_arrival_time)
        lt_max = np.nanmax(mean_photon_arrival_time)
    if lt_max == lt_min:
        raise ValueError(f"lt_max and lt_min must differ — got {lt_min}")

    # Intensity normalization with adjustable contrast
    if int_min is None or int_max is None:
        int_min = np.nanmin(intensity)
        int_max = np.nanmax(intensity)
    if int_max == int_min:
        raise ValueError("int_max and int_min must differ")

    LT_normalized = np.clip((mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1)
    LT_rgb = colormap(LT_normalized)[..., :3]  # Drop alpha
    intensity_normalized = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)

    return LT_rgb * intensity_normalized[..., np.newaxis]



def estimate_tcspc_bins(header_tags: dict, buffer: int = 10) -> int:
    rep_rate = header_tags.get("TTResult_SyncRate", 40e6)  # Hz
    resolution = header_tags.get("MeasDesc_Resolution", 5e-12)  # s
    bins = int(np.ceil(1 / resolution / rep_rate)) + buffer
    return bins




def estimate_bidirectional_shift(reader: TTTRReader, 
                                 config: ScanConfig,
                                 roi: Optional[np.ndarray] = None, 
                                 wrap: int = 1024,
                                 max_shift: float = .01, 
                                 steps: int = 11, 
                                 verbose: bool = True) -> float:
    """
    Estimate the optimal phase shift (as fraction of line duration) for backward lines 
    in bidirectional scanning.
    
    Args:
        reader: TTTRReader instance
        config: A ScanConfig instance.
        roi: 2D numpy array. If given, first frame is reconstructed (takes longer). If none is given, just one chunk is read for reconstruction
        max_shift: Maximum shift to try (±max_shift).
        steps: Number of shift steps to test.
        verbose: Whether to print progress.

    Returns:
        Best phase shift (float) in units of line duration (e.g., -0.015).
    """
    
    

    if not config.bidirectional:
        raise ValueError("ScanConfig must have bidirectional=True to estimate phase shift.")

    if verbose:
        print("Estimating bidirectional phase shift...")

    base_config = copy.deepcopy(config)
    base_config.frames = 1
    base_config.line_accumulations = (1,)
    base_config.lines = config.lines * config.line_accumulations[0]
    base_config.total_accumulations = 1

    line_bin = config.line_accumulations[0] * 2

    if roi is None:
        stretched_roi = None
    else:
        stretched_roi = ImageReconstructor(config=config)._stretch_roi_mask(roi)[0]
        row_mask = np.any(stretched_roi, axis=1)
        stretched_roi = np.broadcast_to(row_mask[:, None], stretched_roi.shape)


    shifts = np.linspace(config.bidirectional_phase_shift-max_shift,
                         config.bidirectional_phase_shift+max_shift, steps)
    scores = np.zeros_like(shifts)
    corrector = T3OverflowCorrector(wraparound=wrap)
    
    
    for i, shift in enumerate(shifts):
        
        # Clone config and apply shift
        test_config = copy.deepcopy(base_config)
        test_config.bidirectional_phase_shift = shift
        
        recon = ImageReconstructor(config=test_config,roi_mask=stretched_roi)

        if roi is None:
            chunk = reader.read(count=500_000)
            corrected_chunk = corrector.correct(chunk)
            recon.update(corrected_chunk)
            pc = xr.DataArray(recon.photon_count.astype(np.float32))
            pc = pc.rename({"dim_0" : "frame",
                "dim_1" : "line",
                "dim_2" : "pixel",
                "dim_3" : "channel"})
        else:
            for chunk in reader.iter_chunks():
                if recon._finished:
                    break
                corrected_chunk = corrector.correct(chunk)
                recon.update(corrected_chunk)
         # Reconstruct
            result = recon.finalize()
            pc = result.photon_count
            pc = pc.sum(dim = 'sequence')

        

        # pc = pc.transpose('frame', 'sequence', 'channel', 'line', 'pixel')
        pc = pc.sum(dim = 'channel')
        # pc = pc.isel(frame=0, sequence=0)
        pc = pc.isel(frame = 0)
        forward = pc[::2, :]
        backward = pc[1::2, :]
        forward = forward.coarsen(line = line_bin).sum()
        backward = backward.coarsen(line = line_bin).sum()
        
        # Ensure same number of lines
        num_pairs = min(forward.sizes['line'], backward.sizes['line'])
        fwd = forward.isel(line=slice(0, num_pairs))
        bwd = backward.isel(line=slice(0, num_pairs))

        # Mask out zero rows (xarray preserves dims, so we need numpy for row-wise masking)
        fwd_vals = fwd.values
        bwd_vals = bwd.values

        mask = ~((fwd_vals == 0).all(axis=1) | (bwd_vals == 0).all(axis=1))
        fwd_vals = fwd_vals[mask]
        bwd_vals = bwd_vals[mask]

        # Subtract mean along each line (axis=1)
        fwd_vals -= fwd_vals.mean(axis=1, keepdims=True)
        bwd_vals -= bwd_vals.mean(axis=1, keepdims=True)

        # Compute dot products (correlation at lag zero)
        score = np.sum(fwd_vals * bwd_vals)

        scores[i] = score

        if verbose:
            print(f"Shift {shift:.4f} → score {score:.2f}")

    # best_shift = shifts[np.argmax(scores)]

    best_shift, fit = fit_gaussian_peak(shifts, scores)

    if verbose:
        print(f"Best estimated shift: {best_shift:.5f}")

    return best_shift, np.stack((shifts,scores,fit))




def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + c

def fit_gaussian_peak(shifts: np.ndarray, scores: np.ndarray) -> Optional[float]:
    try:
        p0 = [scores.max() - scores.min(), shifts[np.argmax(scores)], 0.1* (shifts.max() - shifts.min()), scores.min()]
        popt, _ = curve_fit(gaussian, shifts, scores, p0=p0)
        fit = gaussian(shifts,*popt)
        return float(popt[1]), fit  # mu = estimated phase shift
        
    except RuntimeError:
        return None


# --- Optional Helpers ---

def marker_events(events: np.ndarray) -> np.ndarray:
    """Return only events where channel == 63 and special != 15 (non-overflow markers)."""
    return events[(events['channel'] < 63) & (events['special'] != 0)]

def overflow_events(events: np.ndarray) -> np.ndarray:
    """Return overflow marker events."""
    return events[(events['channel'] == 63) & (events['special'] != 0)]

def get_marker_distribution(events: np.ndarray) -> Dict[int, int]:
    """Returns a count of each special marker code."""
    mask = (events['channel'] < 63) & (events['special'] != 0)
    markers = events['channel'][mask]
    unique, counts = np.unique(markers, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

