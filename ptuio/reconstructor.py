import numpy as np
from dataclasses import dataclass
from .decoder import T3OverflowCorrector
from typing import Union, Tuple, Optional
import xarray as xr

@dataclass
class LineSegment:
    start_nsync: int
    stop_nsync: int
    frame_idx: int
    line_idx: int
    reversed: bool

class ScanConfig:
    def __init__(
        self,
        lines: int = 512,
        pixels: int = 512,
        frames: int = 1,
        max_channels: int = 64,
        line_accumulations: tuple = (1,), #  > 1 dimension means the scanning is sequential
        bidirectional: bool = False,
        frame_start_marker: Union[int, Tuple[int, ...]] = (4,),
        line_start_marker: Union[int, Tuple[int, ...]] = (1,),
        line_stop_marker: Union[int, Tuple[int, ...]] = (2,)
    ):
        self.lines = lines
        self.pixels = pixels
        self.frames = frames
        self.max_channels = max_channels
        

        # Normalize line_accumulations to tuple
        if isinstance(line_accumulations, int):
            self.line_accumulations = (line_accumulations,)
        else:
            self.line_accumulations = tuple(line_accumulations)

        self.num_sequences = len(self.line_accumulations)
        self.bidirectional = bidirectional
        self.total_accumulations = sum(self.line_accumulations)

        self.frame_start_marker = (
            (frame_start_marker,) if isinstance(frame_start_marker, int)
            else tuple(frame_start_marker)
        )
        self.line_start_marker = (
            (line_start_marker,) if isinstance(line_start_marker, int)
            else tuple(line_start_marker)
        )
        self.line_stop_marker = (
            (line_stop_marker,) if isinstance(line_stop_marker, int)
            else tuple(line_stop_marker)
        )



        

    def to_dict(self):
        return {
            "lines": self.lines,
            "pixels": self.pixels,
            "frames": self.frames,
            "line_accumulations": self.line_accumulations,
            "bidirectional": self.bidirectional,
            "frame_start_marker": self.frame_start_marker,
            "line_start_marker": self.line_start_marker,
            "line_stop_marker": self.line_stop_marker,
        }

    # TODO modify for number of sequences
    @classmethod
    def from_dict(cls, d):
        return cls(
            lines=d["lines"],
            pixels=d["pixels"],
            frames=d["frames"],
            line_accumulations=d.get("line_accumulations", (1,)),
            bidirectional=d.get("bidirectional", False),
            frame_start_marker=d.get("frame_start_marker", ()),
            line_start_marker=d.get("line_start_marker", ()),
            line_stop_marker=d.get("line_stop_marker", ())
        )

    def __repr__(self):
        return f"ScanConfig({self.to_dict()})"



class ImageReconstructor:
    def __init__(self, config: ScanConfig, roi_mask: Optional[np.ndarray] = None, omega: float = 0.012, tcspc_channels: Optional[int] = None):
        if not isinstance(config, ScanConfig):
            raise TypeError("ImageReconstructor requires a ScanConfig object")   
        self.config = config
        self.shape = (
            config.frames,
            config.lines * config.total_accumulations,
            config.pixels,
            config.max_channels
        )
        self.omega = omega

        self.active_channels = set()

        # Initialize output arrays
        self.arrival_sum = np.zeros(self.shape, dtype=np.float32)  # for mean time
        self.photon_count = np.zeros(self.shape, dtype=np.uint32)
        self.phasor_sum = np.zeros(self.shape, dtype=np.complex64)
        # self.histogram = np.zeros()

        # Rolling context
        self.partial_line_marker = None   # stores unmatched marker from previous chunk
        self.current_line_idx = 0
        self.current_frame_idx = 0               # current frame index
        self._pending_photons = np.empty((0,), dtype=np.ndarray)  # same dtype as events
        self._frame_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
        self._start_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
        self._stop_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
        
        self.stop_marker_phase = None
        self._stop_phase_computed = False
        self.line_duration = 0

        # ROI for masking
        self.roi_mask_stretched = None
        if roi_mask is not None:
            self.roi_mask_stretched = self.stretch_roi_mask(roi_mask)

    # def set_roi_mask(self, mask: np.ndarray):
    #     if mask.shape != (self.config.lines, self.config.pixels):
    #         raise ValueError(f"ROI mask must have shape ({self.config.lines}, {self.config.pixels})")
    #     self.roi_mask = mask.astype(bool)

        self.tcspc_channels = tcspc_channels
        self.tcspc_hist = None

        if tcspc_channels is not None:
            self.tcspc_hist = np.zeros(tcspc_channels, dtype=np.uint64)

    def stretch_roi_mask(self, base_mask: np.ndarray) -> np.ndarray:
        if base_mask.shape != (self.config.lines, self.config.pixels):
            raise ValueError(f"Base ROI mask must have shape ({self.config.lines}, {self.config.pixels})")

        # Expand to (frames, sequences, lines, pixels)
        stretched_mask = np.zeros(
            (self.config.frames,
            self.config.lines * self.config.total_accumulations,
            self.config.pixels),
            dtype=bool
        )

        for f_idx in range(self.config.frames):
            stretched_mask[f_idx,:,:] = np.repeat(base_mask, repeats=self.config.total_accumulations, axis=0)

        return stretched_mask


 
    def _build_line_segments(self, frame_markers: np.ndarray, start_markers: np.ndarray, stop_markers: np.ndarray = None) -> list[LineSegment]:
        frame_nsyncs = frame_markers["nsync"]
        start_nsyncs = start_markers["nsync"]
        stop_nsyncs = stop_markers["nsync"] # only used to calculate stop phase and line duration
        self._frame_marker_nsyncs = np.append(self._frame_marker_nsyncs, frame_nsyncs, axis = 0)
        self._frame_marker_nsyncs = np.append(self._frame_marker_nsyncs, (np.inf,), axis = 0) # make sure there is always another sync marker
        next_frame_marker_nsync = self._frame_marker_nsyncs[self.current_frame_idx + 1]

        
        
        if self.partial_line_marker is not None:
            start_nsyncs = np.insert(start_nsyncs, 0, self.partial_line_marker)
            self.partial_line_marker = None


        if not self._stop_phase_computed:
            self.compute_stop_phase(start_nsyncs, stop_nsyncs)

        segments = []

        # Construct segments
        for i in range(len(start_nsyncs) - 1):


            if start_nsyncs[i] > next_frame_marker_nsync:
                self.current_frame_idx += 1
                next_frame_marker_nsync = self._frame_marker_nsyncs[self.current_frame_idx + 1]
                self.current_line_idx = 0

            if self.roi_mask_stretched is not None and not self.roi_mask_stretched[self.current_frame_idx,self.current_line_idx,:].any():
                self.current_line_idx += 1
                continue  # skip this segment entirely
            reversed_line = self.config.bidirectional and (self.current_line_idx % 2 == 1)

            


            segments.append(LineSegment(
                start_nsync = start_nsyncs[i],
                stop_nsync = start_nsyncs[i] + self.line_duration,
                frame_idx = self.current_frame_idx,
                line_idx = self.current_line_idx,
                reversed = reversed_line,
            ))

            self.current_line_idx += 1

        if len(start_nsyncs) > 0:
            self.partial_line_marker = start_nsyncs[-1] 
        self._frame_marker_nsyncs = self._frame_marker_nsyncs[:-1]
        return segments
    
 

    def update(self, events: np.ndarray):
        # Filter non-marker photons
        if len(self._pending_photons) > 0:
            events = np.concatenate([self._pending_photons, events])
            self._pending_photons = np.empty((0,), dtype=events.dtype)
        photon_mask = (events["channel"] < 63) & (events["special"] == 0)
        photons = events[photon_mask]

        
        # Extract frame markers
        frame_markers = self._extract_markers(events, self.config.frame_start_marker)

        # Extract line markers
        start_markers = self._extract_markers(events, self.config.line_start_marker)
        stop_markers = (
            self._extract_markers(events, self.config.line_stop_marker)
            if self.config.line_stop_marker else None
        )

        # Step 3: Build line segments from markers
        line_segments = self._build_line_segments(frame_markers, start_markers, stop_markers)

        self._assign_photons_to_segments(photons, line_segments)


    def _assign_photons_to_segments(self, photons: np.ndarray, segments: list) -> None:
        if len(segments) == 0 or photons.size == 0:
            return

        # Step 1: Segment edges based on start_nsync + line duration
        # segment_edges = np.array([s.start_nsync for s in segments] + [segments[-1].start_nsync + self.line_duration])
        # segment_index = np.searchsorted(segment_edges, photons["nsync"], side="right") - 1
        segment_starts = np.array([s.start_nsync for s in segments])
        segment_ends = np.array([s.stop_nsync for s in segments])
        segment_index = np.searchsorted(segment_starts, photons["nsync"], side="right") - 1

        

        # Step 2: Filter photons that fall within valid segments
        valid = (segment_index >= 0) & (segment_index < len(segment_starts)) & (photons['nsync'] < segment_ends[segment_index])
        if np.count_nonzero(valid) == 0:
            return

        segment_index = segment_index[valid]
        photons_in_segments = photons[valid]

        # Step 3: Extract segment metadata
        segment_info = np.array(
            [(s.frame_idx, s.line_idx, s.reversed) for s in segments],
            dtype=[("frame", int), ("line", int), ("reversed", bool)]
        )

        frames = segment_info["frame"][segment_index]
        lines = segment_info["line"][segment_index]
        reversed_flags = segment_info["reversed"][segment_index]

        # Step 4: Calculate pixel indices

        phase = (photons_in_segments["nsync"] - segment_starts[segment_index]) / (segment_ends[segment_index] - segment_starts[segment_index])
        pixels = np.floor(phase * self.config.pixels).astype(int)

        # Step 5: Handle reversed lines
        pixels = np.where(reversed_flags, self.config.pixels - 1 - pixels, pixels)

        # Step 6: Filter valid pixels
        valid_pixels = (pixels >= 0) & (pixels < self.config.pixels)
        if np.count_nonzero(valid_pixels) == 0:
            return
        
        if self.roi_mask_stretched is not None:
            in_roi = self.roi_mask_stretched[frames,lines,pixels]
            valid_pixels = valid_pixels & in_roi


        pixels = pixels[valid_pixels]
        frames = frames[valid_pixels]
        lines = lines[valid_pixels]
        channels = photons_in_segments["channel"][valid_pixels]
        dtimes = photons_in_segments["dtime"][valid_pixels]
        phasors = np.exp(1j * self.omega * dtimes)

        # Step 7: Accumulate
        np.add.at(self.arrival_sum, (frames, lines, pixels, channels), dtimes)
        np.add.at(self.photon_count, (frames, lines, pixels, channels), 1)
        np.add.at(self.phasor_sum, (frames, lines, pixels, channels), phasors)
        pending_photons_mask = photons["nsync"] >= segment_ends[-1]            
        self._pending_photons = photons[pending_photons_mask]
        self.active_channels.update(np.unique(channels))
        
        # add to tcspc histogram
        if self.tcspc_hist is not None:
            dtimes = photons_in_segments["dtime"]
            np.add.at(self.tcspc_hist, dtimes, 1)


    def finalize(self, return_xarray: bool = False):
        if self.partial_line_marker is not None:
            self._flush_final_line()

        active_channels = sorted(self.active_channels)
        
        photon_count = np.zeros(shape=(
            self.config.frames,
            len(self.config.line_accumulations),
            self.config.lines,
            self.config.pixels,
            max(self.active_channels) + 1
        ))
        arrival_sum = np.zeros_like(photon_count)
        phasor_sum = np.zeros_like(photon_count, dtype=np.complex64)
        pattern = np.repeat(np.arange(len(self.config.line_accumulations)), self.config.line_accumulations)

        # Total pattern applied to all lines
        sequence_pattern = np.tile(pattern, self.config.lines)  # shape: (total_lines,)

        channels = max(active_channels) + 1
        lines = self.config.lines
        pixels = self.config.pixels

        for accu_idx in range(len(self.config.line_accumulations)):
            seq_line_idx = np.where(sequence_pattern == accu_idx)[0]
            accum = self.config.line_accumulations[accu_idx]
            for f in range(self.config.frames):
                # seq_photon_count = self.photon_count[f, seq_line_idx, :, :max(active_channels) + 1]

                                
                summed_PC = self._reshape_and_sum(self.photon_count, f, seq_line_idx, lines, accum, pixels, channels)
                summed_AS = self._reshape_and_sum(self.arrival_sum, f, seq_line_idx, lines, accum, pixels, channels)
                summed_PS = self._reshape_and_sum(self.phasor_sum, f, seq_line_idx, lines, accum, pixels, channels)

                photon_count[f, accu_idx, :, :, :] = summed_PC
                arrival_sum[f, accu_idx, :, :, :] = summed_AS
                phasor_sum[f, accu_idx, :, :, :] = summed_PS


        with np.errstate(divide='ignore', invalid='ignore'):
            mean_arrival = np.true_divide(arrival_sum, photon_count)
            mean_arrival[photon_count == 0] = 0  # set empty pixels to 0

        # Normalize phasor
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_phasor = np.true_divide(phasor_sum, photon_count)
            # norm_phasor[photon_count == 0] = 0
            norm_phasor[photon_count == 0] = np.nan + 1j * np.nan

        g = np.real(norm_phasor)
        s = np.imag(norm_phasor)

        if self.tcspc_hist is not None:
            used = np.nonzero(self.tcspc_hist)[0]
            if len(used) > 0:
                self.tcspc_hist = self.tcspc_hist[:used[-1] + 1]


        if return_xarray:
            coords = {
            "frame": np.arange(photon_count.shape[0]),
            "sequence": np.arange(photon_count.shape[1]),
            "line": np.arange(photon_count.shape[2]),
            "pixel": np.arange(photon_count.shape[3]),
            "channel": np.arange(photon_count.shape[4])
            }

            data = {
                "photon_count": (("frame", "sequence", "line", "pixel", "channel"), photon_count),
                "mean_photon_arrival_time": (("frame", "sequence", "line", "pixel", "channel"), mean_arrival),
                "phasor_g": (("frame", "sequence", "line", "pixel", "channel"), g),
                "phasor_s": (("frame", "sequence", "line", "pixel", "channel"), s)
            }


            if self.tcspc_hist is not None:
                    coords["tcspc_channel"] = np.arange(self.tcspc_hist.shape[0])
                    data["tcspc_histogram"] = ("tcspc_channel", self.tcspc_hist)

            return xr.Dataset(data, coords=coords)
        
        else:
            return ReconstructionResult(photon_count, mean_arrival, g, s)

    def _reshape_and_sum(self, array, f_idx, line_indices, lines, accum, pixels, channels):
        sliced = array[f_idx, line_indices, :, :channels]
        reshaped = sliced.reshape(lines, accum, pixels, channels)
        return reshaped.sum(axis=1)

    def _flush_final_line(self):

        final_segment = LineSegment(
            start_nsync= self.partial_line_marker,
            stop_nsync= self.partial_line_marker + self.line_duration,
            frame_idx=self.current_frame_idx,
            line_idx=self.current_line_idx,
            reversed=self.config.bidirectional and (self.current_line_idx % 2 == 1),
        )

        self._assign_photons_to_segments(self._pending_photons, [final_segment])
        self._pending_photons = np.empty((0,), dtype=self._pending_photons.dtype)
        self.partial_line_marker = None


    def _extract_markers(self, events, codes):
        return events[(events["special"] != 0) & np.isin(events["channel"], codes)]

    def compute_stop_phase(self, start_nsyncs: np.ndarray, stop_nsyncs: np.ndarray, default_phase = 0.75) -> float:
        pair_count = min(len(stop_nsyncs), len(start_nsyncs) - 1)
        durations = stop_nsyncs[:pair_count] - start_nsyncs[:pair_count]
        intervals = start_nsyncs[1:1+pair_count] - start_nsyncs[:pair_count]            # Get intervals between consecutive start markers
        phase_estimates = durations / intervals

        # Filter out unrealistic values (e.g. <0 or >1.5)
        good = (phase_estimates > 0) & (phase_estimates < 1.2)
        if np.count_nonzero(good) < 5:
            print("Too few valid stop marker timings, using default.")
            self.stop_marker_phase = default_phase
            self.line_duration = int(np.median(intervals) * default_phase)
        else:
            self.stop_marker_phase = float(np.median(phase_estimates[good]))
            self.line_duration = int(np.median(durations))
            
        self._stop_phase_computed = True
        return self.stop_marker_phase, self.line_duration


@dataclass
class ReconstructionResult:
    photon_count: np.ndarray
    mean_arrival: np.ndarray


