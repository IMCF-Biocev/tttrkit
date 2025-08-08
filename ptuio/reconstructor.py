import numpy as np
from dataclasses import dataclass
from .decoder import T3OverflowCorrector
from typing import Union, Tuple, Optional, Literal, Sequence
import xarray as xr
import copy
import time
from enum import Enum


@dataclass
class LineSegment:
    start_nsync: int
    stop_nsync: int
    frame_idx: int
    line_idx: int
    reversed: bool

AVAILABLE_OUTPUTS = [
    "photon_count",
    "mean_arrival_time",
    "phasor",
    "tcspc_histogram",
]


class ScanConfig:
    def __init__(
        self,
        lines: int = 512,
        pixels: int = 512,
        frames: int = 1,
        max_channels: int = 64,
        line_accumulations: tuple = (1,), #  > 1 dimension means the scanning is sequential
        bidirectional: bool = False,
        bidirectional_phase_shift: float = 0.0,
        frame_start_marker: Union[int, Tuple[int, ...]] = (4,),
        line_start_marker: Union[int, Tuple[int, ...]] = (1,),
        line_stop_marker: Union[int, Tuple[int, ...]] = (2,)
    ):
        self.lines = lines
        self.pixels = pixels
        self.frames = frames
        self.max_channels = max_channels
        self.bidirectional_phase_shift = bidirectional_phase_shift
        

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
    def __init__(
            self, config: ScanConfig, 
            roi_mask: Optional[np.ndarray] = None,
            outputs: Optional[Sequence[str]] = None,
            omega: float = 0.012, 
            tcspc_channels: int = 2 ** 15
            ):
                
                
                """
                Initialize an image reconstructor.

                Parameters:
                    config (ScanConfig): Scan configuration object.

                    roi_mask (ndarray, optional): Binary mask of shape (lines, pixels)
                        to restrict processing to specific regions.

                    outputs (list of str, optional): List of outputs to compute.
                        Each item must be one of the following:
                        - "photon_count"
                        - "mean_arrival_time"
                        - "phasor"
                        - "tcspc_histogram"
                        If None, all outputs are computed.

                    omega (float): Angular frequency for phasor computation.

                    tcspc_channels (int): Number of time bins for TCSPC histogram.
                """

        
        
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

                if outputs is None:
                    outputs = AVAILABLE_OUTPUTS.copy()
                # Validate requested outputs
                invalid = [o for o in outputs if o not in AVAILABLE_OUTPUTS]
                if invalid:
                    raise ValueError(f"Invalid output(s): {invalid}. Must be in {AVAILABLE_OUTPUTS}")

                self.requested_outputs = set(outputs)
                self._resolve_dependencies()
                # Initialize output arrays
                # self.arrival_sum = np.zeros(self.shape, dtype=np.float32)  # for mean time
                # self.photon_count = np.zeros(self.shape, dtype=np.uint32)
                # self.phasor_sum = np.zeros(self.shape, dtype=np.complex64)
                # self.histogram = np.zeros()
                # self.tcspc_channels = tcspc_channels
                # self.tcspc_hist = np.zeros((self.config.frames, self.config.max_channels, tcspc_channels), dtype=np.uint64)
                self.tcspc_channels = tcspc_channels
                if "arrival_sum" in self._required:
                    self.arrival_sum = np.zeros(self.shape, dtype=np.float32)
                if "photon_count" in self._required:
                    self.photon_count = np.zeros(self.shape, dtype=np.uint32)
                if "phasor_sum" in self._required:
                    self.phasor_sum = np.zeros(self.shape, dtype=np.complex64)
                if "tcspc_hist" in self._required:
                    self.tcspc_hist = np.zeros((self.config.frames, self.config.max_channels, tcspc_channels), dtype=np.uint64)  # existing shape logic

                # Rolling context
                self.partial_line_marker = None   # stores unmatched marker from previous chunk
                self.current_line_idx = 0
                self.current_frame_idx = 0               # current frame index
                self._pending_photons = np.empty((0,), dtype=np.ndarray)  # same dtype as events
                self._frame_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
                self._start_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
                self._stop_marker_nsyncs = np.empty((0,), dtype=np.ndarray)

                self._finished = False  # flag: stop processing when max frames reached
                
                self.stop_marker_phase = None
                self._stop_phase_computed = False
                self.line_duration = 0

                # ROI for masking
                self.roi_mask_stretched = None
                if roi_mask is not None:
                    self.roi_mask_stretched = self._stretch_roi_mask(roi_mask)

    def _resolve_dependencies(self):
        required = set()

        if "photon_count" in self.requested_outputs:
            required.add("photon_count")
        if "mean_arrival_time" in self.requested_outputs:
            required.update(["photon_count", "arrival_sum"])
        if "phasor" in self.requested_outputs:
            required.update(["photon_count", "phasor_sum"])
        if "tcspc_histogram" in self.requested_outputs:
            required.add("tcspc_hist")

        self._required = required

    @property
    def required_outputs(self):
        return list(self.requested_outputs)

    def get_available_outputs(self):
        return list(self.requested_outputs)

    def _stretch_roi_mask(self, base_mask: np.ndarray) -> np.ndarray:
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
        if self._finished:
            return []       
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
                if self.current_frame_idx >= self.config.frames:
                    self._finished = True
                    break  # stop collecting more segments
                next_frame_marker_nsync = self._frame_marker_nsyncs[self.current_frame_idx + 1]
                self.current_line_idx = 0

            if self.roi_mask_stretched is not None and not self.roi_mask_stretched[self.current_frame_idx,self.current_line_idx,:].any():
                self.current_line_idx += 1
                continue  # skip this segment entirely

            reversed_line = self.config.bidirectional and (self.current_line_idx % 2 == 1)

            start = start_nsyncs[i]
            stop = start + self.line_duration

            if reversed_line:
                interval = start_nsyncs[i+1] - start_nsyncs[i]  # safe since i < len-1
                shift = int(self.config.bidirectional_phase_shift * interval)
                start += shift
                stop += shift

            segments.append(LineSegment(
                # start_nsync = start_nsyncs[i],
                # stop_nsync = start_nsyncs[i] + self.line_duration,
                start_nsync = start,
                stop_nsync = stop,
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
        if self._finished:
            return

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
        segment_starts = np.array([s.start_nsync for s in segments])
        segment_ends = np.array([s.stop_nsync for s in segments])
        segment_index = np.searchsorted(segment_starts, photons["nsync"], side="right") - 1




        if photons['dtime'].max() >= self.tcspc_channels:
            print("\033[91mTCSPC channel overflow detected! Check the entire decay by incresing tcspc_channels.\033[0m")

        if photons['channel'].max() >= self.config.max_channels:
            print(f"\033[91mPhotons detected in channel {photons['channel'].max()} discarded!\033[0m")


        # Step 2: Filter photons that fall within valid segments
        valid = ((segment_index >= 0) & 
                 (segment_index < len(segment_starts)) & 
                 (photons['nsync'] < segment_ends[segment_index]) & 
                 (photons['dtime'] < self.tcspc_channels) & 
                 (photons['channel'] < self.config.max_channels))

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
        # np.add.at(self.arrival_sum, (frames, lines, pixels, channels), dtimes)
        # np.add.at(self.photon_count, (frames, lines, pixels, channels), 1)
        # np.add.at(self.phasor_sum, (frames, lines, pixels, channels), phasors)
        # np.add.at(self.tcspc_hist, (frames,channels,dtimes), 1)
        if "arrival_sum" in self._required:
            np.add.at(self.arrival_sum, (frames, lines, pixels, channels), dtimes)

        if "photon_count" in self._required:
            np.add.at(self.photon_count, (frames, lines, pixels, channels), 1)

        if "phasor_sum" in self._required:
            phasors = np.exp(1j * self.omega * dtimes)
            np.add.at(self.phasor_sum, (frames, lines, pixels, channels), phasors)

        if "tcspc_hist" in self._required:
            np.add.at(self.tcspc_hist, (frames, channels, dtimes), 1)

        pending_photons_mask = photons["nsync"] >= segment_ends[-1]            
        self._pending_photons = photons[pending_photons_mask]
        self.active_channels.update(np.unique(channels))



    def finalize(self):
        if self.partial_line_marker is not None:
            self._flush_final_line()

        data = {}
        active_channels = sorted(self.active_channels)
        channels = max(active_channels) + 1

        if "tcspc_histogram" in self.requested_outputs:
            self.tcspc_hist = self.tcspc_hist[:,:channels,:]
            data["tcspc_histogram"] = (("frame","channel","tcspc_channel"),self.tcspc_hist)

        if "photon_count" in self._required:
            photon_count = np.zeros(shape=(
                self.config.frames,
                len(self.config.line_accumulations),
                self.config.lines,
                self.config.pixels,
                max(self.active_channels) + 1
            ))

            pattern = np.repeat(np.arange(len(self.config.line_accumulations)), self.config.line_accumulations)

            # Total pattern applied to all lines
            sequence_pattern = np.tile(pattern, self.config.lines)  # shape: (total_lines,)

            
            lines = self.config.lines
            pixels = self.config.pixels


        if "arrival_sum" in self._required:
            arrival_sum = np.zeros_like(photon_count)

        if "phasor_sum" in self._required:
            phasor_sum = np.zeros_like(photon_count, dtype=np.complex64)

        if "photon_count" in self._required:

            for accu_idx in range(len(self.config.line_accumulations)):
                seq_line_idx = np.where(sequence_pattern == accu_idx)[0]
                accum = self.config.line_accumulations[accu_idx]
                for f in range(self.config.frames):
                    # seq_photon_count = self.photon_count[f, seq_line_idx, :, :max(active_channels) + 1]
                                    
                    summed_PC = self._reshape_and_sum(self.photon_count, f, seq_line_idx, lines, accum, pixels, channels)
                    photon_count[f, accu_idx, :, :, :] = summed_PC
                    if "arrival_sum" in self._required:
                        summed_AS = self._reshape_and_sum(self.arrival_sum, f, seq_line_idx, lines, accum, pixels, channels)
                        arrival_sum[f, accu_idx, :, :, :] = summed_AS
                    if "phasor_sum" in self._required:
                        summed_PS = self._reshape_and_sum(self.phasor_sum, f, seq_line_idx, lines, accum, pixels, channels)
                        phasor_sum[f, accu_idx, :, :, :] = summed_PS

        if "photon_count" in self.requested_outputs:                        
            data["photon_count"] = (("frame", "sequence", "line", "pixel", "channel"), photon_count)
                    
        if "mean_arrival_time" in self.requested_outputs:
            with np.errstate(divide='ignore', invalid='ignore'):
                mean_arrival = np.true_divide(arrival_sum, photon_count)
                mean_arrival[photon_count == 0] = 0  # set empty pixels to 0
            data["mean_arrival_time"] = (("frame", "sequence", "line", "pixel", "channel"), mean_arrival)
        

        if "phasor" in self.requested_outputs:

            # Normalize phasor
            with np.errstate(divide='ignore', invalid='ignore'):
                norm_phasor = np.true_divide(phasor_sum, photon_count)
                # norm_phasor[photon_count == 0] = 0
                norm_phasor[photon_count == 0] = np.nan + 1j * np.nan

            g = np.real(norm_phasor)
            s = np.imag(norm_phasor)
            data["phasor_g"] = (("frame", "sequence", "line", "pixel", "channel"), g)
            data["phasor_s"] = (("frame", "sequence", "line", "pixel", "channel"), s)

        all_coords = {
            "frame": np.arange(self.config.frames),
            "sequence": np.arange(len(self.config.line_accumulations)),
            "line": np.arange(self.config.lines),
            "pixel": np.arange(self.config.pixels),
            "channel": np.arange(channels),
            "tcspc_channel": np.arange(self.tcspc_channels),
        }

        # Determine used dimensions
        used_dims = set()
        for data_array in data.values():
            dims = data_array[0]  # First item in tuple is dims
            used_dims.update(dims)

        # Only keep relevant coords
        coords = {k: v for k, v in all_coords.items() if k in used_dims}

        return xr.Dataset(data, coords=coords)

    def _reshape_and_sum(self, array, f_idx, line_indices, lines, accum, pixels, channels):
        sliced = array[f_idx, line_indices, :, :channels]
        reshaped = sliced.reshape(lines, accum, pixels, channels)
        return reshaped.sum(axis=1)

    def _flush_final_line(self):
        if self._finished:
            return


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
    



