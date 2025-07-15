import numpy as np
from dataclasses import dataclass
from .decoder import T3OverflowCorrector
from typing import Union, Tuple
import xarray as xr

# class ScanConfig:
#     def __init__(
#         self,
#         lines: int = 512,
#         pixels: int = 512,
#         frames: int = 1,
#         line_accumulations: int = 1,
#         bidirectional: bool = False,
#         marker_roles: dict = None,
#     ):
#         self.lines = lines
#         self.pixels = pixels
#         self.frames = frames
#         self.line_accumulations = line_accumulations
#         self.bidirectional = bidirectional

#         self.marker_roles = marker_roles or {
#             "frame_start": "frame_start",
#             "forward_line": "forward_line_start",
#             "backward_line": "backward_line_start" if bidirectional else "",
#         }

#         self._validate()

#     def _validate(self):
#         required_roles = ["frame_start", "forward_line"]
#         if self.bidirectional:
#             required_roles.append("backward_line")
#         for role in required_roles:
#             label = self.marker_roles.get(role, "")
#             if not label:
#                 raise ValueError(f"Missing marker role: {role}")

#     def get_line_labels(self):
#         labels = [self.marker_roles["forward_line"]]
#         if self.bidirectional:
#             labels.append(self.marker_roles["backward_line"])
#         return labels

#     @classmethod
#     def from_dict(cls, d: dict) -> "ScanConfig":
#         return cls(
#             lines=d["lines"],
#             pixels=d["pixels"],
#             frames=d["frames"],
#             line_accumulations=d.get("line_accumulations", 1),
#             bidirectional=d.get("bidirectional", False),
#             marker_roles=d.get("marker_roles", None)
#         )
    
#     def to_dict(self) -> dict:
#         return {
#             "lines": self.lines,
#             "pixels": self.pixels,
#             "frames": self.frames,
#             "line_accumulations": self.line_accumulations,
#             "bidirectional": self.bidirectional,
#             "marker_roles": dict(self.marker_roles),  # ensure plain dict
#         }

#     def __repr__(self):
#         return f"<ScanConfig {self.to_dict()}>"
@dataclass
class LineSegment:
    start_nsync: int
    end_nsync: int
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



# class ScanConfig:
#     def __init__(
#         self,
#         lines: int = 512,
#         pixels: int = 512,
#         frames: int = 1,
#         max_channels: int = 64,
#         # line_accumulations: int = 1,
#         line_accumulations: tuple = (1,),
#         bidirectional: bool = False,
#         frame_start_marker: Union[int, Tuple[int, ...]] = (4,),
#         line_start_marker: Union[int, Tuple[int, ...]] = (1,),
#         line_stop_marker: Union[int, Tuple[int, ...]] = (2,)
#     ):
#         self.lines = lines
#         self.pixels = pixels
#         self.frames = frames
#         self.max_channels = max_channels
#         self.line_accumulations = line_accumulations
#         self.bidirectional = bidirectional

#         self.frame_start_marker = (
#             (frame_start_marker,) if isinstance(frame_start_marker, int)
#             else tuple(frame_start_marker)
#         )
#         self.line_start_marker = (
#             (line_start_marker,) if isinstance(line_start_marker, int)
#             else tuple(line_start_marker)
#         )
#         self.line_stop_marker = (
#             (line_stop_marker,) if isinstance(line_stop_marker, int)
#             else tuple(line_stop_marker)
#         )

        

    # TODO modify for number of sequences
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
            line_accumulations=d.get("line_accumulations", 1),
            bidirectional=d.get("bidirectional", False),
            frame_start_marker=d.get("frame_start_marker", ()),
            line_start_marker=d.get("line_start_marker", ()),
            line_stop_marker=d.get("line_stop_marker", ())
        )

    def __repr__(self):
        return f"ScanConfig({self.to_dict()})"



class ImageReconstructor:
    def __init__(self, config: ScanConfig):
        if not isinstance(config, ScanConfig):
            raise TypeError("ImageReconstructor requires a ScanConfig object")   
        self.config = config
        self.shape = (
            config.frames,
            config.lines * config.total_accumulations,
            config.pixels,
            config.max_channels
        )
        
        self.active_channels = set()

        # Initialize output arrays
        self.arrival_sum = np.zeros(self.shape, dtype=np.float32)  # for mean time
        self.photon_count = np.zeros(self.shape, dtype=np.uint32)

        # Rolling context
        self.partial_line_marker = None   # stores unmatched marker from previous chunk
        self.current_line_idx = 0
        # self.partial_frame_marker = None     # most recent frame start nsync
        self.current_frame_idx = 0               # current frame index
        self._pending_photons = np.empty((0,), dtype=np.ndarray)  # same dtype as events
        self._frame_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
        self._start_marker_nsyncs = np.empty((0,), dtype=np.ndarray)
        self._stop_marker_nsyncs = np.empty((0,), dtype=np.ndarray)


    def _build_line_segments(self, frame_markers: np.ndarray, start_markers: np.ndarray, stop_markers: np.ndarray = None) -> list[LineSegment]:
        frame_nsyncs = np.sort(frame_markers["nsync"])
        start_nsyncs = np.sort(start_markers["nsync"])
        self._frame_marker_nsyncs = np.append(self._frame_marker_nsyncs, frame_nsyncs, axis = 0)
        self._frame_marker_nsyncs = np.append(self._frame_marker_nsyncs, (np.inf,), axis = 0) # make sure there is always another sync marker
        next_frame_marker_nsync = self._frame_marker_nsyncs[self.current_frame_idx + 1]

        # if self.partial_frame_marker is not None:
        #     frame_nsyncs = np.insert(frame_nsyncs, 0, self.partial_frame_marker)
        #     self.partial_frame_marker = None

        if self.partial_line_marker is not None:
            start_nsyncs = np.insert(start_nsyncs, 0, self.partial_line_marker)
            self.partial_line_marker = None

        segments = []

        # if stop_markers is not None and len(stop_markers) > 0:
        stop_nsyncs = np.sort(stop_markers["nsync"])

        # Handle mismatched lengths
        count = min(len(start_nsyncs), len(stop_nsyncs))

        
            
        for i in range(count):

            start = start_nsyncs[i]
            end = stop_nsyncs[i]

            if start > next_frame_marker_nsync:
                self.current_frame_idx += 1
                next_frame_marker_nsync = self._frame_marker_nsyncs[self.current_frame_idx + 1]
                self.current_line_idx = 0
                

            # # Case: first time seeing a frame marker
            # if self._last_frame_marker is None or start >= self._next_frame_marker_nsync:
            #     self._frame_idx += 1
            #     self._last_frame_marker = start

            #     # Shift next frame marker
            #     if frame_markers_available:
            #         self._next_frame_marker_nsync = next(frame_marker_iter, np.inf)
            #     else:
            #         self._next_frame_marker_nsync = np.inf


            reversed_line = self.config.bidirectional and (self.current_line_idx % 2 == 1)

            segments.append(LineSegment(
                start_nsync=start,
                end_nsync=end,
                frame_idx=self.current_frame_idx,
                line_idx=self.current_line_idx,
                reversed=reversed_line,
            ))

            self.current_line_idx += 1
            

        # Carry over unpaired start if any
        if len(start_nsyncs) > count:
            self.partial_line_marker = start_nsyncs[count]

        self._frame_marker_nsyncs = self._frame_marker_nsyncs[:-1] # drop the inf element
        return segments
    

    def update(self, events: np.ndarray):
        # Filter non-marker photons
        if len(self._pending_photons) > 0:
            events = np.concatenate([self._pending_photons, events])
            self._pending_photons = np.empty((0,), dtype=events.dtype)

        


        photon_mask = (events["channel"] < 63) & (events["special"] == 0)
        photons = events[photon_mask]
        used_mask = np.zeros(len(photons), dtype=bool)
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
        for ch in np.unique(photons["channel"]):
        # Step 4: Assign photons to segments and accumulate
            for segment in line_segments:
            # Select photons within this segment
                in_segment = (photons["nsync"] >= segment.start_nsync) & (photons["nsync"] < segment.end_nsync)
                seg_photons = photons[in_segment]
                if len(seg_photons) == 0:
                    continue
                used_mask |= in_segment # mark used photons in the segment
                self.active_channels.update(np.unique(seg_photons["channel"])) # update the set of used channels
            
                seg_photons = seg_photons[seg_photons["channel"] == ch]

                # Calculate phase and pixel index
                duration = segment.end_nsync - segment.start_nsync
                if duration <= 0:
                    continue  # Invalid segment, skip

                phase = (seg_photons["nsync"] - segment.start_nsync) / duration
                # pos = 0.5 * (1 - np.cos(np.pi * phase)) # for harmonic scanner

                pixel_indices = np.floor(phase * self.config.pixels).astype(int)
                # pixel_indices = np.floor(pos * self.config.pixels).astype(int)
                pixel_indices = np.clip(pixel_indices, 0, self.config.pixels - 1)

                if segment.reversed:
                    pixel_indices = self.config.pixels - 1 - pixel_indices

                # Assign to image buffers
                f = segment.frame_idx
                l = segment.line_idx
                # l  = segment.line_idx % self.config.line_accumulations

                for i, pix in enumerate(pixel_indices):
                    self.arrival_sum[f, l, pix, ch] += seg_photons["dtime"][i]
                    self.photon_count[f, l, pix, ch] += 1
            
        self._pending_photons = photons[~used_mask]

    def finalize(self, return_xarray: bool = False):
        # if self.config.line_accumulations > 1:
            
        #     expected = self.config.lines * self.config.line_accumulations

        #     if self.current_line_idx != expected:
        #         print(f"[Warning] Expected {expected} lines, got {self.current_line_idx}. Possible data loss.")
        # out_shape = (
        #     self.config.frames,
        #     len(self.config.line_accumulations),
        #     self.config.lines,
        #     self.config.pixels,
        #     len(self.active_channels)
        # )
        # photon_count = np.zeros(shape=out_shape)

        active_channels = sorted(self.active_channels)
        photon_count = np.zeros(shape=(
            self.config.frames,
            len(self.config.line_accumulations),
            self.config.lines,
            self.config.pixels,
            max(self.active_channels) + 1
        ))
        # usable_lines = (self.current_line_idx // self.config.line_accumulations) * self.config.line_accumulations
        # Pattern [0, 1, 1, 1]
        pattern = np.repeat(np.arange(len(self.config.line_accumulations)), self.config.line_accumulations)
        tile_count = self.config.lines

        # Total pattern applied to all lines
        sequence_pattern = np.tile(pattern, tile_count)  # shape: (total_lines,)


        for accu_idx in range(len(self.config.line_accumulations)):
            seq_line_idx = np.where(sequence_pattern == accu_idx)[0]
            seq_photon_count = self.photon_count[:, seq_line_idx, :, :max(active_channels) + 1]

            reshaped = seq_photon_count.reshape(
                self.config.frames,
                self.config.line_accumulations[accu_idx],
                self.config.lines,
                self.config.pixels,
                max(active_channels) + 1
            )
            summed = reshaped.sum(axis = 1)
            photon_count[:, accu_idx, :, :, :] = summed

            # reshaped = seq_photon_count.reshape(
            #     self.config.frames,
            #     self.config.line_accumulations[accu_idx],
            #     self.config.lines,
            #     self.config.pixels,
            #     max(active_channels) + 1
            # )
            # summed = reshaped.sum(axis=1)

            # photon_count[:, accu_idx, :, :, :] = summed


        arrival_sum = np.zeros_like(photon_count, dtype=np.float64)
        for accu_idx in range(len(self.config.line_accumulations)):
            seq_line_idx = np.where(sequence_pattern == accu_idx)[0]
            seq_arrival_sum = self.arrival_sum[:, seq_line_idx, :, :max(active_channels) + 1]
            # arrival_sum[:, accu_idx, :, :, :] = seq_arrival_sum.reshape(
            #     self.config.frames,
            #     1,
            #     self.config.line_accumulations[accu_idx],
            #     self.config.lines,
            #     self.config.pixels,
            #     max(active_channels) + 1
            # ).sum(axis=2)
            reshaped = seq_arrival_sum.reshape(
                self.config.frames,
                self.config.line_accumulations[accu_idx],
                self.config.lines,
                self.config.pixels,
                max(active_channels) + 1
            )
            summed = reshaped.sum(axis=1)
            arrival_sum[:, accu_idx, :, :, :] = summed



        # mean_arrival = np.zeros(shape=photon_count.shape)        

        # photon_count = self.photon_count[:, :usable_lines, :, active_channels]
        # mean_arrival = self.arrival_sum[:, :usable_lines, :, active_channels]
        # # now reshape
        # photon_count = photon_count.reshape(
        #     self.config.frames,
        #     usable_lines // self.config.line_accumulations,
        #     self.config.line_accumulations,
        #     self.config.pixels
        # ).sum(axis=2)

        # mean_arrival = mean_arrival.reshape(
        #     self.config.frames,
        #     usable_lines // self.config.line_accumulations,
        #     self.config.line_accumulations,
        #     self.config.pixels
        # ).sum(axis=2)


        with np.errstate(divide='ignore', invalid='ignore'):
            mean_arrival = np.true_divide(arrival_sum, photon_count)
            mean_arrival[photon_count == 0] = 0  # set empty pixels to 0

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
            }

            return xr.Dataset(data, coords=coords)
        else:
            return ReconstructionResult(photon_count, mean_arrival)


    def _extract_markers(self, events, codes):
        return events[(events["special"] != 0) & np.isin(events["channel"], codes)]


@dataclass
class ReconstructionResult:
    photon_count: np.ndarray
    mean_arrival: np.ndarray

    # def _update_marker_context(self, frame_markers, line_markers):
    #     # Append to self.frame_markers and self.line_markers (keep sorted)
    #     pass

    # def _find_frame_index(self, nsync):
    #     # Binary search over self.frame_markers → return frame index
    #     pass

    # def _find_line_index(self, nsync, frame_idx):
    #     # Return line index within frame, and line start/end nsync
    #     pass

    # def _estimate_pixel_index(self, nsync, start, end):
    #     fraction = (nsync - start) / (end - start)
    #     return int(fraction * self.config.pixels)

