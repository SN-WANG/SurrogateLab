# Flow Sequence Visualization — PyVista / GPU-accelerated
# Author: Shengning Wang
#
# GPU setup (NVIDIA headless server):
#   export PYVISTA_FORCE_EGL=1    # select EGL backend before starting Python
#   export PYVISTA_OFF_SCREEN=true
# macOS / workstation with display: no extra env vars needed.

import os
import subprocess
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm
import pyvista as pv

from wsnet.utils.hue_logger import hue, logger


# ---------------------------------------------------------------------------
# Per-channel visualization metadata
# ---------------------------------------------------------------------------

def _channel_role(ch_idx: int, spatial_dim: int) -> str:
    """Return role tag: 'velocity', 'pressure', or 'temperature'."""
    if ch_idx < spatial_dim:
        return 'velocity'
    if ch_idx == spatial_dim:
        return 'pressure'
    return 'temperature'


_CMAP: dict = {
    'velocity':    'RdBu_r',   # diverging, zero = white — correct for ±Vx/Vy
    'pressure':    'plasma',   # perceptually uniform sequential (log scale)
    'temperature': 'plasma',   # same
    'error':       'Reds',     # sequential, always non-negative
}


class FlowVis:
    """
    GPU-accelerated CFD visualization engine using PyVista (VTK/OpenGL backend).

    Features:
    - Off-screen GPU rendering via VTK + EGL (NVIDIA headless servers)
    - log₁₀(P) colormap for high-pressure-ratio (10000:1) flows
    - Percentile-clipped (2 %–98 %) color limits: reveals small variations
      in near-constant fields such as temperature
    - Diverging colormap (RdBu_r) for velocity channels (zero = white)
    - Error column uses Reds (strictly non-negative)
    - No imageio-ffmpeg dependency: encodes via system ffmpeg subprocess

    Notes:
        - For MP4 output, system ffmpeg must be on PATH (or set FFMPEG_EXE env var).
        - For GIF output, PIL/Pillow is used (already a PyVista dependency).
    """

    FFMPEG_EXE: str = os.environ.get('FFMPEG_EXE', 'ffmpeg')

    def __init__(
        self,
        output_dir: Union[str, Path],
        spatial_dim: int = 2,
        fps: int = 30,
        theme: str = 'document',
        window_width: int = 2400,
        subplot_height: int = 250,
    ) -> None:
        """
        Args:
            output_dir:     Directory to save rendered animations.
            spatial_dim:    Spatial dimensionality (2 or 3).
            fps:            Frames per second for output video.
            theme:          PyVista theme ('document', 'paraview', 'dark').
            window_width:   Total window width in pixels (for comparison layout).
            subplot_height: Pixel height per channel row.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spatial_dim = spatial_dim
        self.fps = fps
        self.p_idx = spatial_dim        # pressure channel index
        self.window_width = window_width
        self.subplot_height = subplot_height

        pv.set_plot_theme(theme)

        if spatial_dim == 2:
            self.ch_names: List[str] = ['Vx', 'Vy', 'P', 'T']
        elif spatial_dim == 3:
            self.ch_names: List[str] = ['Vx', 'Vy', 'Vz', 'P', 'T']
        else:
            self.ch_names = [f'Field_{i}' for i in range(spatial_dim + 2)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_points(self, coords: Tensor) -> np.ndarray:
        """
        Convert coordinate tensor to (N, 3) float32 array for VTK.
        Pads Z = 0 for 2-D data.

        Args:
            coords: (N, spatial_dim)

        Returns:
            (N, 3) float32 numpy array.
        """
        pts = coords.detach().cpu().numpy().astype(np.float32)
        if self.spatial_dim == 2:
            pts = np.hstack([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)])
        return pts

    def _build_mesh(self, points: np.ndarray) -> pv.PolyData:
        """
        Build a Delaunay-triangulated mesh with alpha-shape filtering.

        Alpha is computed automatically as 2.5x the mean nearest-neighbor
        distance. This removes spurious cross-domain triangles at concave
        boundaries (e.g. pipe-vessel junctions) while preserving interior
        coverage.

        Falls back to raw point cloud if Delaunay produces no cells.

        Args:
            points: (N, 3) float32 array.

        Returns:
            pv.PolyData with triangulation (or raw points as fallback).
        """
        cloud = pv.PolyData(points)
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(points[:, :self.spatial_dim])
            dd, _ = tree.query(points[:, :self.spatial_dim], k=2)
            nn_dists = dd[:, 1]  # distance to nearest neighbor (skip self)
            alpha = float(np.mean(nn_dists)) * 2.5
            mesh = cloud.delaunay_2d(alpha=alpha)
            if mesh.n_cells == 0:
                logger.warning('delaunay_2d produced 0 cells, falling back to point cloud')
                return cloud
            return mesh
        except Exception as e:
            logger.warning(f'delaunay_2d failed ({e}), falling back to point cloud')
            return cloud

    def _compute_window_size(self, points: np.ndarray, num_cols: int,
                             num_channels: int) -> Tuple[int, int]:
        """
        Compute window size that matches the data aspect ratio.

        Args:
            points:       (N, 3) point coordinates.
            num_cols:     Number of subplot columns.
            num_channels: Number of subplot rows (channels).

        Returns:
            (width, height) in pixels.
        """
        x_range = float(points[:, 0].max() - points[:, 0].min()) or 1.0
        y_range = float(points[:, 1].max() - points[:, 1].min()) or 1.0
        aspect = x_range / y_range  # data aspect ratio per subplot

        col_width = self.window_width // num_cols
        subplot_h = max(int(col_width / aspect), 80)
        # clamp height to reasonable range
        subplot_h = min(subplot_h, self.subplot_height)
        subplot_h = max(subplot_h, 80)

        return self.window_width, subplot_h * num_channels

    def _preprocess(self, data: np.ndarray, ch_idx: int) -> np.ndarray:
        """
        Apply channel-specific visualization transform.

        Pressure: log₁₀(clip(P, 1.0)) — compresses 10000:1 dynamic range.
        Others:   identity.

        Args:
            data:   Array of any shape for a single channel.
            ch_idx: Channel index.

        Returns:
            Transformed array (same shape as input).
        """
        if ch_idx == self.p_idx:
            return np.log10(np.clip(data, 1.0, None))
        return data

    def _get_clim(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Percentile-clipped color limits from a full temporal stack.

        Uses 2nd–98th percentile so near-constant fields (e.g. temperature
        with ±0.3 K variation) still produce a meaningful color gradient.

        Args:
            data: Array of any shape (will be raveled).

        Returns:
            (vmin, vmax) float tuple.
        """
        flat = data.ravel()
        lo = float(np.percentile(flat, 2))
        hi = float(np.percentile(flat, 98))
        if abs(hi - lo) < 1e-9:          # truly constant field: add tiny epsilon
            center = (lo + hi) * 0.5
            lo, hi = center - 1e-6, center + 1e-6
        return lo, hi

    def _channel_cmap(self, ch_idx: int) -> str:
        """Colormap string for a given channel (not error column)."""
        return _CMAP[_channel_role(ch_idx, self.spatial_dim)]

    def _scalar_bar_title(self, ch_idx: int, col: int) -> str:
        """
        Scalar bar label for (channel_index, column).
        col: 0 = GT / single, 1 = Pred, 2 = Abs Error.
        """
        name = self.ch_names[ch_idx] if ch_idx < len(self.ch_names) else f'Ch{ch_idx}'
        if col == 2:
            # Error is always in physical units (Pa for pressure, not log₁₀)
            suffix = ' (Pa)' if ch_idx == self.p_idx else ''
            return f'|\u0394{name}|{suffix}'
        if ch_idx == self.p_idx:
            return 'P (log\u2081\u2080 Pa)'   # P (log₁₀ Pa)
        return name

    def _setup_camera(self, plotter: pv.Plotter, points: np.ndarray = None) -> None:
        """
        Fix camera view for the active subplot.

        For 2D data, sets a tight orthographic projection based on the
        bounding box of the point cloud with minimal padding.

        Args:
            plotter: Active plotter instance.
            points:  (N, 3) coordinates for tight framing. If None, uses reset_camera.
        """
        if self.spatial_dim == 2:
            plotter.view_xy()
            if points is not None:
                x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
                y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
                cx = (x_min + x_max) * 0.5
                cy = (y_min + y_max) * 0.5
                dx = (x_max - x_min) or 1.0
                dy = (y_max - y_min) or 1.0
                # 3% padding
                pad = max(dx, dy) * 0.03
                plotter.camera.focal_point = (cx, cy, 0.0)
                plotter.camera.position = (cx, cy, 1.0)
                plotter.camera.parallel_scale = max(dx, dy) * 0.5 + pad
                plotter.camera.parallel_projection = True
            else:
                plotter.reset_camera()
        else:
            plotter.view_isometric()
            plotter.reset_camera()

    def _encode_mp4(
        self,
        plotter: pv.Plotter,
        update_fn,
        seq_len: int,
        out_path: Path,
        desc: str,
    ) -> None:
        """
        Capture frames via plotter.screenshot() and encode to MP4 using
        system ffmpeg (stdin raw-video pipe). No imageio-ffmpeg dependency.

        Args:
            plotter:   Configured PyVista Plotter (off_screen=True).
            update_fn: Callable(t: int) — updates all mesh point_data for frame t.
            seq_len:   Total number of frames.
            out_path:  Output .mp4 path.
            desc:      tqdm progress bar description.
        """
        # First frame is already in the mesh (set during plotter setup).
        # Capture it to detect the actual pixel dimensions.
        first_frame = plotter.screenshot(return_img=True)  # (H, W, 3) uint8
        H, W = first_frame.shape[:2]

        # Ensure even dimensions (libx264 requirement)
        W_enc = W + (W % 2)
        H_enc = H + (H % 2)

        ffmpeg_cmd = [
            self.FFMPEG_EXE, '-y',
            '-framerate', str(self.fps),
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
            '-i', 'pipe:0',
            '-vf', f'pad={W_enc}:{H_enc}:0:0',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '22',
            str(out_path),
        ]
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        proc.stdin.write(first_frame.tobytes())
        for t in tqdm(range(1, seq_len), desc=desc, leave=False):
            update_fn(t)
            plotter.render()
            proc.stdin.write(plotter.screenshot(return_img=True).tobytes())

        proc.stdin.close()
        proc.wait()
        plotter.close()

        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {proc.returncode}. "
                f"Ensure ffmpeg is on PATH or set FFMPEG_EXE env var."
            )

    def _encode_gif(
        self,
        plotter: pv.Plotter,
        update_fn,
        seq_len: int,
        out_path: Path,
        desc: str,
    ) -> None:
        """
        Capture frames and write to GIF via imageio (uses PIL, no ffmpeg needed).

        Args:
            plotter:   Configured PyVista Plotter (off_screen=True).
            update_fn: Callable(t: int) — updates all mesh point_data for frame t.
            seq_len:   Total number of frames.
            out_path:  Output .gif path.
            desc:      tqdm progress bar description.
        """
        import imageio
        frames: List[np.ndarray] = []
        for t in tqdm(range(seq_len), desc=desc, leave=False):
            update_fn(t)
            plotter.render()
            frames.append(plotter.screenshot(return_img=True))
        plotter.close()
        imageio.mimwrite(str(out_path), frames, fps=self.fps, loop=0)

    def _animate(
        self,
        plotter: pv.Plotter,
        update_fn,
        seq_len: int,
        out_path: Path,
        file_format: str,
        desc: str,
    ) -> None:
        """Dispatch to the right encoder based on file_format."""
        if file_format == 'mp4':
            self._encode_mp4(plotter, update_fn, seq_len, out_path, desc)
        elif file_format == 'gif':
            self._encode_gif(plotter, update_fn, seq_len, out_path, desc)
        else:
            raise ValueError(f"Unsupported format '{file_format}'. Use 'mp4' or 'gif'.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def animate_sequence(
        self,
        sequence: Tensor,
        coords: Tensor,
        case_name: str,
        file_format: str = 'mp4',
        point_size: int = 5,
    ) -> None:
        """
        Render a single-sequence animation.

        Args:
            sequence:    (seq_len, N, C) field data in physical units.
            coords:      (N, spatial_dim) node coordinates.
            case_name:   Output filename prefix.
            file_format: 'mp4' or 'gif'.
            point_size:  Rendered point size in pixels.
        """
        seq_len, _, num_channels = sequence.shape
        seq_np = sequence.detach().cpu().numpy()
        points = self._prepare_points(coords)
        base_mesh = self._build_mesh(points)
        use_surface = base_mesh.n_cells > 0

        # Apply per-channel transforms; compute color limits from full stack.
        seq_vis = np.stack(
            [self._preprocess(seq_np[:, :, c], c) for c in range(num_channels)],
            axis=-1,
        )  # (seq_len, N, C)
        clims = [self._get_clim(seq_vis[:, :, c]) for c in range(num_channels)]
        for c in range(num_channels):
            if _channel_role(c, self.spatial_dim) == 'velocity':
                lo, hi = clims[c]
                if lo < 0 < hi:
                    vmax = max(abs(lo), abs(hi))
                    clims[c] = (-vmax, vmax)

        win_w, win_h = self._compute_window_size(points, 1, num_channels)

        plotter = pv.Plotter(
            shape=(num_channels, 1),
            off_screen=True,
            window_size=(win_w // 3, win_h),
        )

        sbar_args = {
            'height': 0.55, 'width': 0.07,
            'position_x': 0.88, 'position_y': 0.22,
            'vertical': True, 'fmt': '%.2e',
            'title_font_size': 9, 'label_font_size': 8,
        }

        meshes: List[pv.PolyData] = []
        for c in range(num_channels):
            plotter.subplot(c, 0)
            mesh = base_mesh.copy()
            mesh.point_data['scalar'] = seq_vis[0, :, c].astype(np.float32)

            add_kw = dict(
                scalars='scalar',
                cmap=self._channel_cmap(c),
                clim=clims[c],
                scalar_bar_args={**sbar_args, 'title': self._scalar_bar_title(c, 0)},
            )
            if use_surface:
                plotter.add_mesh(mesh, **add_kw)
            else:
                plotter.add_mesh(
                    mesh, **add_kw,
                    point_size=point_size,
                    render_points_as_spheres=True,
                )
            plotter.add_text(
                f'Field: {self.ch_names[c]}', font_size=10, position='upper_edge',
            )
            self._setup_camera(plotter, points)
            meshes.append(mesh)

        def _update(t: int) -> None:
            for c, mesh in enumerate(meshes):
                mesh.point_data['scalar'] = seq_vis[t, :, c].astype(np.float32)

        out_path = self.output_dir / f'{case_name}_seq.{file_format}'
        self._animate(plotter, _update, seq_len, out_path, file_format,
                      desc=f'Rendering {case_name}')
        logger.info(f'sequence animation saved to {hue.g}{out_path}{hue.q}')

    def animate_comparison(
        self,
        gt: Tensor,
        pred: Tensor,
        coords: Tensor,
        case_name: str,
        file_format: str = 'mp4',
        point_size: int = 5,
    ) -> None:
        """
        Render Ground Truth / Prediction / Abs Error comparison animation.

        Layout: num_channels rows x 3 columns.
        GT and Pred share the same color limits; Error has its own.
        Pressure is shown in log10(Pa) in GT/Pred columns and in raw Pa in
        the Error column (absolute residual preserves physical interpretation).

        Args:
            gt:          (seq_len, N, C) ground truth in physical units.
            pred:        (seq_len, N, C) prediction in physical units.
            coords:      (N, spatial_dim) node coordinates.
            case_name:   Output filename prefix.
            file_format: 'mp4' or 'gif'.
            point_size:  Rendered point size in pixels.
        """
        seq_len, _, num_channels = gt.shape
        err = torch.abs(gt - pred)

        gt_np   = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        err_np  = err.detach().cpu().numpy()
        points  = self._prepare_points(coords)
        base_mesh = self._build_mesh(points)
        use_surface = base_mesh.n_cells > 0

        # Log10(P) applied to GT/Pred only; error stays in physical units (Pa).
        gt_vis   = np.stack(
            [self._preprocess(gt_np[:, :, c],   c) for c in range(num_channels)], axis=-1,
        )
        pred_vis = np.stack(
            [self._preprocess(pred_np[:, :, c], c) for c in range(num_channels)], axis=-1,
        )

        # GT + Pred share one range; error gets its own.
        val_clims = [
            self._get_clim(
                np.concatenate([gt_vis[:, :, c].ravel(), pred_vis[:, :, c].ravel()])
            )
            for c in range(num_channels)
        ]
        for c in range(num_channels):
            if _channel_role(c, self.spatial_dim) == 'velocity':
                lo, hi = val_clims[c]
                if lo < 0 < hi:
                    vmax = max(abs(lo), abs(hi))
                    val_clims[c] = (-vmax, vmax)
        err_clims = [self._get_clim(err_np[:, :, c]) for c in range(num_channels)]

        win_w, win_h = self._compute_window_size(points, 3, num_channels)

        plotter = pv.Plotter(
            shape=(num_channels, 3),
            off_screen=True,
            window_size=(win_w, win_h),
        )

        sbar_args = {
            'height': 0.55, 'width': 0.07,
            'position_x': 0.88, 'position_y': 0.22,
            'vertical': True, 'fmt': '%.2e',
            'title_font_size': 9, 'label_font_size': 8,
        }

        col_titles = ['Ground Truth', 'Prediction', 'Abs Error']
        meshes: List[List[pv.PolyData]] = []  # meshes[channel][col]

        for c in range(num_channels):
            srcs  = [gt_vis,       pred_vis,       err_np]
            clims = [val_clims[c], val_clims[c],   err_clims[c]]
            cmaps = [self._channel_cmap(c), self._channel_cmap(c), _CMAP['error']]

            row: List[pv.PolyData] = []
            for col in range(3):
                plotter.subplot(c, col)
                mesh = base_mesh.copy()
                mesh.point_data['scalar'] = srcs[col][0, :, c].astype(np.float32)

                add_kw = dict(
                    scalars='scalar',
                    cmap=cmaps[col],
                    clim=clims[col],
                    scalar_bar_args={**sbar_args, 'title': self._scalar_bar_title(c, col)},
                )
                if use_surface:
                    plotter.add_mesh(mesh, **add_kw)
                else:
                    plotter.add_mesh(
                        mesh, **add_kw,
                        point_size=point_size,
                        render_points_as_spheres=True,
                    )
                if c == 0:
                    plotter.add_text(col_titles[col], font_size=11, position='upper_edge')
                self._setup_camera(plotter, points)
                row.append(mesh)

            meshes.append(row)

        def _update(t: int) -> None:
            for c in range(num_channels):
                meshes[c][0].point_data['scalar'] = gt_vis[t, :, c].astype(np.float32)
                meshes[c][1].point_data['scalar'] = pred_vis[t, :, c].astype(np.float32)
                meshes[c][2].point_data['scalar'] = err_np[t, :, c].astype(np.float32)

        out_path = self.output_dir / f'{case_name}_comp.{file_format}'
        self._animate(plotter, _update, seq_len, out_path, file_format,
                      desc=f'Rendering {case_name}')
        logger.info(f'comparison animation saved to {hue.g}{out_path}{hue.q}')

    def export_html(
        self,
        gt: Tensor,
        pred: Tensor,
        coords: Tensor,
        case_name: str,
        n_frames: int = 100,
        include_plotlyjs: str = 'cdn',
    ) -> None:
        """
        Export an interactive self-contained HTML visualization.

        Embeds all frame data as gzip+base64 uint8, generates a Plotly
        Scattergl visualization with channel switching, frame scrubbing,
        play/pause, and dynamic vs fixed color-scale modes.

        Args:
            gt:              (seq_len, N, C) ground truth in physical units.
            pred:            (seq_len, N, C) prediction in physical units.
            coords:          (N, 2) node coordinates.
            case_name:       Output filename prefix.
            n_frames:        Frames to uniformly sample from seq_len (default 100).
            include_plotlyjs: 'cdn' uses CDN link; True embeds JS inline (~4 MB larger).

        Output:
            {output_dir}/{case_name}_interactive.html
        """
        import gzip as _gzip
        import base64 as _base64
        import json as _json

        seq_len, N, C = gt.shape
        frame_indices = np.linspace(0, seq_len - 1, n_frames, dtype=int)

        gt_np   = gt.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()

        gt_sel   = gt_np[frame_indices]                                    # (n_frames, N, C)
        pred_sel = pred_np[frame_indices]
        err_sel  = np.abs(gt_np[frame_indices] - pred_np[frame_indices])   # physical units

        # Channel-specific preprocessing (log10 for P)
        gt_vis   = np.stack([self._preprocess(gt_sel[:, :, c],   c) for c in range(C)], axis=-1)
        pred_vis = np.stack([self._preprocess(pred_sel[:, :, c], c) for c in range(C)], axis=-1)
        err_vis  = err_sel.copy()   # no log preprocessing on error

        # --- Fixed global color limits (GT/Pred share; Error has its own) ---
        clims_gp  = []
        clims_err = []
        for c in range(C):
            combined = np.concatenate([gt_vis[:, :, c].ravel(), pred_vis[:, :, c].ravel()])
            lo = float(np.percentile(combined, 2))
            hi = float(np.percentile(combined, 98))
            if abs(hi - lo) < 1e-9:
                center = (lo + hi) * 0.5
                lo, hi = center - 1e-6, center + 1e-6
            if _channel_role(c, self.spatial_dim) == 'velocity' and lo < 0 < hi:
                vmax = max(abs(lo), abs(hi))
                lo, hi = -vmax, vmax
            clims_gp.append([lo, hi])
            clims_err.append(list(self._get_clim(err_vis[:, :, c])))

        clims_fixed = [clims_gp, clims_gp, clims_err]  # [panel=GT/Pred/Err][channel]

        # --- Dynamic per-frame color limits ---
        def _frame_clim(arr, c):
            lo = float(np.percentile(arr[:, c], 2))
            hi = float(np.percentile(arr[:, c], 98))
            if abs(hi - lo) < 1e-9:
                center = (lo + hi) * 0.5
                lo, hi = center - 1e-6, center + 1e-6
            return [lo, hi]

        dyn_gt   = [[_frame_clim(gt_vis[f],   c) for c in range(C)] for f in range(n_frames)]
        dyn_pred = [[_frame_clim(pred_vis[f],  c) for c in range(C)] for f in range(n_frames)]
        dyn_err  = [[_frame_clim(err_vis[f],   c) for c in range(C)] for f in range(n_frames)]

        # Synchronise GT/Pred dynamic clims per frame
        for f in range(n_frames):
            for c in range(C):
                lo = min(dyn_gt[f][c][0], dyn_pred[f][c][0])
                hi = max(dyn_gt[f][c][1], dyn_pred[f][c][1])
                if abs(hi - lo) < 1e-9:
                    center = (lo + hi) * 0.5
                    lo, hi = center - 1e-6, center + 1e-6
                if _channel_role(c, self.spatial_dim) == 'velocity' and lo < 0 < hi:
                    vmax = max(abs(lo), abs(hi))
                    lo, hi = -vmax, vmax
                dyn_gt[f][c]   = [lo, hi]
                dyn_pred[f][c] = [lo, hi]

        clims_dynamic = [dyn_gt, dyn_pred, dyn_err]

        # --- Quantise to uint8 (using fixed clims as reference) ---
        def _to_uint8(arr2d, vmin, vmax):
            rng = max(vmax - vmin, 1e-9)
            return np.clip((arr2d - vmin) / rng * 255, 0, 255).round().astype(np.uint8)

        panels_u8 = []
        for pidx, pvis in enumerate([gt_vis, pred_vis, err_vis]):
            p = np.zeros((n_frames, N, C), dtype=np.uint8)
            for c in range(C):
                vmin, vmax = clims_fixed[pidx][c]
                p[:, :, c] = _to_uint8(pvis[:, :, c], vmin, vmax)
            panels_u8.append(p)

        # Stack → (3, n_frames, N, C), flatten, gzip, base64
        raw_bytes  = np.stack(panels_u8, axis=0).tobytes()
        compressed = _gzip.compress(raw_bytes, compresslevel=6)
        encoded    = _base64.b64encode(compressed).decode('ascii')

        # --- Coordinates and metadata ---
        coords_np = coords.detach().cpu().numpy()
        x_coords  = coords_np[:, 0].tolist()
        y_coords  = coords_np[:, 1].tolist()

        ch_colormaps = [
            'RdBu_r' if _channel_role(c, self.spatial_dim) == 'velocity' else 'plasma'
            for c in range(C)
        ]
        ch_names = (self.ch_names[:C] if len(self.ch_names) >= C
                    else [f'Ch{i}' for i in range(C)])

        # --- Plotly.js source ---
        if include_plotlyjs is True:
            try:
                import plotly as _plotly
                js_path = (Path(_plotly.__file__).parent / 'package_data' / 'plotly.min.js')
                plotlyjs_tag = f'<script>{js_path.read_text()}</script>'
            except Exception:
                plotlyjs_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'
        else:
            plotlyjs_tag = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'

        meta = {
            'n_frames':       n_frames,
            'n_nodes':        N,
            'n_channels':     C,
            'frame_times':    frame_indices.tolist(),
            'ch_names':       ch_names,
            'ch_colormaps':   ch_colormaps,
            'clims_fixed':    clims_fixed,
            'clims_dynamic':  clims_dynamic,
            'p_idx':          self.p_idx,
        }

        # Dynamic section: Python fills in the data values
        data_js = (
            f'const ENCODED = "{encoded}";\n'
            f'const X_COORDS = {_json.dumps(x_coords)};\n'
            f'const Y_COORDS = {_json.dumps(y_coords)};\n'
            f'const META = {_json.dumps(meta)};\n'
            f'const FRAME_MAX = {n_frames - 1};\n'
        )

        # Static JavaScript (raw string — no brace escaping needed)
        static_js = r"""let DATA = null;
let currentChannel = 0;
let currentFrame   = 0;
let scaleMode      = 'fixed';
let playing        = false;
let animTimer      = null;

async function decodeData(encoded) {
    const bin   = atob(encoded);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const ds     = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(bytes);
    writer.close();
    const chunks = [];
    while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        if (value) chunks.push(value);
    }
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const out   = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
}

function extractColors(panel, frame, channel) {
    const {n_frames, n_nodes, n_channels} = META;
    const base   = (panel * n_frames * n_nodes + frame * n_nodes) * n_channels + channel;
    const stride = n_channels;
    let vmin, vmax;
    if (scaleMode === 'fixed') {
        [vmin, vmax] = META.clims_fixed[panel][channel];
    } else {
        [vmin, vmax] = META.clims_dynamic[panel][frame][channel];
    }
    const scale  = (vmax - vmin) / 255;
    const colors = new Array(n_nodes);
    for (let i = 0; i < n_nodes; i++) colors[i] = vmin + DATA[base + i * stride] * scale;
    return {colors, vmin, vmax};
}

const DIVS       = ['plot-gt', 'plot-pred', 'plot-err'];
const PANEL_TTLS = ['Ground Truth', 'Prediction', 'Abs Error'];
const ERROR_CMAP = 'Reds';

function getCmap(panel) {
    return panel < 2 ? META.ch_colormaps[currentChannel] : ERROR_CMAP;
}

function cbTitle(panel) {
    return panel < 2 ? META.ch_names[currentChannel] : '|\u0394Field|';
}

function initPlots() {
    const mSize = Math.max(2, Math.min(5, Math.round(2500 / META.n_nodes)));
    for (let p = 0; p < 3; p++) {
        const {colors, vmin, vmax} = extractColors(p, 0, currentChannel);
        const trace = {
            type: 'scattergl', mode: 'markers',
            x: X_COORDS, y: Y_COORDS,
            marker: {
                color: colors, colorscale: getCmap(p),
                cmin: vmin, cmax: vmax, size: mSize,
                colorbar: {
                    thickness: 12, len: 0.8,
                    title: {text: cbTitle(p), font: {size: 11, color: '#ccc'}},
                },
            },
            hovertemplate: 'x:%{x:.3f} y:%{y:.3f}<br>val:%{marker.color:.3e}<extra></extra>',
        };
        const layout = {
            title: {text: PANEL_TTLS[p], font: {size: 13, color: '#ccc'}},
            paper_bgcolor: '#1a1a2e', plot_bgcolor: '#1a1a2e',
            font: {color: '#ccc'},
            xaxis: {showgrid: false, zeroline: false, color: '#888'},
            yaxis: {scaleanchor: 'x', scaleratio: 1,
                    showgrid: false, zeroline: false, color: '#888'},
            margin: {l: 40, r: 60, t: 30, b: 30},
        };
        Plotly.newPlot(DIVS[p], [trace], layout, {responsive: true, displayModeBar: false});
    }
    updateInfo();
}

function updatePlots() {
    for (let p = 0; p < 3; p++) {
        const {colors, vmin, vmax} = extractColors(p, currentFrame, currentChannel);
        Plotly.restyle(DIVS[p], {
            'marker.color':              [colors],
            'marker.cmin':               [vmin],
            'marker.cmax':               [vmax],
            'marker.colorscale':         [getCmap(p)],
            'marker.colorbar.title.text':[cbTitle(p)],
        });
    }
    const ft = META.frame_times[currentFrame];
    document.getElementById('frame-label').textContent =
        `Frame ${currentFrame} / ${FRAME_MAX}  (t=${ft})`;
    document.getElementById('frame-slider').value = currentFrame;
    updateInfo();
}

function updateInfo() {
    const [vmin, vmax] = scaleMode === 'fixed'
        ? META.clims_fixed[0][currentChannel]
        : META.clims_dynamic[0][currentFrame][currentChannel];
    document.getElementById('info-box').textContent =
        `${META.ch_names[currentChannel]}  cmin:${vmin.toExponential(2)}  cmax:${vmax.toExponential(2)}`;
}

function setChannel(ch) {
    currentChannel = ch;
    document.querySelectorAll('.ch-btn').forEach((b, i) => b.classList.toggle('active', i === ch));
    updatePlots();
}

function setFrame(f)  { currentFrame = f; updatePlots(); }
function setScale(m)  { scaleMode = m;    updatePlots(); }

function togglePlay() {
    playing = !playing;
    document.getElementById('play-btn').innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
    if (playing) scheduleNext();
    else if (animTimer) { clearTimeout(animTimer); animTimer = null; }
}

function scheduleNext() {
    animTimer = setTimeout(() => {
        if (!playing) return;
        currentFrame = (currentFrame + 1) % META.n_frames;
        updatePlots();
        scheduleNext();
    }, 80);
}

decodeData(ENCODED).then(data => {
    DATA = data;
    document.getElementById('loading').style.display      = 'none';
    document.getElementById('controls').style.display     = 'flex';
    document.getElementById('plots-container').style.display = 'flex';
    initPlots();
}).catch(err => {
    document.getElementById('loading').textContent = 'Error: ' + err.message;
    console.error(err);
});
"""

        ch_buttons = ''.join(
            f'<button class="ch-btn{" active" if i == 0 else ""}" '
            f'onclick="setChannel({i})">{n}</button>'
            for i, n in enumerate(ch_names)
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{case_name} \u2014 CFD Flow Visualization</title>
{plotlyjs_tag}
<style>
body{{margin:0;background:#1a1a2e;color:#eee;font-family:monospace;overflow-x:hidden}}
#controls{{padding:8px 12px;background:#16213e;display:flex;flex-wrap:wrap;gap:12px;
           align-items:center;border-bottom:1px solid #333}}
.ch-btn{{padding:4px 10px;border:1px solid #555;background:#2a2a4a;color:#ccc;
         cursor:pointer;border-radius:4px;font-size:13px}}
.ch-btn.active{{background:#0f3460;border-color:#4a9eff;color:#fff}}
label{{cursor:pointer;font-size:13px}}
#frame-label{{font-size:12px;color:#aaa;min-width:130px}}
#frame-slider{{width:260px;cursor:pointer}}
#info-box{{font-size:11px;color:#9a9;padding:2px 8px;background:#111;border-radius:3px}}
#plots-container{{display:flex;width:100%;height:calc(100vh - 54px)}}
.plot-div{{flex:1;min-width:0}}
#loading{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
          font-size:18px;color:#4a9eff}}
.play-btn{{padding:4px 10px;background:#0f3460;border:1px solid #4a9eff;
           color:#fff;cursor:pointer;border-radius:4px;font-size:14px}}
</style>
</head>
<body>
<div id="loading">Loading data&#8230;</div>
<div id="controls" style="display:none">
  <div style="display:flex;gap:4px;align-items:center">
    <span style="font-size:12px;color:#aaa;margin-right:4px">Channel:</span>
    {ch_buttons}
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <span style="font-size:12px;color:#aaa">Scale:</span>
    <label><input type="radio" name="scale" value="fixed" checked
                  onchange="setScale(this.value)"> Fixed</label>
    <label><input type="radio" name="scale" value="dynamic"
                  onchange="setScale(this.value)"> Dynamic</label>
  </div>
  <div style="display:flex;align-items:center;gap:8px">
    <button class="play-btn" id="play-btn" onclick="togglePlay()">&#9654;</button>
    <span id="frame-label">Frame 0 / {n_frames - 1}</span>
    <input type="range" id="frame-slider" min="0" max="{n_frames - 1}" value="0"
           oninput="setFrame(parseInt(this.value))">
  </div>
  <div id="info-box">\u2014</div>
</div>
<div id="plots-container" style="display:none">
  <div id="plot-gt"   class="plot-div"></div>
  <div id="plot-pred" class="plot-div"></div>
  <div id="plot-err"  class="plot-div"></div>
</div>
<script>
{data_js}
{static_js}
</script>
</body>
</html>"""

        out_path = self.output_dir / f'{case_name}_interactive.html'
        out_path.write_text(html, encoding='utf-8')
        size_mb = out_path.stat().st_size / 1e6
        logger.info(f'interactive HTML saved to {hue.g}{out_path}{hue.q} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    # Smoke test with synthetic data
    spatial_dim = 2
    T, N, C = 20, 2000, spatial_dim + 2

    logger.info('Generating mock data...')
    mock_coords = torch.rand(N, spatial_dim)

    time_steps = torch.linspace(0, 4 * np.pi, T).view(T, 1, 1)
    wave = torch.sin(time_steps + mock_coords[:, 0].view(1, N, 1) * 5)
    mock_seq = wave.expand(T, N, C).clone()
    mock_seq[:, :, spatial_dim] = torch.rand(T, N) * 4e6 + 5e5   # P: 500 kPa – 4.5 MPa
    mock_seq[:, :, spatial_dim + 1] = 300.0 + torch.randn(T, N) * 0.5  # T: near-constant

    mock_pred = mock_seq * 0.95 + torch.randn_like(mock_seq) * 0.05

    vis = FlowVis(output_dir='vis_outputs', spatial_dim=spatial_dim, fps=10)
    vis.animate_sequence(mock_seq, mock_coords, case_name='demo_seq')
    vis.animate_comparison(mock_seq, mock_pred, mock_coords, case_name='demo_comp')
