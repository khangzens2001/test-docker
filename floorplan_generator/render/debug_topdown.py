import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class DebugTopDownRenderer:
    """Render a lightweight top-down occupancy debug image for PLY floorplans."""

    def __init__(
        self,
        p2d,
        vertices_mm,
        grid_resolution_mm=10,
        occupancy_threshold=0,
        image_max_size=1600,
    ):
        self.p2d = np.asarray(p2d, dtype=float)
        self.vertices_mm = np.asarray(vertices_mm, dtype=float)
        self.grid_resolution_mm = float(grid_resolution_mm)
        self.occupancy_threshold = int(occupancy_threshold)
        self.image_max_size = int(image_max_size)

    def _rasterize(self):
        if self.p2d.ndim != 2 or self.p2d.shape[1] < 2 or len(self.p2d) == 0:
            raise ValueError("No 2D points available for debug top-down render.")
        if self.grid_resolution_mm <= 0:
            raise ValueError("Debug grid resolution must be greater than 0.")

        x_min, x_max = self.p2d[:, 0].min(), self.p2d[:, 0].max()
        y_min, y_max = self.p2d[:, 1].min(), self.p2d[:, 1].max()
        resolution = self.grid_resolution_mm / 1000.0

        width = int(np.ceil((x_max - x_min) / resolution)) + 2
        height = int(np.ceil((y_max - y_min) / resolution)) + 2
        grid = np.zeros((height, width), dtype=np.int32)

        idx_x = np.clip(((self.p2d[:, 0] - x_min) / resolution).astype(int), 0, width - 1)
        idx_y = np.clip(((self.p2d[:, 1] - y_min) / resolution).astype(int), 0, height - 1)
        np.add.at(grid, (idx_y, idx_x), 1)

        occupied = (grid > self.occupancy_threshold).astype(np.uint8) * 255
        return occupied, x_min, y_min, resolution

    def _polygon_pixels(self, x_min, y_min, resolution):
        if self.vertices_mm.ndim != 2 or self.vertices_mm.shape[1] < 2 or len(self.vertices_mm) == 0:
            return None

        poly = np.empty((len(self.vertices_mm), 2), dtype=np.int32)
        poly[:, 0] = np.round((self.vertices_mm[:, 0] / 1000.0 - x_min) / resolution).astype(np.int32)
        poly[:, 1] = np.round((self.vertices_mm[:, 1] / 1000.0 - y_min) / resolution).astype(np.int32)
        return poly

    def _fit_max_size(self, occupied, poly):
        if self.image_max_size <= 0:
            return occupied, poly
        height, width = occupied.shape[:2]
        max_side = max(height, width)
        if max_side <= self.image_max_size:
            return occupied, poly

        scale = self.image_max_size / max_side
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        y_idx = np.minimum((np.arange(new_height) / scale).astype(int), height - 1)
        x_idx = np.minimum((np.arange(new_width) / scale).astype(int), width - 1)
        resized = occupied[y_idx[:, None], x_idx]
        if poly is not None:
            poly = poly.astype(float) * scale
        return resized, poly

    def render(self, output_path):
        print("[debug] Rendering top-down occupancy check...")
        occupied, x_min, y_min, resolution = self._rasterize()
        poly = self._polygon_pixels(x_min, y_min, resolution)
        occupied, poly = self._fit_max_size(occupied, poly)

        height, width = occupied.shape[:2]
        dpi = 120
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(occupied, cmap="gray", origin="lower", interpolation="nearest")
        if poly is not None and len(poly) >= 3:
            closed = np.vstack((poly, poly[0]))
            ax.plot(closed[:, 0], closed[:, 1], color="red", linewidth=2)
            ax.plot(closed[:, 0], closed[:, 1], "o", color="red", markersize=3)
        ax.set_xlim(0, width - 1)
        ax.set_ylim(0, height - 1)
        ax.axis("off")
        fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"       Saved debug top-down: {output_path}")
