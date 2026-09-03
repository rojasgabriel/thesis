import argparse
from typing import Literal, cast

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from scipy.stats import sem
from spks.event_aligned import align_raster_to_event, population_peth
from spks.utils import alpha_function

PSTH_ALPHA_RISE_S = 0.001
PSTH_ALPHA_DECAY_S = 0.025


class PSTHApp(QtWidgets.QMainWindow):
    SPLIT_OPTIONS = (
        "none",
        "stim_category",
        "visual_stim_rate_hz",
        "rewarded",
        "response",
        "prev_rewarded",
        "prev_response",
    )
    SORT_OPTIONS = {
        "heatmap": ("Peak latency", "Unit depth"),
        "raster": ("Trial order", "First-spike latency"),
        "psth": ("Not applicable",),
    }

    def __init__(
        self,
        subject: str,
        session: str,
        unit_criteria_id: int = 1,
        stability_param_id: int | None = None,
        pre_seconds: float = 0.1,
        post_seconds: float = 0.15,
        binwidth_ms: int = 10,
        plot_type: Literal["heatmap", "raster", "psth"] = "heatmap",
    ) -> None:
        super().__init__()
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.binwidth_ms = binwidth_ms
        binwidth_s = binwidth_ms / 1000
        decay_bins = PSTH_ALPHA_DECAY_S / binwidth_s
        self.psth_kernel = alpha_function(
            int(decay_bins * 15),
            t_rise=PSTH_ALPHA_RISE_S / binwidth_s,
            t_decay=decay_bins,
            srate=1 / binwidth_s,
        )

        from thesis.ephys.trials import ALIGNMENT_EV_COLUMNS, build_trial_table
        from thesis.ephys.units import fetch_unit_table

        unit_table = fetch_unit_table(
            subject, session, unit_criteria_id, stability_param_id
        )
        self.units = dict(
            zip(unit_table["unit_id"], unit_table["spike_times_s"], strict=True)
        )
        self.trials = build_trial_table(subject, session)
        self.event_names = ("trial_start_s", *ALIGNMENT_EV_COLUMNS)
        self.unit_ids = list(self.units)
        if not self.unit_ids:
            raise RuntimeError("No units pass the selected filters.")

        self.setWindowTitle(f"PSTH viewer — {subject} {session}")
        self.resize(1280, 760)
        self._build_ui(plot_type)
        self._draw()

    def _build_ui(self, plot_type: str) -> None:
        pg.setConfigOption("background", "white")
        pg.setConfigOption("foreground", "#222222")
        pg.setConfigOption("imageAxisOrder", "row-major")
        self.color_map = pg.colormap.get("viridis", source="matplotlib")

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        self.graphics = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics, stretch=1)

        controls = QtWidgets.QGroupBox("Display options")
        controls.setFixedWidth(280)
        form = QtWidgets.QFormLayout(controls)
        form.setContentsMargins(18, 24, 18, 18)
        form.setSpacing(14)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.event_combo = QtWidgets.QComboBox()
        self.event_combo.addItems(self.event_names)
        self.event_combo.setCurrentText("first_stim_times_s")
        form.addRow("Event", self.event_combo)

        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.addItems(("heatmap", "raster", "psth"))
        self.plot_combo.setCurrentText(plot_type)
        form.addRow("Plot", self.plot_combo)

        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.addItems(self.SPLIT_OPTIONS)
        form.addRow("Split", self.split_combo)

        self.sort_combo = QtWidgets.QComboBox()
        form.addRow("Sort", self.sort_combo)

        unit_control = QtWidgets.QWidget()
        unit_layout = QtWidgets.QHBoxLayout(unit_control)
        unit_layout.setContentsMargins(0, 0, 0, 0)
        self.unit_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.unit_slider.setRange(0, len(self.unit_ids) - 1)
        self.unit_label = QtWidgets.QLabel()
        self.unit_label.setMinimumWidth(45)
        unit_layout.addWidget(self.unit_slider)
        unit_layout.addWidget(self.unit_label)
        form.addRow("Unit", unit_control)
        layout.addWidget(controls)

        controls.setStyleSheet(
            "QGroupBox { font-weight: 600; border: 1px solid #d5d5d5; "
            "border-radius: 8px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; "
            "padding: 0 5px; }"
            "QComboBox { min-height: 30px; padding: 2px 8px; }"
        )

        self._update_sort_options()
        self.event_combo.currentTextChanged.connect(self._draw)
        self.plot_combo.currentTextChanged.connect(self._plot_type_changed)
        self.split_combo.currentTextChanged.connect(self._draw)
        self.sort_combo.currentTextChanged.connect(self._draw)
        self.unit_slider.valueChanged.connect(self._draw)

    def _plot_type_changed(self) -> None:
        self._update_sort_options()
        self._draw()

    def _update_sort_options(self) -> None:
        plot_type = self.plot_combo.currentText()
        self.sort_combo.blockSignals(True)
        self.sort_combo.clear()
        self.sort_combo.addItems(self.SORT_OPTIONS[plot_type])
        self.sort_combo.setEnabled(plot_type != "psth")
        self.unit_slider.setEnabled(plot_type != "heatmap")
        self.sort_combo.blockSignals(False)

    def _event_groups(self) -> list[tuple[str, np.ndarray]]:
        event_name = self.event_combo.currentText()
        event_chunks = []
        trial_indices = []
        for trial_index, trial in self.trials.iterrows():
            if trial["response"] not in (-1, 1):
                continue
            event_times = (
                [trial["trial_start_s"]]
                if event_name == "trial_start_s"
                else trial[event_name]
            )
            if event_times:
                event_chunks.append(np.asarray(event_times, dtype=float))
                trial_indices.extend([trial_index] * len(event_times))
        if not event_chunks:
            return []
        event_times = np.concatenate(event_chunks)
        trial_idx = np.asarray(trial_indices, dtype=int)

        split_col = self.split_combo.currentText()
        if split_col == "none":
            return [("", event_times)]

        categories = self.trials[split_col].to_numpy()[trial_idx]
        grouped = pd.DataFrame(
            {"event_time": event_times, "category": categories}
        ).dropna(subset=["category"])
        if split_col == "stim_category":
            grouped["category"] = pd.Categorical(
                grouped["category"],
                categories=("low_rate", "boundary", "high_rate"),
                ordered=True,
            )
        elif split_col in ("response", "prev_response"):
            grouped["category"] = pd.Categorical(
                grouped["category"], categories=(-1, 1), ordered=True
            )
            grouped = grouped.dropna(subset=["category"])
        return [
            (
                f"{split_col} = {category}  (n={len(group)})",
                group["event_time"].to_numpy(),
            )
            for category, group in grouped.groupby("category", sort=True, observed=True)
        ]

    def _draw(self) -> None:
        if not hasattr(self, "graphics"):
            return
        self.graphics.ci.clear()
        unit_id = self.unit_ids[self.unit_slider.value()]
        self.unit_label.setText(str(unit_id))
        groups = self._event_groups()
        if not groups:
            self.graphics.ci.addLabel("No events match this split.")
            return

        plot_type = self.plot_combo.currentText()
        sort_order = None
        if plot_type == "heatmap" and self.sort_combo.currentText() == "Peak latency":
            sort_order = self._peak_latency_order(
                np.concatenate([event_times for _, event_times in groups])
            )

        n_cols = min(3, len(groups))
        first_plot = None
        heatmaps = []
        for index, (title, event_times) in enumerate(groups):
            plot = self.graphics.ci.addPlot(
                row=index // n_cols, col=index % n_cols, title=title
            )
            if first_plot is None:
                first_plot = plot
            else:
                plot.setYLink(first_plot)
            if plot_type == "heatmap":
                heatmaps.append(
                    (plot, *self._draw_heatmap(plot, event_times, sort_order))
                )
            elif plot_type == "raster":
                self._draw_raster(plot, self.units[unit_id], event_times)
            else:
                self._draw_psth(plot, self.units[unit_id], event_times)
            if index:
                plot.hideAxis("left")

        if heatmaps:
            vmax = max(1.0, *(maximum for _, _, maximum in heatmaps))
            for _, image, _ in heatmaps:
                image.setLevels((0, vmax))
            heatmaps[-1][0].addColorBar(
                heatmaps[-1][1],
                colorMap=self.color_map,
                values=(0, vmax),
                label="sp/s",
            )

    def _peth(
        self,
        spike_times: list[np.ndarray],
        event_times: np.ndarray,
        kernel: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        peth, bin_edges, _ = population_peth(
            all_spike_times=spike_times,
            alignment_times=event_times,
            pre_seconds=self.pre_seconds,
            post_seconds=self.post_seconds,
            binwidth_ms=self.binwidth_ms,
            kernel=kernel,
        )
        return (
            peth / (self.binwidth_ms / 1000),
            (bin_edges[:-1] + bin_edges[1:]) / 2,
        )

    def _peak_latency_order(self, event_times: np.ndarray) -> np.ndarray:
        peth, bin_centers = self._peth(list(self.units.values()), event_times)
        rates = np.mean(peth, axis=1)
        return np.argsort(np.argmax(rates[:, bin_centers >= 0], axis=1), kind="stable")

    def _prepare_plot(self, plot: pg.PlotItem, ylabel: str) -> None:
        plot.getAxis("bottom").enableAutoSIPrefix(False)
        plot.setLabel("bottom", "time from event (s)")
        plot.setLabel("left", ylabel)
        plot.getViewBox().setXRange(-self.pre_seconds, self.post_seconds, padding=0)
        plot.addItem(
            pg.InfiniteLine(
                pos=0,
                angle=90,
                pen=pg.mkPen("#00a7b5", width=1, style=QtCore.Qt.PenStyle.DashLine),
            )
        )

    def _draw_heatmap(
        self, plot: pg.PlotItem, event_times: np.ndarray, order: np.ndarray | None
    ) -> tuple[pg.ImageItem, float]:
        peth, _ = self._peth(list(self.units.values()), event_times)
        rates = np.mean(peth, axis=1)
        unit_ids = self.unit_ids
        if order is not None:
            rates = rates[order]
            unit_ids = [unit_ids[index] for index in order]

        image = pg.ImageItem(rates.astype(np.float32), axisOrder="row-major")
        image.setRect(
            QtCore.QRectF(
                -self.pre_seconds,
                0,
                self.pre_seconds + self.post_seconds,
                len(unit_ids),
            )
        )
        image.setColorMap(self.color_map)
        plot.addItem(image)
        plot.getViewBox().invertY(True)
        plot.getViewBox().setYRange(0, len(unit_ids), padding=0)
        tick_step = max(1, len(unit_ids) // 10)
        plot.getAxis("left").setTicks(
            [
                [
                    (index + 0.5, str(unit_ids[index]))
                    for index in range(0, len(unit_ids), tick_step)
                ]
            ]
        )
        self._prepare_plot(
            plot, f"unit ID (sorted by {self.sort_combo.currentText().lower()})"
        )
        return image, float(np.nanmax(rates))

    def _draw_raster(
        self, plot: pg.PlotItem, spike_times: np.ndarray, event_times: np.ndarray
    ) -> None:
        rasters = align_raster_to_event(
            event_times,
            spike_times,
            self.pre_seconds,
            self.post_seconds,
        )
        if self.sort_combo.currentText() == "First-spike latency":
            first_spikes = [
                next((spike for spike in raster if spike > 0), np.inf)
                for raster in rasters
            ]
            rasters = [
                rasters[index] for index in np.argsort(first_spikes, kind="stable")
            ]

        n_spikes = sum(len(raster) for raster in rasters)
        x = np.full(n_spikes * 3, np.nan)
        y = np.full(n_spikes * 3, np.nan)
        offset = 0
        for trial, raster in enumerate(rasters):
            stop = offset + len(raster) * 3
            x[offset:stop:3] = raster
            x[offset + 1 : stop : 3] = raster
            y[offset:stop:3] = trial - 0.4
            y[offset + 1 : stop : 3] = trial + 0.4
            offset = stop
        plot.plot(x, y, pen=pg.mkPen("#222222", width=1), connect="finite")
        plot.getViewBox().setYRange(-0.5, max(0.5, len(rasters) - 0.5), padding=0)
        self._prepare_plot(plot, "trial")

    def _draw_psth(
        self, plot: pg.PlotItem, spike_times: np.ndarray, event_times: np.ndarray
    ) -> None:
        peth, bin_centers = self._peth(
            [spike_times], event_times, kernel=self.psth_kernel
        )
        mean_rate = np.mean(peth[0], axis=0)
        sem_rate = sem(peth[0], axis=0)
        band_pen = pg.mkPen(0, 0, 0, 64, width=1)
        upper = plot.plot(bin_centers, mean_rate + sem_rate, pen=band_pen)
        lower = plot.plot(bin_centers, mean_rate - sem_rate, pen=band_pen)
        plot.addItem(pg.FillBetweenItem(upper, lower, brush=(0, 0, 0, 51)))
        plot.plot(bin_centers, mean_rate, pen=pg.mkPen("#000000", width=2.5))
        self._prepare_plot(plot, "sp/s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse event-aligned neural activity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-a", "--subject", required=True, default=argparse.SUPPRESS, help="Subject name"
    )
    required.add_argument(
        "-s", "--session", required=True, default=argparse.SUPPRESS, help="Session name"
    )
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-h", "--help", action="help", help="Show this help message")
    optional.add_argument(
        "--unit-criteria-id", type=int, default=1, help="Unit quality criteria"
    )
    optional.add_argument(
        "--stability-param-id",
        type=int,
        default=None,
        help="Also require units to pass this stability parameter set",
    )
    optional.add_argument(
        "--pre-seconds", type=float, default=0.1, help="Time shown before the event"
    )
    optional.add_argument(
        "--post-seconds", type=float, default=0.15, help="Time shown after the event"
    )
    optional.add_argument(
        "--binwidth-ms", type=int, default=10, help="PSTH bin width in milliseconds"
    )
    optional.add_argument(
        "--plot-type",
        choices=("heatmap", "raster", "psth"),
        default="heatmap",
        help="Initial plot type",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = pg.mkQApp("PSTH viewer")
    viewer = PSTHApp(
        subject=args.subject,
        session=args.session,
        unit_criteria_id=args.unit_criteria_id,
        stability_param_id=args.stability_param_id,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        binwidth_ms=args.binwidth_ms,
        plot_type=cast(Literal["heatmap", "raster", "psth"], args.plot_type),
    )
    viewer.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
