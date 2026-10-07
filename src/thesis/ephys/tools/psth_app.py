import argparse
from typing import cast

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

PSTH_GAUSSIAN_SIGMA_S = 0.010
EVENT_LABELS = {
    "first_stim_times_s": "First stimulus onset",
    "stim_pulse_times_s": "Visual stimulus pulse",
    "audio_stim_times_s": "Auditory stimulus",
    "go_cue_times_s": "Go cue",
    "punish_wrong_times_s": "Wrong-choice punishment",
    "punish_early_times_s": "Early-withdrawal punishment",
}


def _event_label(event_name: str) -> str:
    """Return a readable label for one trial-table event column."""
    return EVENT_LABELS.get(
        event_name,
        event_name.removesuffix("_times_s")
        .removesuffix("_s")
        .replace("_", " ")
        .capitalize(),
    )


def _available_event_names(
    trials: pd.DataFrame, candidates: tuple[str, ...]
) -> tuple[str, ...]:
    """Return known alignment events that occur at least once."""
    return tuple(
        event_name
        for event_name in candidates
        if event_name in trials
        and (
            trials[event_name].notna().any()
            if event_name == "trial_start_s"
            else any(len(event_times) for event_times in trials[event_name])
        )
    )


def _align_spikes(
    event_times: np.ndarray,
    spike_times: np.ndarray,
    pre_seconds: float,
    post_seconds: float,
) -> list[np.ndarray]:
    """Align sorted spikes without scanning the full recording per event."""
    return [
        spike_times[
            np.searchsorted(
                spike_times, event_time - pre_seconds, side="left"
            ) : np.searchsorted(spike_times, event_time + post_seconds, side="right")
        ]
        - event_time
        for event_time in event_times
    ]


def _centered_smooth(counts: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Smooth rows around each bin without shifting the zero-lag sample."""
    center = len(kernel) // 2
    return np.vstack(
        [
            np.convolve(row, kernel, mode="full")[center : center + row.size]
            for row in counts
        ]
    )


def _gaussian_kernel(sigma_bins: float) -> np.ndarray:
    """Return a normalized Gaussian sampled symmetrically around zero."""
    support = np.arange(-np.ceil(3 * sigma_bins), np.ceil(3 * sigma_bins) + 1)
    kernel = np.exp(-0.5 * (support / sigma_bins) ** 2)
    return kernel / kernel.sum()


def _self_check() -> None:
    aligned = _align_spikes(
        np.array([1.0, 2.0]), np.array([0.5, 1.0, 1.5, 2.0, 2.5]), 0.5, 0.5
    )
    np.testing.assert_allclose(aligned[0], [-0.5, 0.0, 0.5])
    np.testing.assert_allclose(aligned[1], [-0.5, 0.0, 0.5])
    counts = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        _centered_smooth(counts, np.array([0.25, 0.5, 0.25])),
        [[0.0, 0.25, 0.5, 0.25, 0.0]],
    )
    kernel = _gaussian_kernel(1.0)
    np.testing.assert_allclose(kernel, kernel[::-1])
    np.testing.assert_allclose(kernel.sum(), 1.0)


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
    ) -> None:
        super().__init__()
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        binwidth_s = binwidth_ms / 1000
        self.binwidth_s = binwidth_s
        self.bin_edges = np.append(
            -np.arange(0, pre_seconds, binwidth_s)[1:][::-1],
            np.arange(0, post_seconds, binwidth_s),
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.psth_kernel = _gaussian_kernel(PSTH_GAUSSIAN_SIGMA_S / binwidth_s)

        from thesis.ephys.trials import ALIGNMENT_EV_COLUMNS, build_trial_table
        from thesis.ephys.units import fetch_unit_table

        unit_table = fetch_unit_table(
            subject,
            session,
            unit_criteria_id,
            stability_param_id,
            include_metrics=False,
        )
        self.units = dict(
            zip(unit_table["unit_id"], unit_table["spike_times_s"], strict=True)
        )
        if any(np.any(np.diff(spikes) < 0) for spikes in self.units.values()):
            raise ValueError("Unit spike times must be sorted")
        self.trials = build_trial_table(subject, session, include_frames=False)
        self.event_names = _available_event_names(
            self.trials, ("trial_start_s", *ALIGNMENT_EV_COLUMNS)
        )
        self.unit_ids = list(self.units)
        if not self.unit_ids:
            raise RuntimeError("No units pass the selected filters.")

        self.setWindowTitle(f"PSTH viewer — {subject} {session}")
        self.resize(1280, 760)
        self._build_ui()
        self._draw()

    def _build_ui(self) -> None:
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
        controls.setFixedWidth(320)
        form = QtWidgets.QFormLayout(controls)
        form.setContentsMargins(18, 24, 18, 18)
        form.setSpacing(14)
        form.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.event_combo = QtWidgets.QComboBox()
        for event_name in self.event_names:
            self.event_combo.addItem(_event_label(event_name), event_name)
        default_event_index = self.event_combo.findData("first_stim_times_s")
        self.event_combo.setCurrentIndex(max(0, default_event_index))
        form.addRow("Event", self.event_combo)

        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.addItems(("heatmap", "raster", "psth"))
        form.addRow("Plot", self.plot_combo)

        self.split_combo = QtWidgets.QComboBox()
        self.split_combo.addItems(self.SPLIT_OPTIONS)
        form.addRow("Split", self.split_combo)

        self.choice_only_checkbox = QtWidgets.QCheckBox("Choice trials only")
        form.addRow(self.choice_only_checkbox)

        self.smoothing_checkbox = QtWidgets.QCheckBox("Gaussian smoothing")
        self.smoothing_checkbox.setChecked(True)
        form.addRow(self.smoothing_checkbox)

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

        shortcut_hint = QtWidgets.QLabel(
            "← → units\n"
            "↑ ↓ events\n"
            "c → choice trials only\n"
            "m → smoothing\n"
            "1 → heatmap\n"
            "2 → raster\n"
            "3 → PSTH\n"
            "s → next split\n"
            "Esc → close"
        )
        shortcut_hint.setStyleSheet("color: #666666;")
        form.addRow("Keys", shortcut_hint)
        layout.addWidget(controls)

        controls.setStyleSheet(
            "QGroupBox { font-weight: 600; border: 1px solid #d5d5d5; "
            "border-radius: 8px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; "
            "padding: 0 5px; }"
            "QComboBox { min-height: 30px; padding: 2px 8px; }"
        )

        self._update_sort_options()
        self.event_combo.currentIndexChanged.connect(self._draw)
        self.plot_combo.currentTextChanged.connect(self._plot_type_changed)
        self.split_combo.currentTextChanged.connect(self._draw)
        self.choice_only_checkbox.toggled.connect(self._draw)
        self.smoothing_checkbox.toggled.connect(self._draw)
        self.sort_combo.currentTextChanged.connect(self._draw)
        self.unit_slider.valueChanged.connect(self._draw)
        self._build_shortcuts()

    def _build_shortcuts(self) -> None:
        shortcut_actions = (
            ("Left", lambda: self._step_unit(-1)),
            ("Right", lambda: self._step_unit(1)),
            ("Up", lambda: self._step_combo(self.event_combo, -1)),
            ("Down", lambda: self._step_combo(self.event_combo, 1)),
            ("C", self.choice_only_checkbox.toggle),
            ("M", self.smoothing_checkbox.toggle),
            ("1", lambda: self.plot_combo.setCurrentIndex(0)),
            ("2", lambda: self.plot_combo.setCurrentIndex(1)),
            ("3", lambda: self.plot_combo.setCurrentIndex(2)),
            ("S", lambda: self._step_combo(self.split_combo, 1)),
            ("Esc", self.close),
        )
        for key, action in shortcut_actions:
            QtGui.QShortcut(QtGui.QKeySequence(key), self).activated.connect(action)

    def _step_unit(self, step: int) -> None:
        self.unit_slider.setValue(self.unit_slider.value() + step)

    @staticmethod
    def _step_combo(combo: QtWidgets.QComboBox, step: int) -> None:
        if combo.count():
            combo.setCurrentIndex((combo.currentIndex() + step) % combo.count())

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
        self.smoothing_checkbox.setEnabled(plot_type == "psth")
        self.sort_combo.blockSignals(False)

    def _event_groups(self) -> list[tuple[str, np.ndarray]]:
        event_name = cast(str, self.event_combo.currentData())
        event_chunks = []
        trial_indices = []
        for trial_index, trial in self.trials.iterrows():
            if self.choice_only_checkbox.isChecked() and trial["response"] not in (
                -1,
                1,
            ):
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
            return [
                (
                    f"{len(event_times)} events from "
                    f"{len(np.unique(trial_idx))} trials",
                    event_times,
                )
            ]

        categories = self.trials[split_col].to_numpy()[trial_idx]
        grouped = pd.DataFrame(
            {
                "event_time": event_times,
                "trial_index": trial_idx,
                "category": categories,
            }
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
                f"{split_col} = {category}  "
                f"({len(group)} events from {group['trial_index'].nunique()} trials)",
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
        heatmap_rates = None
        if plot_type == "heatmap":
            heatmap_rates = [
                self._mean_event_rates(event_times) for _, event_times in groups
            ]
            if self.sort_combo.currentText() == "Peak latency":
                pooled_rates = np.average(
                    np.stack(heatmap_rates),
                    axis=0,
                    weights=[len(event_times) for _, event_times in groups],
                )
                sort_order = np.argsort(
                    np.argmax(pooled_rates[:, self.bin_centers >= 0], axis=1),
                    kind="stable",
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
                assert heatmap_rates is not None
                heatmaps.append(
                    (plot, *self._draw_heatmap(plot, heatmap_rates[index], sort_order))
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
        self, spike_times: np.ndarray, event_times: np.ndarray, smooth: bool
    ) -> np.ndarray:
        """Return event-by-bin rates for one unit; smoothing pads the window."""
        pad = len(self.psth_kernel) // 2 if smooth else 0
        step = self.binwidth_s * np.arange(1, pad + 1)
        edges = np.r_[
            self.bin_edges[0] - step[::-1], self.bin_edges, self.bin_edges[-1] + step
        ]
        extra = pad * self.binwidth_s
        rasters = _align_spikes(
            event_times,
            spike_times,
            self.pre_seconds + extra,
            self.post_seconds + extra,
        )
        counts = np.vstack([np.histogram(raster, edges)[0] for raster in rasters])
        if pad:
            counts = _centered_smooth(counts, self.psth_kernel)
            counts = counts[:, pad : pad + len(self.bin_centers)]
        return counts / self.binwidth_s

    def _mean_event_rates(self, event_times: np.ndarray) -> np.ndarray:
        """Return mean rates without retaining one matrix per event."""
        return np.asarray(
            [
                np.histogram(
                    np.concatenate(
                        _align_spikes(
                            event_times,
                            spike_times,
                            self.pre_seconds,
                            self.post_seconds,
                        )
                    ),
                    self.bin_edges,
                )[0]
                / len(event_times)
                / self.binwidth_s
                for spike_times in self.units.values()
            ]
        )

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
        self, plot: pg.PlotItem, rates: np.ndarray, order: np.ndarray | None
    ) -> tuple[pg.ImageItem, float]:
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
        rasters = _align_spikes(
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

        trial = np.repeat(np.arange(len(rasters)), [len(r) for r in rasters])
        x = np.repeat(np.concatenate(rasters), 2)
        y = np.column_stack([trial - 0.4, trial + 0.4]).ravel()
        plot.plot(x, y, pen=pg.mkPen("#222222", width=1), connect="pairs")
        plot.getViewBox().setYRange(-0.5, max(0.5, len(rasters) - 0.5), padding=0)
        self._prepare_plot(plot, "event")

    def _draw_psth(
        self, plot: pg.PlotItem, spike_times: np.ndarray, event_times: np.ndarray
    ) -> None:
        peth = self._peth(spike_times, event_times, self.smoothing_checkbox.isChecked())
        mean_rate = np.mean(peth, axis=0)
        sem_rate = (
            np.std(peth, axis=0, ddof=1) / np.sqrt(len(peth))
            if len(peth) > 1
            else np.full_like(mean_rate, np.nan)
        )
        band_pen = pg.mkPen(0, 0, 0, 64, width=1)
        upper = plot.plot(self.bin_centers, mean_rate + sem_rate, pen=band_pen)
        lower = plot.plot(self.bin_centers, mean_rate - sem_rate, pen=band_pen)
        plot.addItem(pg.FillBetweenItem(upper, lower, brush=(0, 0, 0, 51)))
        plot.plot(self.bin_centers, mean_rate, pen=pg.mkPen("#000000", width=2.5))
        self._prepare_plot(plot, "sp/s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse event-aligned neural activity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a", "--subject", required=True, default=argparse.SUPPRESS, help="Subject name"
    )
    parser.add_argument(
        "-s", "--session", required=True, default=argparse.SUPPRESS, help="Session name"
    )
    parser.add_argument(
        "--unit-criteria-id", type=int, default=1, help="Unit quality criteria"
    )
    parser.add_argument(
        "--stability-param-id",
        type=int,
        default=None,
        help="Also require units to pass this stability parameter set",
    )
    parser.add_argument(
        "--pre-seconds", type=float, default=0.1, help="Time shown before the event"
    )
    parser.add_argument(
        "--post-seconds", type=float, default=0.15, help="Time shown after the event"
    )
    parser.add_argument(
        "--binwidth-ms", type=int, default=10, help="PSTH bin width in milliseconds"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _self_check()
    app = pg.mkQApp("PSTH viewer")
    viewer = PSTHApp(**vars(args))
    viewer.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
