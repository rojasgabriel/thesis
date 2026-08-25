"""Standalone browser for stored stimulus-response classifications."""

import argparse
import warnings

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from spks.event_aligned import population_peth

warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")


def _load_data(
    subject: str,
    session: str,
    unit_criteria_id: int,
    stim_response_param_id: int,
    stability_param_id: int | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    from labdata_plugin.schema import (
        StimulusResponse,
        StimulusResponseParam,
    )
    from thesis.ephys.events import fetch_session_events
    from thesis.ephys.units import fetch_unit_table

    key = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": unit_criteria_id,
        "stim_response_param_id": stim_response_param_id,
    }
    StimulusResponse().populate(key)
    params = (StimulusResponseParam & key).fetch1()
    rows = pd.DataFrame(
        (StimulusResponse.Unit & key & 'response_type != "none"').fetch(as_dict=True)
    ).sort_values(["response_type", "n_components", "unit_id"])
    unit_table = fetch_unit_table(
        subject, session, unit_criteria_id, stability_param_id
    )
    spikes_by_unit = dict(
        zip(unit_table["unit_id"], unit_table["spike_times_s"], strict=True)
    )
    rows = rows[rows["unit_id"].isin(spikes_by_unit)]
    if rows.empty:
        raise RuntimeError("No responsive units pass the selected filters")
    _, stimulus_pulses = fetch_session_events(subject, session)
    first_stimulus = stimulus_pulses.loc[
        stimulus_pulses["first_in_train"], "timestamp"
    ].to_numpy(dtype=float)
    unit_ids = list(spikes_by_unit)
    peth, bin_edges, _ = population_peth(
        all_spike_times=list(spikes_by_unit.values()),
        alignment_times=first_stimulus,
        pre_seconds=params["peth_pre_seconds"],
        post_seconds=params["peth_post_seconds"],
        binwidth_ms=params["binwidth_ms"],
    )
    peth = peth / (params["binwidth_ms"] / 1000)
    rows["peth_index"] = rows["unit_id"].map(
        {unit_id: index for index, unit_id in enumerate(unit_ids)}
    )
    return (
        rows.reset_index(drop=True),
        peth,
        (bin_edges[:-1] + bin_edges[1:]) / 2,
        params,
    )


class StimulusResponseApp(QtWidgets.QMainWindow):
    def __init__(
        self, rows: pd.DataFrame, peth: np.ndarray, bins: np.ndarray, params: dict
    ) -> None:
        super().__init__()
        self.rows = rows
        self.peth = peth
        self.bins = bins
        self.params = params
        self.position = 0

        pg.setConfigOption("background", "white")
        pg.setConfigOption("foreground", "#222222")
        self.setWindowTitle("Stimulus response browser")
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        plots = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.trace_plot = pg.PlotWidget()
        self.summary_plot = pg.PlotWidget()
        self.summary_plot.setMaximumHeight(220)
        plot_layout.addWidget(self.trace_plot, stretch=1)
        plot_layout.addWidget(self.summary_plot)
        layout.addWidget(plots, stretch=1)

        controls = QtWidgets.QGroupBox("Browse units")
        controls.setFixedWidth(250)
        form = QtWidgets.QFormLayout(controls)
        form.setContentsMargins(18, 24, 18, 18)
        form.setSpacing(14)
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(("all", "excited", "suppressed"))
        form.addRow("Response", self.filter_combo)
        self.unit_label = QtWidgets.QLabel()
        form.addRow("Unit", self.unit_label)
        buttons = QtWidgets.QWidget()
        button_layout = QtWidgets.QHBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        previous = QtWidgets.QPushButton("Previous")
        next_ = QtWidgets.QPushButton("Next")
        button_layout.addWidget(previous)
        button_layout.addWidget(next_)
        form.addRow(buttons)
        layout.addWidget(controls)

        controls.setStyleSheet(
            "QGroupBox { font-weight: 600; border: 1px solid #d5d5d5; "
            "border-radius: 8px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; "
            "padding: 0 5px; }"
            "QComboBox, QPushButton { min-height: 30px; padding: 2px 8px; }"
        )
        self.filter_combo.currentTextChanged.connect(self._set_filter)
        previous.clicked.connect(lambda: self._step(-1))
        next_.clicked.connect(lambda: self._step(1))
        self._draw_summary()
        self._draw()

    def _filtered(self) -> pd.DataFrame:
        response_filter = self.filter_combo.currentText()
        if response_filter == "all":
            return self.rows
        return self.rows[self.rows["response_type"] == response_filter]

    def _set_filter(self) -> None:
        self.position = 0
        self._draw()

    def _step(self, amount: int) -> None:
        rows = self._filtered()
        if len(rows):
            self.position = (self.position + amount) % len(rows)
            self._draw()

    def _draw_summary(self) -> None:
        self.summary_plot.clear()
        legend = self.summary_plot.addLegend()
        counts = (
            self.rows.groupby(["n_components", "response_type"])
            .size()
            .unstack(fill_value=0)
        )
        components = counts.index.to_numpy(dtype=float)
        width = 0.35
        for response_type, offset, color in (
            ("excited", -width / 2, "#f58518"),
            ("suppressed", width / 2, "#4c78a8"),
        ):
            values = counts.get(response_type, pd.Series(0, index=counts.index))
            bars = pg.BarGraphItem(
                x=components + offset,
                height=values.to_numpy(),
                width=width,
                brush=color,
            )
            self.summary_plot.addItem(bars)
            legend.addItem(bars, response_type)
        self.summary_plot.setTitle("Response shapes")
        self.summary_plot.setLabel("bottom", "components")
        self.summary_plot.setLabel("left", "units")
        self.summary_plot.getAxis("bottom").setTicks(
            [[(value, str(int(value))) for value in components]]
        )

    def _draw(self) -> None:
        rows = self._filtered()
        if rows.empty:
            self.trace_plot.clear()
            self.trace_plot.addItem(pg.TextItem("No units match this filter"))
            self.unit_label.setText("none")
            return
        self.position %= len(rows)
        row = rows.iloc[self.position]
        trace = self.peth[int(row["peth_index"])].mean(axis=0)
        baseline_window = (self.params["baseline_start"], self.params["baseline_end"])
        response_window = (self.params["response_start"], self.params["response_end"])
        baseline = trace[
            (self.bins >= baseline_window[0]) & (self.bins < baseline_window[1])
        ].mean()
        response_type = row["response_type"]
        color = "#f58518" if response_type == "excited" else "#4c78a8"

        self.trace_plot.clear()
        for window, brush in (
            (baseline_window, (76, 120, 168, 25)),
            (response_window, (245, 133, 24, 25)),
        ):
            self.trace_plot.addItem(
                pg.LinearRegionItem(values=window, movable=False, brush=brush, pen=None)
            )
        self.trace_plot.addLine(
            x=0, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DashLine)
        )
        self.trace_plot.addLine(
            y=baseline, pen=pg.mkPen("#888888", style=QtCore.Qt.PenStyle.DotLine)
        )
        self.trace_plot.plot(self.bins, trace, pen=pg.mkPen("#222222", width=2))
        self.trace_plot.plot(
            row["component_latencies_s"],
            row["component_rates"],
            pen=None,
            symbol="o",
            symbolBrush=color,
            symbolPen=color,
            symbolSize=8,
        )
        unit_id = int(row["unit_id"])
        self.unit_label.setText(f"{unit_id}  ({self.position + 1}/{len(rows)})")
        self.trace_plot.setTitle(
            f"Unit {unit_id}: {response_type}, {int(row['n_components'])} component(s)"
        )
        self.trace_plot.setLabel("bottom", "time from first stimulus (s)")
        self.trace_plot.setLabel("left", "firing rate (sp/s)")
        self.trace_plot.getAxis("bottom").enableAutoSIPrefix(False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse stimulus-responsive units",
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
        "--stim-response-param-id",
        type=int,
        default=0,
        help="Stimulus response parameters",
    )
    optional.add_argument(
        "--stability-param-id",
        type=int,
        default=None,
        help="Also require units to pass this stability parameter set",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows, peth, bins, params = _load_data(
        args.subject,
        args.session,
        args.unit_criteria_id,
        args.stim_response_param_id,
        args.stability_param_id,
    )
    app = pg.mkQApp("Stimulus response browser")
    browser = StimulusResponseApp(rows, peth, bins, params)
    browser.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
