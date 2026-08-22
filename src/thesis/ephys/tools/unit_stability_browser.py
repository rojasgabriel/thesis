"""Standalone browser for stored unit-stability selections."""

import argparse
import warnings

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtWidgets

warnings.filterwarnings("ignore", category=UserWarning, module="datajoint.plugin")

from labdata.schema import EphysRecording, SpikeSorting, UnitCount  # noqa: E402

from labdata_plugin.schema import UnitStability, UnitStabilityParams  # noqa: E402

FILTERS = (
    "All",
    "Fails amplitude",
    "Fails unimodality",
    "Fails both",
    "Passes selection",
)


def load_data(
    subject: str,
    session: str,
    liberal_criteria_id: int,
    unit_criteria_id: int,
    stability_param_id: int,
) -> tuple[dict[int, pd.DataFrame], dict[str, float]]:
    key = {
        "subject_name": subject,
        "session_name": session,
        "unit_criteria_id": unit_criteria_id,
        "unit_stability_param_id": stability_param_id,
    }
    UnitStability().populate(key)
    params = (UnitStabilityParams & key).fetch1()
    data = {}

    for master_key in (UnitStability & key).fetch("KEY"):
        sorting_key = {
            field: master_key[field]
            for field in (
                "subject_name",
                "session_name",
                "dataset_name",
                "probe_num",
                "parameter_set_num",
            )
        }
        rows = pd.DataFrame((UnitStability.Unit & master_key).fetch(as_dict=True))
        unit_keys = (UnitStability.Unit & master_key).fetch("KEY")
        raw_units = (SpikeSorting.Unit & unit_keys).fetch(
            "unit_id", "spike_times", "spike_amplitudes", as_dict=True
        )
        recording_duration, sampling_rate = (
            EphysRecording * EphysRecording.ProbeSetting & master_key
        ).fetch1("recording_duration", "sampling_rate")
        edges = np.linspace(
            0,
            float(recording_duration) * float(sampling_rate),
            params["n_time_windows"] + 1,
        )[1:-1]
        chunks = {}
        for unit in raw_units:
            window = np.digitize(unit["spike_times"], edges)
            chunks[unit["unit_id"]] = [
                unit["spike_amplitudes"][window == index]
                for index in range(params["n_time_windows"])
            ]
        rows["chunks"] = rows["unit_id"].map(chunks)
        rows = rows.sort_values("unit_id").reset_index(drop=True)

        liberal_count = (
            UnitCount & sorting_key & {"unit_criteria_id": liberal_criteria_id}
        ).fetch1("sua")
        rows.attrs["counts"] = (
            liberal_count,
            len(rows),
            int(rows["passes"].sum()),
        )
        data[master_key["probe_num"]] = rows

    if not data:
        raise RuntimeError(f"No unit stability rows for {subject} {session}")
    return data, params


class UnitStabilityBrowser(QtWidgets.QMainWindow):
    def __init__(
        self, data: dict[int, pd.DataFrame], params: dict, n_bins: int = 50
    ) -> None:
        super().__init__()
        self.data = data
        self.params = params
        self.n_bins = n_bins
        self.position = 0

        pg.setConfigOption("background", "white")
        pg.setConfigOption("foreground", "#222222")
        self.setWindowTitle("Unit stability browser")
        self.resize(1100, 700)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        plots = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plots)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.hist_plot = pg.PlotWidget()
        self.hist_plot.addLegend()
        self.count_plot = pg.PlotWidget()
        self.count_plot.setMaximumHeight(220)
        plot_layout.addWidget(self.hist_plot, stretch=1)
        plot_layout.addWidget(self.count_plot)
        layout.addWidget(plots, stretch=1)

        controls = QtWidgets.QGroupBox("Browse units")
        controls.setFixedWidth(270)
        form = QtWidgets.QFormLayout(controls)
        form.setContentsMargins(18, 24, 18, 18)
        form.setSpacing(14)
        self.probe_combo = QtWidgets.QComboBox()
        self.probe_combo.addItems([f"imec{probe}" for probe in sorted(data)])
        form.addRow("Probe", self.probe_combo)
        self.filter_combo = QtWidgets.QComboBox()
        self.filter_combo.addItems(FILTERS)
        form.addRow("Filter", self.filter_combo)
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
        self.probe_combo.currentTextChanged.connect(self._reset)
        self.filter_combo.currentTextChanged.connect(self._reset)
        previous.clicked.connect(lambda: self._step(-1))
        next_.clicked.connect(lambda: self._step(1))
        self._draw()

    @property
    def probe(self) -> int:
        return int(self.probe_combo.currentText().removeprefix("imec"))

    def _filtered(self) -> pd.DataFrame:
        rows = self.data[self.probe]
        filter_name = self.filter_combo.currentText()
        if filter_name == "Fails amplitude":
            return rows[~rows["passes_amplitude_stability"].astype(bool)]
        if filter_name == "Fails unimodality":
            return rows[~rows["passes_unimodality"].astype(bool)]
        if filter_name == "Fails both":
            return rows[
                ~rows["passes_amplitude_stability"].astype(bool)
                & ~rows["passes_unimodality"].astype(bool)
            ]
        if filter_name == "Passes selection":
            return rows[rows["passes"].astype(bool)]
        return rows

    def _reset(self) -> None:
        self.position = 0
        self._draw()

    def _step(self, amount: int) -> None:
        rows = self._filtered()
        if len(rows):
            self.position = (self.position + amount) % len(rows)
            self._draw()

    def _draw_counts(self) -> None:
        self.count_plot.clear()
        counts = self.data[self.probe].attrs["counts"]
        self.count_plot.addItem(
            pg.BarGraphItem(x=(0, 1, 2), height=counts, width=0.65, brush="#4c78a8")
        )
        self.count_plot.getAxis("bottom").setTicks(
            [[(0, "Liberal"), (1, "Standard"), (2, "Stable")]]
        )
        self.count_plot.setTitle(f"imec{self.probe} unit counts")
        self.count_plot.setLabel("left", "units")

    def _draw(self) -> None:
        rows = self._filtered()
        self.hist_plot.clear()
        self._draw_counts()
        if rows.empty:
            self.hist_plot.addItem(pg.TextItem("No units match this filter"))
            self.unit_label.setText("none")
            return

        self.position %= len(rows)
        row = rows.iloc[self.position]
        colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
        for index, chunk in enumerate(row["chunks"]):
            counts, edges = np.histogram(chunk, bins=self.n_bins)
            self.hist_plot.plot(
                np.repeat(edges, 2)[1:-1],
                np.repeat(counts, 2),
                pen=pg.mkPen(colors[index % len(colors)], width=2),
                name=f"Time {index + 1}",
            )
        status = "PASS" if row["passes"] else "FAIL"
        unit_id = int(row["unit_id"])
        self.unit_label.setText(f"{unit_id}  ({self.position + 1}/{len(rows)})")
        self.hist_plot.setTitle(
            f"Unit {unit_id}: {status} | amplitude drift={row['amplitude_drift']:.3f} "
            f"| dip q={row['dip_q_value']:.3g}"
        )
        self.hist_plot.setLabel("bottom", "spike amplitude")
        self.hist_plot.setLabel("left", "count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse stored unit stability metrics",
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
        "--liberal-criteria-id",
        type=int,
        default=0,
        help="Liberal unit criteria used for comparison",
    )
    optional.add_argument(
        "--unit-criteria-id", type=int, default=1, help="Unit quality criteria"
    )
    optional.add_argument(
        "--stability-param-id", type=int, default=0, help="Unit stability parameters"
    )
    optional.add_argument(
        "--bins", type=int, default=50, help="Amplitude histogram bin count"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, params = load_data(
        args.subject,
        args.session,
        args.liberal_criteria_id,
        args.unit_criteria_id,
        args.stability_param_id,
    )
    app = pg.mkQApp("Unit stability browser")
    browser = UnitStabilityBrowser(data, params, args.bins)
    browser.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
