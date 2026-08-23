"""Interactive survey-map browser."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtWidgets


def load_survey_map(path: Path) -> pd.DataFrame:
    if path.suffix == ".txt":
        return pd.read_csv(path, sep="\t")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError("Survey map must be a .txt or .csv file")


class SurveyMapApp(QtWidgets.QMainWindow):
    def __init__(self, survey_map: pd.DataFrame) -> None:
        super().__init__()
        self.survey_map = survey_map
        self.vmin = 1
        self.color_map = pg.colormap.get("magma_r", source="matplotlib")

        pg.setConfigOption("background", "white")
        pg.setConfigOption("foreground", "#222222")
        self.setWindowTitle("Survey map browser")
        self.resize(850, 850)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        self.plot = pg.PlotWidget()
        self.color_legend = pg.GradientLegend(size=(20, 160), offset=(-20, -20))
        self.color_legend.setGradient(self.color_map.getGradient())
        layout.addWidget(self.plot, stretch=1)

        controls = QtWidgets.QGroupBox("Display options")
        controls.setFixedWidth(245)
        form = QtWidgets.QFormLayout(controls)
        form.setContentsMargins(18, 24, 18, 18)
        form.setSpacing(14)

        self.vmax = QtWidgets.QSpinBox()
        self.vmax.setRange(1, 60)
        self.vmax.setValue(30)
        form.addRow("Maximum", self.vmax)

        zum_min = int(survey_map["Zum"].min())
        zum_max = int(survey_map["Zum"].max())
        self.depth_min = QtWidgets.QSpinBox()
        self.depth_min.setRange(zum_min, zum_max)
        self.depth_min.setValue(max(zum_min, 0))
        form.addRow("Min depth", self.depth_min)
        self.depth_max = QtWidgets.QSpinBox()
        self.depth_max.setRange(zum_min, zum_max)
        self.depth_max.setValue(min(zum_max, 5000))
        form.addRow("Max depth", self.depth_max)
        layout.addWidget(controls)

        controls.setStyleSheet(
            "QGroupBox { font-weight: 600; border: 1px solid #d5d5d5; "
            "border-radius: 8px; margin-top: 10px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; "
            "padding: 0 5px; }"
            "QSpinBox { min-height: 30px; padding: 2px 8px; }"
        )
        self.vmax.valueChanged.connect(self._draw)
        self.depth_min.valueChanged.connect(self._draw)
        self.depth_max.valueChanged.connect(self._draw)
        self._draw()

    def _draw(self) -> None:
        self.plot.clear()
        self.plot.addItem(self.color_legend)
        depth_min = self.depth_min.value()
        depth_max = self.depth_max.value()
        data = self.survey_map[
            (self.survey_map["Zum"] >= depth_min)
            & (self.survey_map["Zum"] <= depth_max)
        ]
        shanks = list(pd.unique(self.survey_map["Shank"]))
        shank_index = {shank: index for index, shank in enumerate(shanks)}
        values = data["Val"].to_numpy(dtype=float)
        scaled = np.clip(
            (values - self.vmin) / max(1, self.vmax.value() - self.vmin), 0, 1
        )
        self.plot.addItem(
            pg.ScatterPlotItem(
                x=[shank_index[shank] for shank in data["Shank"]],
                y=data["Zum"].to_numpy(),
                size=7,
                pen=None,
                brush=self.color_map.map(scaled, mode="qcolor"),
            )
        )
        self.plot.getAxis("bottom").setTicks(
            [[(index, str(shank)) for index, shank in enumerate(shanks)]]
        )
        self.plot.setLabel("bottom", "shanks (M to L)")
        self.plot.setLabel("left", "depth from probe tip (µm)")
        self.plot.setTitle(f"Normalized channel voltage (maximum {self.vmax.value()})")
        self.color_legend.setLabels({str(self.vmax.value()): 1.0, str(self.vmin): 0.0})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse a survey map",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        default=argparse.SUPPRESS,
        help="Survey map .txt or .csv file",
    )
    optional = parser.add_argument_group("optional arguments")
    optional.add_argument("-h", "--help", action="help", help="Show this help message")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    survey_map = load_survey_map(args.file)
    app = pg.mkQApp("Survey map browser")
    browser = SurveyMapApp(survey_map)
    browser.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
