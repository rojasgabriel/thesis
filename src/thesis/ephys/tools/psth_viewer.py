from typing import Literal, Optional

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output, display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.stats import sem
from spks.event_aligned import population_peth
from spks.utils import alpha_function
from spks.viz import plot_event_aligned_raster


class PSTHViewer:
    def __init__(
        self,
        subject: Optional[str] = None,
        session: Optional[str] = None,
        unit_criteria_id: int = 1,
        pre_seconds: float = 1,
        post_seconds: float = 1,
        binwidth_ms: int = 10,
        plot_type: Literal["raster", "heatmap", "psth"] = "heatmap",
        t_rise: Optional[float] = None,
        t_decay: Optional[float] = None,
        figure: Optional[Figure] = None,
        axes: Optional[Axes] = None,
    ) -> None:

        self.subject = subject
        self.session = session
        self.unit_criteria_id = unit_criteria_id
        self.pre_seconds = pre_seconds
        self.post_seconds = post_seconds
        self.binwidth_ms = binwidth_ms
        self.plot_type = plot_type
        self.split_by: Optional[str] = None
        self.sort_by = "peak_latency" if plot_type == "heatmap" else "trial_order"
        self.fig = figure
        self.ax = axes
        self._cax: Optional[Axes] = None  # dedicated axes for colorbar
        self.kernel = None
        if t_rise is not None and t_decay is not None:
            decay_bins = t_decay / (binwidth_ms / 1000)
            self.kernel = alpha_function(
                int(decay_bins * 15),
                t_rise=t_rise,
                t_decay=decay_bins,
                srate=1000 / binwidth_ms,
            )

    def compute(
        self,
    ) -> tuple[dict[int, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
        """Fetch session data (units, events, trial metadata)."""
        if self.subject is None or self.session is None:
            raise ValueError("subject and session must be set before calling compute()")
        from thesis.ephys.io_chipmunk_trials import fetch_trial_metadata
        from thesis.ephys.io_digital_events import fetch_session_events
        from thesis.ephys.io_session_units import fetch_good_units

        subject: str = self.subject
        session: str = self.session
        st_per_unit = fetch_good_units(subject, session, self.unit_criteria_id)
        align_ev = fetch_session_events(subject, session)
        trial_df = fetch_trial_metadata(subject, session, align_ev)
        return st_per_unit, align_ev, trial_df

    def plot(
        self,
        spike_times: np.ndarray,
        event_times: np.ndarray,
        all_spike_times: Optional[dict[int, np.ndarray]] = None,
    ) -> None:
        """Plot raster, psth, or population heatmap onto self.ax."""
        if self.ax is None or self.fig is None:
            return
        self.ax.cla()
        # hide colorbar axes by default; heatmap will re-show it
        if self._cax is not None:
            self._cax.cla()
            self._cax.set_visible(False)
        if self.plot_type == "raster":
            plot_event_aligned_raster(
                event_times=event_times,
                spike_times=spike_times,
                sorting=(
                    "time_to_first_spike"
                    if self.sort_by == "first_spike_latency"
                    else ""
                ),
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
                ax=self.ax,
            )
            self.ax.set_ylim(0, len(event_times))
            self.ax.vlines(0, 0, len(event_times), colors="r", linestyles="--")
            self.ax.set_xlabel("time from event (s)")
            self.ax.set_ylabel("trial")

        elif self.plot_type == "psth":
            peth, bin_edges, _ = population_peth(
                all_spike_times=[spike_times],
                alignment_times=event_times,
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
                binwidth_ms=self.binwidth_ms,
                kernel=self.kernel,
            )
            peth = peth / (self.binwidth_ms / 1000)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mean_fr = np.mean(peth[0], axis=0)
            sem_fr = sem(peth[0], axis=0)
            self.ax.plot(bin_centers, mean_fr, color="k")
            self.ax.fill_between(
                bin_centers,
                mean_fr - sem_fr,
                mean_fr + sem_fr,
                alpha=0.3,
                color="k",
            )
            ymin, ymax = self.ax.get_ylim()
            self.ax.vlines(0, ymin, ymax, colors="r", linestyles="--")
            self.ax.set_xlabel("time from event (s)")
            self.ax.set_ylabel("sp/s")

        elif self.plot_type == "heatmap":
            if all_spike_times is None:
                return
            unit_ids = list(all_spike_times.keys())
            peth, bin_edges, _ = population_peth(
                all_spike_times=list(all_spike_times.values()),
                alignment_times=event_times,
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
                binwidth_ms=self.binwidth_ms,
                kernel=self.kernel,
            )
            peth = peth / (self.binwidth_ms / 1000)
            pop_matrix = np.mean(peth, axis=1)
            if self.sort_by == "peak_latency":
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                post_event = bin_centers >= 0
                order = np.argsort(
                    np.argmax(pop_matrix[:, post_event], axis=1), kind="stable"
                )
                pop_matrix = pop_matrix[order]
                unit_ids = [unit_ids[i] for i in order]
            n_units = pop_matrix.shape[0]
            im = self.ax.imshow(
                pop_matrix,
                aspect="auto",
                origin="upper",
                extent=(-self.pre_seconds, self.post_seconds, float(n_units), 0.0),
                cmap="viridis",
            )
            self.ax.vlines(0, 0, n_units, colors="cyan", linestyles="--", linewidth=0.8)
            # label y-ticks with real unit IDs
            tick_step = max(1, n_units // 10)
            tick_indices = list(range(0, n_units, tick_step))
            self.ax.set_yticks([i + 0.5 for i in tick_indices])
            self.ax.set_yticklabels(
                [str(unit_ids[i]) for i in tick_indices], fontsize=7
            )
            self.ax.set_xlabel("time from event (s)")
            self.ax.set_ylabel(f"unit ID (sorted by {self.sort_by.replace('_', ' ')})")
            if self._cax is not None:
                self._cax.set_visible(True)
                self.fig.colorbar(im, cax=self._cax, label="sp/s")

        if self.fig is not None:
            self.fig.canvas.draw_idle()

    def plot_split(
        self,
        spike_times: np.ndarray,
        all_spike_times: dict,
        trial_df: pd.DataFrame,
        split_col: str,
        event_times: np.ndarray,
        fig: Figure | None = None,
    ) -> Figure:
        """Plot one subplot per category value of split_col."""
        # -- Assign each event to a trial via trial_start_ts -----------
        trial_starts = np.asarray(trial_df["trial_start_ts"])
        trial_idx = np.searchsorted(trial_starts, event_times, side="right") - 1
        valid = (trial_idx >= 0) & (trial_idx < len(trial_df))
        event_times = event_times[valid]
        trial_idx = trial_idx[valid]

        cats = trial_df[split_col].values[trial_idx]
        ev_df = pd.DataFrame({"event_time": event_times, "category": cats}).dropna(
            subset=["category"]
        )

        groups = ev_df.groupby("category", sort=True, observed=True)
        group_keys = list(groups.groups.keys())
        n_groups = len(group_keys)
        n_cols = min(n_groups, 3)
        n_rows = int(np.ceil(n_groups / n_cols))

        fig_w = 4 * n_cols
        fig_h = (4 if self.plot_type == "heatmap" else 3) * n_rows
        if fig is None:
            fig = plt.figure(figsize=(fig_w, fig_h))
        else:
            fig.clear()
            fig.set_size_inches(fig_w, fig_h)
        axes = fig.subplots(
            n_rows,
            n_cols,
            sharey=self.plot_type != "raster",
            squeeze=False,
        )

        heatmap_order = None
        if self.plot_type == "heatmap" and self.sort_by == "peak_latency":
            sort_peth, bin_edges, _ = population_peth(
                all_spike_times=list(all_spike_times.values()),
                alignment_times=event_times,
                pre_seconds=self.pre_seconds,
                post_seconds=self.post_seconds,
                binwidth_ms=self.binwidth_ms,
                kernel=self.kernel,
            )
            sort_matrix = np.mean(sort_peth / (self.binwidth_ms / 1000), axis=1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            post_event = bin_centers >= 0
            heatmap_order = np.argsort(
                np.argmax(sort_matrix[:, post_event], axis=1), kind="stable"
            )

        for col_idx, key in enumerate(group_keys):
            row, col = divmod(col_idx, n_cols)
            ax = axes[row, col]
            grp = groups.get_group(key)
            grp_event_times = grp["event_time"].values

            if self.plot_type == "raster":
                plot_event_aligned_raster(
                    event_times=grp_event_times,
                    spike_times=spike_times,
                    sorting=(
                        "time_to_first_spike"
                        if self.sort_by == "first_spike_latency"
                        else ""
                    ),
                    pre_seconds=self.pre_seconds,
                    post_seconds=self.post_seconds,
                    ax=ax,
                )
                ax.set_ylim(0, len(grp_event_times))
                ax.vlines(0, 0, len(grp_event_times), colors="r", linestyles="--")
                ax.set_xlabel("time (s)")
                if col == 0:
                    ax.set_ylabel("trial")

            elif self.plot_type == "psth":
                peth, bin_edges, _ = population_peth(
                    all_spike_times=[spike_times],
                    alignment_times=grp_event_times,
                    pre_seconds=self.pre_seconds,
                    post_seconds=self.post_seconds,
                    binwidth_ms=self.binwidth_ms,
                    kernel=self.kernel,
                )
                peth = peth / (self.binwidth_ms / 1000)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                mean_fr = np.mean(peth[0], axis=0)
                sem_fr = sem(peth[0], axis=0)
                ax.plot(bin_centers, mean_fr, color="k")
                ax.fill_between(
                    bin_centers,
                    mean_fr - sem_fr,
                    mean_fr + sem_fr,
                    alpha=0.3,
                    color="k",
                )
                ymin, ymax = ax.get_ylim()
                ax.vlines(0, ymin, ymax, colors="r", linestyles="--")
                ax.set_xlabel("time (s)")
                if col == 0:
                    ax.set_ylabel("sp/s")

            elif self.plot_type == "heatmap":
                peth, _, _ = population_peth(
                    all_spike_times=list(all_spike_times.values()),
                    alignment_times=grp_event_times,
                    pre_seconds=self.pre_seconds,
                    post_seconds=self.post_seconds,
                    binwidth_ms=self.binwidth_ms,
                    kernel=self.kernel,
                )
                peth = peth / (self.binwidth_ms / 1000)
                pop_matrix = np.mean(peth, axis=1)
                unit_ids = list(all_spike_times.keys())
                if heatmap_order is not None:
                    pop_matrix = pop_matrix[heatmap_order]
                    unit_ids = [unit_ids[i] for i in heatmap_order]
                n_units = pop_matrix.shape[0]
                im = ax.imshow(
                    pop_matrix,
                    aspect="auto",
                    origin="upper",
                    extent=(-self.pre_seconds, self.post_seconds, float(n_units), 0.0),
                    cmap="viridis",
                )
                ax.vlines(0, 0, n_units, colors="cyan", linestyles="--", linewidth=0.8)
                tick_step = max(1, n_units // 8)
                tick_indices = list(range(0, n_units, tick_step))
                ax.set_yticks([i + 0.5 for i in tick_indices])
                ax.set_yticklabels([str(unit_ids[i]) for i in tick_indices], fontsize=6)
                ax.set_xlabel("time (s)")
                if col == 0:
                    ax.set_ylabel(f"unit ID ({self.sort_by.replace('_', ' ')})")
                fig.colorbar(im, ax=ax, label="sp/s", shrink=0.6)

            title_val = f"{key:.0f}" if isinstance(key, float) else str(key)
            n_events = len(grp_event_times)
            ax.set_title(f"{split_col}={title_val}\n(n={n_events})", fontsize=9)

        # hide any unused axes in the last row
        for empty_idx in range(n_groups, n_rows * n_cols):
            row, col = divmod(empty_idx, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle(f"Split by: {split_col}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw_idle()
        return fig


class PSTHWidget:
    SPLIT_OPTIONS = [
        "none",
        "stim_category",
        "stim_rate_vision",
        "rewarded",
        "response",
        "prev_rewarded",
        "prev_response",
    ]
    SORT_OPTIONS = {
        "raster": [
            ("trial order", "trial_order"),
            ("first-spike latency", "first_spike_latency"),
        ],
        "psth": [("not applicable", "none")],
        "heatmap": [
            ("peak latency", "peak_latency"),
            ("unit depth", "unit_depth"),
        ],
    }

    def __init__(self, viewer: PSTHViewer) -> None:
        self.viewer = viewer

        # fetch data once on init
        self.st_per_unit, self.align_ev, self.trial_df = viewer.compute()

        # build widgets
        self.unit_slider = widgets.SelectionSlider(
            options=list(self.st_per_unit.keys()),
            value=list(self.st_per_unit.keys())[0],
            description="Unit ID:",
            continuous_update=False,
            layout=widgets.Layout(width="440px", margin="0 16px 0 0"),
            style={"description_width": "60px"},
        )
        self.event_dropdown = widgets.Dropdown(
            options=list(self.align_ev.keys()),
            value="first_stim_ev",
            description="Event:",
            layout=widgets.Layout(width="360px"),
            style={"description_width": "65px"},
        )
        self.plot_type_toggle = widgets.RadioButtons(
            options=["heatmap", "raster", "psth"],
            value=self.viewer.plot_type,
            description="Plot type:",
            layout=widgets.Layout(width="170px"),
            style={"description_width": "70px"},
        )
        split_opts = self.SPLIT_OPTIONS if self.trial_df is not None else ["none"]
        self.split_dropdown = widgets.Dropdown(
            options=split_opts,
            value="none",
            description="Split by:",
            layout=widgets.Layout(width="360px"),
            style={"description_width": "65px"},
        )
        sort_options = self.SORT_OPTIONS[self.viewer.plot_type]
        self.viewer.sort_by = sort_options[0][1]
        self.sort_dropdown = widgets.Dropdown(
            options=sort_options,
            value=self.viewer.sort_by,
            description="Sort by:",
            disabled=self.viewer.plot_type == "psth",
            layout=widgets.Layout(width="360px"),
            style={"description_width": "65px"},
        )

        self._split_fig: Figure | None = None

        # wire up callbacks
        self.unit_slider.observe(self._update, names="value")
        self.event_dropdown.observe(self._update, names="value")
        self.plot_type_toggle.observe(self._on_plot_type_change, names="value")
        self.split_dropdown.observe(self._on_split_change, names="value")
        self.sort_dropdown.observe(self._on_sort_change, names="value")

    def _on_plot_type_change(self, change: object) -> None:
        self.viewer.plot_type = self.plot_type_toggle.value
        is_heatmap = self.viewer.plot_type == "heatmap"
        self.unit_slider.disabled = is_heatmap
        sort_options = self.SORT_OPTIONS[self.viewer.plot_type]
        self.sort_dropdown.options = sort_options
        self.sort_dropdown.value = sort_options[0][1]
        self.sort_dropdown.disabled = self.viewer.plot_type == "psth"
        self.viewer.sort_by = self.sort_dropdown.value
        self._update(None)

    def _on_split_change(self, change: object) -> None:
        split_val = self.split_dropdown.value
        self.viewer.split_by = None if split_val == "none" else split_val
        # in split mode, unit slider only relevant for raster/psth
        self._update(None)

    def _on_sort_change(self, change: object) -> None:
        self.viewer.sort_by = self.sort_dropdown.value
        self._update(None)

    def _update(self, change: object) -> None:
        unit_id = self.unit_slider.value
        event = self.event_dropdown.value
        event_times = self.align_ev[event]

        if self.viewer.split_by is not None and self.trial_df is not None:
            self._out.layout.display = "none"
            self._split_out.layout.display = ""
            with plt.ioff():
                fig = self.viewer.plot_split(
                    spike_times=self.st_per_unit[unit_id],
                    all_spike_times=self.st_per_unit,
                    trial_df=self.trial_df,
                    split_col=self.viewer.split_by,
                    event_times=event_times,
                    fig=self._split_fig,
                )
            if self._split_fig is None:
                self._split_fig = fig
                with self._split_out:
                    display(fig.canvas)
        else:
            self._out.layout.display = ""
            self._split_out.layout.display = "none"
            # close any lingering split figure
            if self._split_fig is not None:
                plt.close(self._split_fig)
                self._split_fig = None
            with self._split_out:
                clear_output(wait=False)
            self.viewer.plot(
                spike_times=self.st_per_unit[unit_id],
                event_times=event_times,
                all_spike_times=self.st_per_unit,
            )

    def show(self) -> None:
        if self.viewer.fig is None:
            with plt.ioff():
                self.viewer.fig = plt.figure(figsize=(6, 5))
                gs = self.viewer.fig.add_gridspec(
                    1, 2, width_ratios=[20, 1], wspace=0.05
                )
                self.viewer.ax = self.viewer.fig.add_subplot(gs[0])
                self.viewer._cax = self.viewer.fig.add_subplot(gs[1])
                self.viewer._cax.set_visible(False)
                if hasattr(self.viewer.fig.canvas, "header_visible"):
                    setattr(self.viewer.fig.canvas, "header_visible", False)

        self._out = widgets.Output()
        with self._out:
            display(self.viewer.fig.canvas)

        self._split_out = widgets.Output()
        self._split_out.layout.display = "none"

        controls = widgets.HBox(
            [
                self.unit_slider,
                widgets.VBox(
                    [
                        self.event_dropdown,
                        self.split_dropdown,
                        self.sort_dropdown,
                    ],
                    layout=widgets.Layout(margin="0 16px 0 0"),
                ),
                self.plot_type_toggle,
            ],
            layout=widgets.Layout(
                align_items="flex-start",
                border="1px solid #d9d9d9",
                padding="12px 16px",
                width="fit-content",
            ),
        )

        display(
            widgets.VBox(
                [
                    controls,
                    self._out,
                    self._split_out,
                ]
            )
        )
        self._update(None)
