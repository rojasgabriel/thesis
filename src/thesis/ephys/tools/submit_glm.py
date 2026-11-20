"""Submit the V1 GLM pipeline to Slurm through labdata's scheduler.

labdata's ``slurm_submit`` writes and submits an sbatch script and exports
``LABDATA_N_JOBS`` = ``--cpus-per-task`` into the job, so the fit uses exactly
the cores it was allocated instead of grabbing the whole node. ``--nice`` keeps
the job polite in the queue so it yields to other people's compute.

Run from the repo root:
    uv run submit-glm                 # sample units, defaults below
    uv run submit-glm --units all --cpus 24 --walltime 12:00:00
    uv run submit-glm --table         # populate GLMFit, all units

`--table` runs `populate-glm`: `GLMFit.populate` with job reservation, so
several jobs can run at the same time and each session is fit once. The tables
must exist (`initialize_schema`).
"""

import argparse
import shlex
from pathlib import Path

PIPELINE = (
    "cd {repository} && export MPLCONFIGDIR=/tmp/glm_mpl_$SLURM_JOB_ID && "
    "uv run glm prepare && "
    "uv run glm fit    --units {units} --workers $LABDATA_N_JOBS && "
    "uv run glm unique --units {units} --workers $LABDATA_N_JOBS"
)
TABLE = "cd {repository} && export MPLCONFIGDIR=/tmp/glm_mpl_$SLURM_JOB_ID && uv run populate-glm"


def populate() -> None:
    """Populate GLMFit, one session at a time, with job reservation."""
    parser = argparse.ArgumentParser(description=populate.__doc__)
    parser.add_argument("--subject", help="Only this subject.")
    parser.add_argument("--session", help="Only this session.")
    args = parser.parse_args()
    from labdata_plugin.schema import GLMFit

    restriction = {
        name: value
        for name, value in (
            ("subject_name", args.subject),
            ("session_name", args.session),
        )
        if value
    }
    GLMFit.populate(restriction, reserve_jobs=True, display_progress=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", choices=("sample", "all"), default="sample")
    parser.add_argument("--cpus", type=int, default=24)
    # Hodgkin's Slurm reports 250 MB of node memory, so any --mem request fails there.
    parser.add_argument(
        "--mem", help="Memory per job, for example 64G; none by default."
    )
    parser.add_argument("--walltime", default="12:00:00")
    parser.add_argument("--table", action="store_true", help="Populate GLMFit.")
    args = parser.parse_args()

    repository = shlex.quote(str(Path.cwd()))
    command = (
        TABLE.format(repository=repository)
        if args.table
        else PIPELINE.format(repository=repository, units=args.units)
    )
    from labdata.compute.schedulers import slurm_submit

    job_id = slurm_submit(
        "glm_fit_table" if args.table else f"v1_glm_{args.units}",
        command,
        ncpuspertask=args.cpus,
        memory=args.mem,
        walltime=args.walltime,
        sbatch_append="--nice=100",
    )
    if job_id is None:
        raise RuntimeError("Slurm did not accept the GLM job")
    print(f"submitted job {job_id}")


if __name__ == "__main__":
    main()
