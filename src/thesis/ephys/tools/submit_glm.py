"""Submit the V1 GLM pipeline to Slurm through labdata's scheduler.

labdata's ``slurm_submit`` writes and submits an sbatch script and exports
``LABDATA_N_JOBS`` = ``--cpus-per-task`` into the job, so the fit uses exactly
the cores it was allocated instead of grabbing the whole node. ``--nice`` keeps
the job polite in the queue so it yields to other people's compute.

Run from the repo root:
    uv run submit-glm                 # sample units, defaults below
    uv run submit-glm --units all --cpus 24 --mem 64G --walltime 12:00:00
"""

import argparse

from labdata.compute.schedulers import slurm_submit

PIPELINE = (
    "cd ~/lib/thesis && export MPLCONFIGDIR=/tmp/glm_mpl_$SLURM_JOB_ID && "
    "uv run glm prepare && "
    "uv run glm fit    --units {units} --workers $LABDATA_N_JOBS && "
    "uv run glm unique --units {units} --workers $LABDATA_N_JOBS"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--units", choices=("sample", "all"), default="sample")
    parser.add_argument("--cpus", type=int, default=24)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--walltime", default="12:00:00")
    args = parser.parse_args()

    command = PIPELINE.format(units=args.units)
    job_id = slurm_submit(
        f"v1_glm_{args.units}",
        command,
        ncpuspertask=args.cpus,
        memory=args.mem,
        walltime=args.walltime,
        sbatch_append="--nice=100",
    )
    print(f"submitted job {job_id}")


if __name__ == "__main__":
    main()
