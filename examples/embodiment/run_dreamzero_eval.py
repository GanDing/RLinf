#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Launch a DreamZero LIBERO eval with a balanced number of tries per task.

You specify: the model checkpoint, what to evaluate (full LIBERO-130, a single
suite, or a single task), and how many tries you want per task. The wrapper
computes the LIBERO env config (``total_num_envs`` + ``eval_rollout_epoch``)
that realizes that per-task count under a per-epoch parallelism cap, snapping to
the nearest realizable count when the exact one does not fit the cap, and then
launches ``eval_embodied_agent.py`` with the right Hydra overrides.

How the count maps to config
----------------------------
With ``is_eval=True`` + ordered reset ids, LIBERO covers init states in
trial-major interleaved order: (t0,trial0),(t1,trial0),...,(tN,trial0),
(t0,trial1),... Each eval epoch runs ``total_num_envs`` fixed-length episodes
(we pin ``max_steps_per_rollout_epoch = max_episode_steps`` so one episode per
env per epoch), consuming trajectories sequentially from that order. Over
``eval_rollout_epoch`` epochs the consumed set is a contiguous prefix of length
``total_num_envs * eval_rollout_epoch``; when that equals ``k * num_tasks`` every
task gets exactly ``k`` tries (init states repeat once ``k`` exceeds the ~50
init states per task).

Cap / search model
------------------
``--per-epoch-cap`` (default 100) bounds ``total_num_envs`` (parallel envs per
epoch — the real GPU-memory limit). For a per-task count ``k`` the eval must run
``k * N`` trajectories (``N = num_tasks``), factored as
``total_num_envs * eval_rollout_epoch``. We pick the configuration by an exact
search that, for the realized ``k``, maximises ``total_num_envs`` (hence
minimises the epoch count) subject to:

  1. ``total_num_envs <= cap``;
  2. ``total_num_envs % num_gpus == 0`` (so RLinf's env worker splits evenly and
     ``validate_cfg`` accepts it);
  3. ``total_num_envs`` divides ``k * N`` (integer epoch count); and
  4. the resulting per-process contiguous-block consumption is **exactly**
     per-task balanced (verified, not assumed).

The requested ``--per-task-num`` is snapped to the nearest ``k`` for which such a
configuration exists (ties prefer the lower ``k``). Constraint 4 matters: with
multiple GPUs ``total_num_envs % num_gpus == 0`` alone does *not* guarantee
balance (it holds whenever ``num_gpus | k``, but otherwise depends on block
alignment), so each candidate is checked against the real interleaved-consumption
counts before being accepted.

``--num-gpus`` must equal the number of env processes the run actually uses
(``cluster.component_placement`` → the visible accelerators); the launcher pins
that placement to ``0..num_gpus-1`` so the assumed and actual process counts
always agree.

Examples
--------
Full LIBERO-130, ~50 tries/task, cap 100
    python examples/embodiment/run_dreamzero_eval.py \\
        --model-path /ckpt/dreamzero_sft --metadata-path /ckpt/metadata.json \\
        --suite libero_130 --per-task-num 50

One suite
    ... --suite libero_spatial --per-task-num 50

One task (task 3 of libero_goal), 20 tries
    ... --suite libero_goal --task-id 3 --per-task-num 20

Just print the plan + command (default); add --run to actually launch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# task counts per LIBERO benchmark (init states per task is ~50 for all).
SUITE_NUM_TASKS = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_10": 10,
    "libero_90": 90,
    "libero_130": 130,
}
TRIALS_PER_TASK = 50  # LIBERO ships ~50 init states per task
_DEFAULT_BASE_CONFIG = "libero_spatial_eval_dreamzero"


def _per_task_counts(
    num_tasks: int,
    trials_per_task: int,
    total_num_envs: int,
    eval_rollout_epoch: int,
    num_procs: int,
) -> list[int]:
    """Exact per-task trajectory counts for an eval run.

    Replicates ``LiberoEnv``'s ordered reset-state-id consumption: the
    trial-major interleaved sequence (length ``num_tasks * trials_per_task``) is
    reshaped to ``(num_procs, row_len)`` and each process consumes a contiguous
    prefix of its own row, ``total_num_envs // num_procs`` ids per epoch over
    ``eval_rollout_epoch`` epochs (wrapping to the row start when it runs out,
    matching ``_get_ordered_reset_state_ids``). For uniform trials/task the
    interleaved index ``i`` maps to task ``i % num_tasks``.
    """
    total = num_tasks * trials_per_task
    row_len = (total - total % num_procs) // num_procs
    envs_per_proc = total_num_envs // num_procs
    counts = [0] * num_tasks
    for p in range(num_procs):
        base = p * row_len
        start = 0
        for _ in range(eval_rollout_epoch):
            if start + envs_per_proc > row_len:
                start = 0
            for j in range(start, start + envs_per_proc):
                counts[(base + j) % num_tasks] += 1
            start += envs_per_proc
    return counts


def _best_envs_for_k(
    k: int, num_tasks: int, cap: int, num_gpus: int, trials_per_task: int
) -> tuple[int, int] | None:
    """Largest ``total_num_envs`` (hence smallest epoch count) realizing exactly
    ``k`` tries per task, or ``None`` if no balanced config exists.

    Constraints: ``total_num_envs <= cap``, ``total_num_envs % num_gpus == 0``,
    ``total_num_envs`` divides ``k * num_tasks``, and the resulting consumption
    is exactly per-task balanced (checked, not assumed).
    """
    total = k * num_tasks
    upper = min(cap, total)
    upper -= upper % num_gpus
    for total_num_envs in range(upper, 0, -num_gpus):
        if total % total_num_envs != 0:
            continue
        epochs = total // total_num_envs
        counts = _per_task_counts(
            num_tasks, trials_per_task, total_num_envs, epochs, num_gpus
        )
        if min(counts) == max(counts) == k:
            return total_num_envs, epochs
    return None


def _nearest_search_order(target: int, limit: int):
    """Yield positive ints by increasing distance from *target*; ties -> lower."""
    yield target
    for d in range(1, limit + 1):
        if target - d >= 1:
            yield target - d
        yield target + d


def plan_eval_counts(
    per_task_num: int,
    num_tasks: int,
    cap: int,
    num_gpus: int = 1,
    trials_per_task: int = TRIALS_PER_TASK,
) -> dict:
    """Resolve (total_num_envs, eval_rollout_epoch, realized per-task num).

    Snaps ``per_task_num`` to the nearest realizable ``k`` (see ``Cap / search
    model`` in the module docstring) and returns a dict with keys:
    total_num_envs, eval_rollout_epoch, per_task_num (possibly snapped),
    total_trajectories, cap_utilization, snapped (bool).
    """
    if num_tasks < 1 or cap < 1 or per_task_num < 1:
        raise ValueError("num_tasks, cap and per_task_num must be >= 1")
    if cap < num_gpus:
        raise ValueError(f"per-epoch cap ({cap}) must be >= num_gpus ({num_gpus})")

    # k = num_gpus is always realizable (num_gpus | k => guaranteed balanced, with
    # total_num_envs = num_gpus, epochs = num_tasks), so the search always finds a
    # solution within |num_gpus - per_task_num| of the request.
    limit = abs(num_gpus - per_task_num) + num_gpus + 1
    resolved = None
    for k in _nearest_search_order(per_task_num, limit):
        found = _best_envs_for_k(k, num_tasks, cap, num_gpus, trials_per_task)
        if found is not None:
            resolved = (k, *found)
            break
    if resolved is None:  # defensive: the k = num_gpus fallback should prevent this
        raise ValueError(
            f"no realizable per-task count near {per_task_num} "
            f"(num_tasks={num_tasks}, cap={cap}, num_gpus={num_gpus})"
        )

    k, total_num_envs, eval_rollout_epoch = resolved
    total = total_num_envs * eval_rollout_epoch
    return {
        "total_num_envs": total_num_envs,
        "eval_rollout_epoch": eval_rollout_epoch,
        "per_task_num": k,
        "total_trajectories": total,
        "cap_utilization": total_num_envs / cap,
        "snapped": k != per_task_num,
    }


def build_overrides(args, plan: dict) -> list[str]:
    suite = args.suite
    overrides = [
        "runner.only_eval=True",
        # The model (incl. trained weights) loads from actor.model.model_path.
        # runner.ckpt_path is for an extra *.pt file loaded post-init; leave null
        # (pointing it at the model dir raises IsADirectoryError).
        "runner.ckpt_path=null",
        f"rollout.model.model_path={args.model_path}",
        f"env.train.task_suite_name={suite}",
        f"env.eval.task_suite_name={suite}",
        # The rollout worker divides by env.train.total_num_envs even in eval-only
        # mode, so it must be non-null; mirror the eval count.
        f"env.train.total_num_envs={plan['total_num_envs']}",
        f"env.eval.total_num_envs={plan['total_num_envs']}",
        "env.eval.group_size=1",
        "env.eval.is_eval=True",
        "env.eval.auto_reset=True",
        "env.eval.ignore_terminations=True",
        "env.eval.use_fixed_reset_state_ids=True",
        "env.eval.use_ordered_reset_state_ids=True",
        # one episode per env per epoch (R=1) regardless of the suite's value
        "env.eval.max_steps_per_rollout_epoch=${env.eval.max_episode_steps}",
        f"algorithm.eval_rollout_epoch={plan['eval_rollout_epoch']}",
        f"runner.logger.experiment_name={args.exp_name}",
    ]
    if args.metadata_path:
        overrides.append(f"rollout.model.metadata_json_path={args.metadata_path}")
    # task_id_filter is not declared in the env config schema, so force-add with ++.
    if args.task_id is not None:
        overrides.append(f"++env.eval.task_id_filter=[{args.task_id}]")
    else:
        overrides.append("++env.eval.task_id_filter=null")
    overrides.extend(args.extra or [])
    return overrides


def _print_plan(args, num_tasks: int, plan: dict) -> None:
    scope = (
        f"task {args.task_id} of {args.suite}"
        if args.task_id is not None
        else (
            "full LIBERO-130"
            if args.suite == "libero_130"
            else f"suite {args.suite}"
        )
    )
    print("\n===================== eval plan =====================")
    print(f"  scope               : {scope}  (num_tasks={num_tasks})")
    print(f"  per-epoch cap       : {args.per_epoch_cap}  (num_gpus={args.num_gpus})")
    print(f"  requested per-task  : {args.per_task_num}")
    realized = plan["per_task_num"]
    note = "  <-- snapped to nearest realizable" if plan["snapped"] else ""
    print(f"  realized per-task   : {realized}{note}")
    print(f"  total_num_envs      : {plan['total_num_envs']}")
    print(f"  eval_rollout_epoch  : {plan['eval_rollout_epoch']}")
    print(f"  total trajectories  : {plan['total_trajectories']}")
    print(f"  cap utilization     : {plan['cap_utilization'] * 100:.0f}%")
    if realized > TRIALS_PER_TASK:
        print(
            f"  NOTE: per-task ({realized}) > ~{TRIALS_PER_TASK} init states/task; "
            "init states will repeat."
        )
    print("=====================================================\n")


def _uses_custom_placement(args) -> bool:
    """True if the run defines its own cluster placement (so the launcher must
    not cap visible devices). Triggered by a non-default base config or any
    ``cluster.*`` / ``node_groups`` Hydra override passed via --extra."""
    if args.base_config != _DEFAULT_BASE_CONFIG:
        return True
    for tok in args.extra or []:
        key = tok.lstrip("+~")
        if key.startswith("cluster.") or key.startswith("cluster=") or "node_groups" in key:
            return True
    return False


def _build_command(args, overrides: list[str]) -> tuple[list[str], dict]:
    embodied_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.dirname(os.path.dirname(embodied_path))
    src_file = os.path.join(embodied_path, "eval_embodied_agent.py")

    env = dict(os.environ)
    dreamzero_path = args.dreamzero_path or env.get("DREAMZERO_PATH", "")
    pythonpath = os.pathsep.join(
        p for p in [repo_path, dreamzero_path, env.get("PYTHONPATH", "")] if p
    )
    env.update(
        {
            "EMBODIED_PATH": embodied_path,
            "REPO_PATH": repo_path,
            "PYTHONPATH": pythonpath,
            "MUJOCO_GL": env.get("MUJOCO_GL", "osmesa"),
            "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM", "osmesa"),
            "HYDRA_FULL_ERROR": "1",
            "ROBOT_PLATFORM": env.get("ROBOT_PLATFORM", "LIBERO"),
            "LIBERO_TYPE": env.get("LIBERO_TYPE", "standard"),
        }
    )
    # Pin the run to exactly --num-gpus accelerators so the env world size matches
    # the count the planner assumed (total_num_envs % num_gpus == 0). The default
    # base config uses ``component_placement: {actor,env,rollout: all}`` ("all" =
    # all *visible* devices); the comma-joined key cannot be overridden cleanly via
    # Hydra's CLI grammar, so we cap visibility instead. Covers CUDA and Ascend;
    # the other var is harmless on the platform that ignores it. User-set values
    # win.
    #
    # Skip this for a custom / heterogeneous placement (e.g. env on a consumer-GPU
    # node group, rollout on an NPU node group): there the user controls device
    # selection through ``cluster.node_groups`` / ``component_placement``, and
    # capping visibility on the launcher would clobber the head node's device list
    # and conflict with per-node-group resource detection. --num-gpus then only
    # sizes the env split (it must equal the env group's world size).
    if _uses_custom_placement(args):
        print(
            "[info] custom/heterogeneous placement detected "
            "(non-default base-config or cluster.* override) -> NOT pinning "
            "CUDA_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES; ensure --num-gpus "
            "matches the env group's world size."
        )
    else:
        visible = ",".join(str(i) for i in range(args.num_gpus))
        for var in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
            env.setdefault(var, visible)
    cmd = [
        sys.executable,
        src_file,
        "--config-path",
        os.path.join(embodied_path, "config"),
        "--config-name",
        args.base_config,
        *overrides,
    ]
    return cmd, env


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-path", required=True, help="DreamZero SFT checkpoint dir/path.")
    p.add_argument("--metadata-path", default=None, help="metadata.json for normalization.")
    p.add_argument(
        "--suite",
        default="libero_130",
        choices=sorted(SUITE_NUM_TASKS),
        help="Benchmark to eval (libero_130 = full 130).",
    )
    p.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Eval only this task id within --suite (per-task mode).",
    )
    p.add_argument("--per-task-num", type=int, required=True, help="Desired tries per task.")
    p.add_argument("--per-epoch-cap", type=int, default=100, help="Max total_num_envs/epoch.")
    p.add_argument("--num-gpus", type=int, default=1, help="Eval process count.")
    p.add_argument(
        "--base-config",
        default=_DEFAULT_BASE_CONFIG,
        help="Base eval config (suite is overridden via task_suite_name).",
    )
    p.add_argument("--exp-name", default="dreamzero_eval", help="Logger experiment name.")
    p.add_argument("--dreamzero-path", default=None, help="DreamZero repo path (PYTHONPATH).")
    p.add_argument(
        "--extra",
        nargs="*",
        default=None,
        help="Extra raw Hydra overrides appended verbatim.",
    )
    p.add_argument("--run", action="store_true", help="Execute (default: print only).")
    return p


def _ensure_cgroup_pids(min_limit: int = 16384) -> None:
    """Raise the cgroup pids.max if it would be too low for a busy eval.

    Each thread counts against cgroup pids.max, not just processes.  A run
    with O(60) env subprocesses, two RolloutGroup workers with a large
    diffusion model, and Ray's internal infrastructure easily peaks at
    ~7000+ pids.  The Linux default of 7680 is too tight; 16384 gives
    comfortable headroom.  We only write when we can (root / writable) and
    when the current limit is below the minimum.
    """
    pids_max_path = "/sys/fs/cgroup/pids.max"
    try:
        raw = open(pids_max_path).read().strip()
        if raw == "max":
            return
        current = int(raw)
        if current >= min_limit:
            return
        with open(pids_max_path, "w") as f:
            f.write(str(min_limit))
        print(
            f"[preflight] raised cgroup pids.max {current} → {min_limit} "
            f"(needed for {min_limit // 2}+ concurrent threads)"
        )
    except (PermissionError, OSError):
        print(
            f"[preflight] WARNING: cgroup pids.max={raw} may be too low "
            f"(target {min_limit}). Run: echo {min_limit} | sudo tee {pids_max_path}"
        )
    except FileNotFoundError:
        pass


def main() -> None:
    args = _build_parser().parse_args()

    num_tasks = 1 if args.task_id is not None else SUITE_NUM_TASKS[args.suite]
    plan = plan_eval_counts(
        per_task_num=args.per_task_num,
        num_tasks=num_tasks,
        cap=args.per_epoch_cap,
        num_gpus=args.num_gpus,
    )
    _print_plan(args, num_tasks, plan)

    overrides = build_overrides(args, plan)
    cmd, env = _build_command(args, overrides)

    printable = " ".join(
        (f"'{c}'" if any(ch in c for ch in "[]${} ") else c) for c in cmd
    )
    print("Command:\n" + printable + "\n")

    if not args.run:
        print("(dry-run: pass --run to execute)")
        return
    _ensure_cgroup_pids()
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
