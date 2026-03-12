"""
Timing analysis for OneMap evaluation runs.
Usage: python read_timing.py [--results results/] [--log eval_output.log]
"""

import os
import argparse
import numpy as np
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="OneMap timing analysis")
    parser.add_argument("--results", default="results/", help="Results directory")
    parser.add_argument("--log", default="eval_output.log", help="Eval log file path")
    return parser.parse_args()


def parse_log_timestamps(log_path):
    """Extract start and end wall-clock times from the eval log.

    Handles two formats:
      [03/10/26 17:24:33] WARNING ...          (Python logging)
      [17:24:53:532914]:[Sim] ...              (habitat-sim, HH:MM:SS:microsec)
    """
    first_ts, last_ts = None, None
    if not os.path.exists(log_path):
        return None, None
    with open(log_path) as f:
        for line in f:
            if not line.startswith("["):
                continue
            bracket_end = line.find("]")
            if bracket_end == -1:
                continue
            ts_str = line[1:bracket_end]
            ts = None
            # Format 1: "03/10/26 17:24:33"
            try:
                ts = datetime.strptime(ts_str, "%m/%d/%y %H:%M:%S")
            except ValueError:
                pass
            # Format 2: "17:24:53:532914"  (HH:MM:SS:microseconds)
            if ts is None:
                parts = ts_str.split(":")
                if len(parts) == 4:
                    try:
                        ts = datetime.strptime(":".join(parts[:3]), "%H:%M:%S")
                    except ValueError:
                        pass
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
    return first_ts, last_ts


def load_trajectories(traj_dir):
    """Return dict {episode_id: step_count}."""
    steps = {}
    if not os.path.isdir(traj_dir):
        return steps
    for fname in os.listdir(traj_dir):
        if not fname.startswith("poses_") or not fname.endswith(".csv"):
            continue
        ep_id = int(fname[len("poses_"):-len(".csv")])
        fpath = os.path.join(traj_dir, fname)
        data = np.loadtxt(fpath, delimiter=",")
        steps[ep_id] = len(data) if data.ndim > 1 else 1
    return steps


def load_states(state_dir):
    """Return dict {episode_id: state_value}."""
    # Result codes from habitat_evaluator.py
    result_names = {1: "SUCCESS", 2: "FAILURE_MISDETECT", 3: "FAILURE_STUCK",
                    4: "FAILURE_OOT", 5: "FAILURE_NOT_REACHED", 6: "FAILURE_ALL_EXPLORED"}
    states = {}
    if not os.path.isdir(state_dir):
        return states, result_names
    for fname in os.listdir(state_dir):
        if not fname.startswith("state_") or not fname.endswith(".txt"):
            continue
        ep_id = int(fname[len("state_"):-len(".txt")])
        with open(os.path.join(state_dir, fname)) as f:
            states[ep_id] = int(f.read().strip())
    return states, result_names


def main():
    args = parse_args()
    results_dir = args.results
    traj_dir = os.path.join(results_dir, "trajectories")
    state_dir = os.path.join(results_dir, "state")

    states, result_names = load_states(state_dir)
    steps_per_ep = load_trajectories(traj_dir)
    n_eps = len(states)

    if n_eps == 0:
        print("No results found in", results_dir)
        return

    # Wall-clock timing from log
    start_ts, end_ts = parse_log_timestamps(args.log)
    total_sec = (end_ts - start_ts).seconds if (start_ts and end_ts) else None

    # Step counts
    all_steps = [steps_per_ep[ep] for ep in sorted(steps_per_ep)]
    total_steps = sum(all_steps)

    print("=" * 55)
    print("  OneMap Timing Analysis")
    print("=" * 55)

    if total_sec is not None:
        h, rem = divmod(total_sec, 3600)
        m, s = divmod(rem, 60)
        time_str = f"{int(h)}h {int(m)}m {int(s)}s" if h else f"{int(m)}m {int(s)}s"
        print(f"  Total wall-clock time   : {time_str}  ({total_sec}s)")
        print(f"  Episodes completed      : {n_eps}")
        print(f"  Avg time / episode      : {total_sec / n_eps:.1f} s")
        if total_steps > 0:
            print(f"  Avg time / step         : {total_sec / total_steps * 1000:.1f} ms")
    else:
        print(f"  (Log file not found — no wall-clock data)")
        print(f"  Episodes completed      : {n_eps}")

    if all_steps:
        print(f"  Avg steps / episode     : {np.mean(all_steps):.1f}")
        print(f"  Min / Max steps         : {int(np.min(all_steps))} / {int(np.max(all_steps))}")

    # Per-result-type breakdown
    print()
    print("  Result breakdown:")
    for code, name in result_names.items():
        count = sum(1 for v in states.values() if v == code)
        if count > 0:
            ep_steps = [steps_per_ep[ep] for ep, v in states.items() if v == code and ep in steps_per_ep]
            avg_steps = f"{np.mean(ep_steps):.0f} steps" if ep_steps else "n/a"
            pct = count / n_eps * 100
            print(f"    {name:<25} {count:>3} eps ({pct:5.1f}%)   avg {avg_steps}")

    # Per-episode detail
    print()
    print(f"  {'Episode':>7}  {'Steps':>6}  {'Result'}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*22}")
    for ep in sorted(states):
        s = steps_per_ep.get(ep, "?")
        r = result_names.get(states[ep], str(states[ep]))
        print(f"  {ep:>7}  {str(s):>6}  {r}")

    print("=" * 55)


if __name__ == "__main__":
    main()
