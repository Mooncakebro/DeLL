# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Task evaluated by each column of skill_matrix. Pass a different order if yours differs.
TEST_COLUMN_ORDER = ['Merging', 'Overtaking', 'EmergencyBrake', 'GiveWay', 'TrafficSign']


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _safe_div(numerator, denominator):
    """Return nan when the denominator is 0 or non-finite (undefined in the paper)."""
    if not np.isfinite(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def _nanmean(values):
    """Mean ignoring nan; nan when every value is nan."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def to_learning_matrix(skill_matrix, learning_order, column_order=None):
    """
    Reorder the evaluation matrix into R[i, j] = SR_{i+1, j+1} and normalize SR to [0, 1].

    Parameters
    ----------
    skill_matrix : array-like, shape (N, N)
        Rows are training stages (row i = evaluation after learning the first i+1 tasks);
        the task of each column is given by column_order. If column_order is None the
        columns are assumed to already follow learning_order.
    learning_order : list[str]
        Learning order of the tasks, length N.
    column_order : list[str], optional
        Task evaluated by each column of skill_matrix.

    Returns
    -------
    R : ndarray, shape (N, N)
        Columns reordered to learning_order, success rates normalized to [0, 1].
    normalized_from_percent : bool
        True if the input was read as percentages (0-100) and divided by 100.
    """
    matrix = np.asarray(skill_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("skill_matrix must be 2-D")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "skill_matrix must be square (rows=stages, cols=tasks), got shape %s"
            % (matrix.shape,)
        )

    n = matrix.shape[0]
    if len(learning_order) != n:
        raise ValueError(
            "learning_order length (%d) does not match matrix size (%d)"
            % (len(learning_order), n)
        )

    if column_order is None:
        if len(set(learning_order)) != n:
            raise ValueError("learning_order has duplicate task names, cannot locate columns")
        ordered = matrix.copy()
    else:
        if len(column_order) != n:
            raise ValueError(
                "column_order length (%d) does not match matrix size (%d)"
                % (len(column_order), n)
            )
        if set(column_order) != set(learning_order):
            missing = set(learning_order) - set(column_order)
            extra = set(column_order) - set(learning_order)
            raise ValueError(
                "column_order and learning_order hold different task sets: missing %s, extra %s"
                % (sorted(missing), sorted(extra))
            )
        position = {task: idx for idx, task in enumerate(column_order)}
        ordered = matrix[:, [position[task] for task in learning_order]]

    # The paper's SR is a 0-1 ratio; normalize percentage input so FT/BT do not inherit that scale.
    normalized_from_percent = bool(np.nanmax(ordered) > 1.0)
    if normalized_from_percent:
        ordered = ordered / 100.0

    return ordered, normalized_from_percent


# --------------------------------------------------------------------------- #
# The four paper metrics (input is the reordered R, 0-based)
# --------------------------------------------------------------------------- #
def forgetting_ratio(R):
    """
    (10) FR: relative decay of each old task between "just learned" and "all learned".

    Returns (per_task, mean) with per_task of length N in learning order; the last task has
    no later stage, so it is nan (matching the paper's bound i = 1..N-1).
    """
    n = R.shape[0]
    per_task = [
        _safe_div(R[i, i] - R[n - 1, i], R[i, i]) for i in range(n - 1)
    ]
    per_task.append(np.nan)
    return per_task, _nanmean(per_task)


def process_forgetting_ratio(R):
    """
    (11) PFR: oscillation during training relative to the running historical best.

    per_task[j] = 1/(N-j) * sum_{i=j+1}^{N} (H_{i,j} - SR_{i,j}) / H_{i,j}
    with H_{i,j} = max{ SR_{i',j} | i' = 1..i-1 }; nan terms are skipped.
    The last task has no later stage, so it is nan.
    """
    n = R.shape[0]
    per_task = []
    for j in range(n - 1):
        terms = []
        for i in range(j + 1, n):
            # 0-based: all evaluations before stage i are rows 0 .. i-1
            historical_best = float(np.max(R[:i, j]))
            terms.append(_safe_div(historical_best - R[i, j], historical_best))
        per_task.append(_nanmean(terms))
    per_task.append(np.nan)
    return per_task, _nanmean(per_task)


def forward_transfer(R):
    """
    (12) FT: zero-shot capability on not-yet-learned tasks after learning task i.

    per_task[i] = 1/(N-i) * sum_{j=i+1}^{N} SR_{i,j}, i.e. the row mean over unlearned tasks.
    The last stage (i=N) has no later task, so it is nan.
    """
    n = R.shape[0]
    per_task = [float(np.mean(R[i, i + 1:])) for i in range(n - 1)]
    per_task.append(np.nan)
    return per_task, _nanmean(per_task)


def backward_transfer(R):
    """
    (13) BT: retention (or reverse enhancement) of old tasks after learning later ones.

    per_task[i] = 1/(i-1) * sum_{j=1}^{i-1} SR_{i,j}, i.e. the row mean over past tasks.
    The first stage has no past task, so it is nan.
    """
    n = R.shape[0]
    per_task = [np.nan]
    per_task.extend(float(np.mean(R[i, :i])) for i in range(1, n))
    return per_task, _nanmean(per_task)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def calculate_metrics(skill_matrix, learning_order, column_order=TEST_COLUMN_ORDER):
    """
    Compute the four continual-learning metrics defined in the paper.

    Returns
    -------
    dict:
        'R'         : reordered performance matrix R[i, j] = SR_{i+1, j+1}
        'tasks'     : task names in learning order
        'FR' / 'PFR' / 'FT' / 'BT'
                    : {'per_task': list[float] (length N, learning order),
                       'mean': float}
    """
    R, normalized_from_percent = to_learning_matrix(skill_matrix, learning_order, column_order)
    tasks = list(learning_order)

    fr_list, fr_avg = forgetting_ratio(R)
    pfr_list, pfr_avg = process_forgetting_ratio(R)
    ft_list, ft_avg = forward_transfer(R)
    bt_list, bt_avg = backward_transfer(R)

    return {
        'R': R,
        'tasks': tasks,
        'normalized_from_percent': normalized_from_percent,
        'FR': {'per_task': fr_list, 'mean': fr_avg},
        'PFR': {'per_task': pfr_list, 'mean': pfr_avg},
        'FT': {'per_task': ft_list, 'mean': ft_avg},
        'BT': {'per_task': bt_list, 'mean': bt_avg},
    }


def metrics_to_frame(result, scale=100.0):
    """Flatten the result into a DataFrame (scale=100 shows percentages)."""
    tasks = result['tasks']
    rows = []
    for metric in ('FR', 'PFR', 'FT', 'BT'):
        mean_value = result[metric]['mean']
        rows.append({
            'Task': 'Overall',
            'Metric': metric,
            'Value': None if np.isnan(mean_value) else mean_value * scale,
        })
        for task, value in zip(tasks, result[metric]['per_task']):
            rows.append({
                'Task': task,
                'Metric': metric,
                'Value': None if np.isnan(value) else value * scale,
            })
    return pd.DataFrame(rows)


def format_summary(result, scale=100.0):
    """Build a printable summary string."""
    arrow = {'FR': 'v', 'PFR': 'v', 'FT': '^', 'BT': '^'}
    lines = ["===== Paper metrics (Eq.10-13) ====="]
    for metric in ('FR', 'PFR', 'FT', 'BT'):
        value = result[metric]['mean'] * scale
        lines.append("  %-4s %-4s = %8.2f%%" % (metric, arrow[metric], value))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Input data
# --------------------------------------------------------------------------- #

LEARNING_ORDER = ['EmergencyBrake', 'TrafficSign', 'Merging', 'Overtaking', 'GiveWay']
# LEARNING_ORDER = ['GiveWay', 'Overtaking', 'Merging', 'TrafficSign', 'EmergencyBrake']

# Fill in one entry per method: {method_name: skill_matrix}.
# If a matrix's columns are not in TEST_COLUMN_ORDER, pass its own column_order below.
METHOD_MATRICES = {
    'TF++': [
        [62.5, 8.89, 40.00, 50.0, 70.00],
        [42.5, 15.56, 83.33, 50.0, 78.42],
        [53.75, 11.11, 51.67, 50.0, 74.74],
        [37.5, 46.67, 21.67, 50.0, 57.37],
        [35.0, 8.89, 26.67, 50.0, 55.79],
    ],
}


def summarize_methods(method_matrices, learning_order, column_order=TEST_COLUMN_ORDER,
                      scale=100.0):
    """
    One row per method with its overall FR / PFR / FT / BT.

    Parameters
    ----------
    method_matrices : dict[str, array-like]
        {method_name: skill_matrix}, each shaped (N, N).
    learning_order : list[str]
        Shared learning order of the tasks.
    column_order : list[str] or None
        Task of each column of the matrices; None if the columns already follow learning_order.
    scale : float
        Use 100 for percentages, 1 for [0, 1] values.
    """
    rows = []
    for name, matrix in method_matrices.items():
        result = calculate_metrics(matrix, learning_order, column_order)
        rows.append({
            'Method': name,
            'FR': result['FR']['mean'] * scale,
            'PFR': result['PFR']['mean'] * scale,
            'FT': result['FT']['mean'] * scale,
            'BT': result['BT']['mean'] * scale,
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("===== Method comparison (%, FR/PFR lower is better; FT/BT higher is better) =====")
    print(summarize_methods(METHOD_MATRICES, LEARNING_ORDER).to_string(index=False))

    name = next(iter(METHOD_MATRICES))
    result = calculate_metrics(METHOD_MATRICES[name], LEARNING_ORDER)

    print("\n===== %s: performance matrix R (rows=stages, cols=learning order, %%) =====" % name)
    print(pd.DataFrame(np.round(result['R'] * 100.0, 2),
                       index=['after %s' % t for t in LEARNING_ORDER],
                       columns=LEARNING_ORDER).to_string())

    print("\n===== %s: per-task metrics (%%) =====" % name)
    print(metrics_to_frame(result).to_string(index=False))

    print("\n" + format_summary(result))
