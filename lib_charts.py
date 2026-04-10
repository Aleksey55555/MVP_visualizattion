import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go

from matplotlib.patches import Rectangle, Circle, Wedge, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import textwrap
from typing import Literal

def make_assessment_long_format(
    df: pd.DataFrame,
    group_cols: list[str] | None,
    respondent_col: str,
    competency_col: str,
    employee_col: str,
    value_col: str,
    *,
    role_col: str | None = None,
    agg: str = "mean",
    dropna_value: bool = True,
    role_map: dict[str, str] | None = None,
    competency_map: dict[str, str] | None = None,
    deduplicate_by: list[str] | None = None,
    score_col_name: str = "score",
    n_col_name: str = "n",
    scale_min: float | None = None,
    scale_max: float | None = None,
    scale_min_col_name: str = "scale_min",
    scale_max_col_name: str = "scale_max",
) -> pd.DataFrame:
    """
    Универсальное преобразование исходного датафрейма в long-формат.

    Если role_col передан:
        Сотрудник | роль | <group_cols> | Компетенция | score | n

    Если role_col=None:
        Сотрудник | <group_cols> | Компетенция | score | n

    Где:
    - score = агрегированное значение value_col
    - n = число уникальных респондентов

    Parameters
    ----------
    df : pd.DataFrame
        Исходный датафрейм.

    group_cols : list[str] | None
        Колонки группировки. Имена сохраняются как есть.

    respondent_col : str
        Колонка с идентификатором респондента.

    competency_col : str
        Колонка с компетенцией.

    employee_col : str
        Колонка идентификации оцениваемого сотрудника.

    value_col : str
        Колонка со значениями, из которых считается score.

    role_col : str | None, default=None
        Колонка роли. Если None, long-формат строится без колонки 'роль'.

    agg : str, default="mean"
        Агрегация для score: "mean", "sum", "median".

    dropna_value : bool, default=True
        Удалять ли строки с пропущенным value_col.

    role_map : dict | None
        Словарь для переименования ролей.
        Используется только если role_col не None.

    competency_map : dict | None
        Словарь для переименования компетенций.

    deduplicate_by : list[str] | None
        Колонки для удаления дублей до агрегации.
        Если None, используется логика по умолчанию:
        - с role_col: [respondent_col, employee_col, role_col, competency_col]
        - без role_col: [respondent_col, employee_col, competency_col]

    score_col_name : str, default="score"
        Имя итоговой колонки score.

    n_col_name : str, default="n"
        Имя итоговой колонки n.

    scale_min : float | None, default=None
        Глобальный минимум шкалы, который будет записан в long-формат.

    scale_max : float | None, default=None
        Глобальный максимум шкалы, который будет записан в long-формат.

    Returns
    -------
    pd.DataFrame
        Датафрейм в long-формате.
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas.DataFrame.")

    if group_cols is None:
        group_cols = []

    if agg not in {"mean", "sum", "median"}:
        raise ValueError("agg должен быть одним из: 'mean', 'sum', 'median'.")

    required_cols = [
        respondent_col,
        competency_col,
        employee_col,
        value_col,
        *group_cols,
    ]

    if role_col is not None:
        required_cols.append(role_col)

    if deduplicate_by is not None:
        required_cols.extend(deduplicate_by)

    required_cols = list(dict.fromkeys(required_cols))
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"В датафрейме отсутствуют колонки: {missing_cols}")

    work = df[required_cols].copy()

    if dropna_value:
        work = work[work[value_col].notna()].copy()

    if work.empty:
        raise ValueError("После удаления пропусков не осталось данных.")

    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work[work[value_col].notna()].copy()

    if work.empty:
        raise ValueError("После приведения value_col к числу не осталось данных.")

    if role_col is not None and role_map is not None:
        work[role_col] = work[role_col].map(lambda x: role_map.get(x, x))

    if competency_map is not None:
        work[competency_col] = work[competency_col].map(lambda x: competency_map.get(x, x))

    # --- дедупликация
    if deduplicate_by is None:
        dedup_cols = [respondent_col, employee_col, competency_col]
        if role_col is not None:
            dedup_cols.insert(2, role_col)
    else:
        dedup_cols = list(deduplicate_by)
        missing_dedup = [c for c in dedup_cols if c not in work.columns]
        if missing_dedup:
            raise ValueError(f"Колонки deduplicate_by не найдены в датафрейме: {missing_dedup}")

    work = work.drop_duplicates(subset=dedup_cols)

    if work.empty:
        raise ValueError("После дедупликации не осталось данных.")

    # --- группировка
    groupby_cols = [employee_col]
    if role_col is not None:
        groupby_cols.append(role_col)
    groupby_cols.extend(group_cols)
    groupby_cols.append(competency_col)

    agg_map = {
        "mean": "mean",
        "sum": "sum",
        "median": "median",
    }

    long_df = (
        work.groupby(groupby_cols, dropna=False)
        .agg(
            **{
                score_col_name: (value_col, agg_map[agg]),
                n_col_name: (respondent_col, pd.Series.nunique),
            }
        )
        .reset_index()
    )

    # --- приведение к рабочему формату
    rename_map = {
        employee_col: "Сотрудник",
        competency_col: "Компетенция",
    }
    if role_col is not None:
        rename_map[role_col] = "роль"

    long_df = long_df.rename(columns=rename_map)
    long_df[scale_min_col_name] = scale_min
    long_df[scale_max_col_name] = scale_max

    final_cols = ["Сотрудник"]
    if role_col is not None:
        final_cols.append("роль")
    final_cols += list(group_cols) + ["Компетенция", score_col_name, n_col_name, scale_min_col_name, scale_max_col_name]

    long_df = long_df[final_cols].copy()

    sort_cols = ["Сотрудник"]
    if role_col is not None:
        sort_cols.append("роль")
    sort_cols += list(group_cols) + ["Компетенция"]

    long_df = long_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return long_df


def make_assessment_from_aggregated_wide(
    df: pd.DataFrame,
    employee_col: str,
    competency_value_cols: list[str],
    group_cols: list[str] | None = None,
    *,
    role_col: str | None = None,
    competency_map: dict[str, str] | None = None,
    score_col_name: str = "score",
    n_col_name: str = "n",
    scale_min: float | None = None,
    scale_max: float | None = None,
    scale_min_col_name: str = "scale_min",
    scale_max_col_name: str = "scale_max",
) -> pd.DataFrame:
    """
    Преобразует агрегированный wide-формат в рабочий long-формат.

    Ожидается, что каждая строка уже соответствует одному сотруднику,
    а каждая колонка из competency_value_cols содержит итоговую оценку
    сотрудника по соответствующей компетенции. Для такого формата
    n всегда выставляется как 1.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas.DataFrame.")

    if group_cols is None:
        group_cols = []

    if not competency_value_cols:
        raise ValueError("Нужно передать хотя бы одну колонку компетенции.")

    id_cols = [employee_col]
    if role_col is not None:
        id_cols.append(role_col)
    id_cols.extend(group_cols)

    required_cols = list(dict.fromkeys(id_cols + list(competency_value_cols)))
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"В датафрейме отсутствуют колонки: {missing_cols}")

    work = df[required_cols].copy()

    long_df = work.melt(
        id_vars=id_cols,
        value_vars=list(competency_value_cols),
        var_name="Компетенция",
        value_name=score_col_name,
    )

    long_df[score_col_name] = pd.to_numeric(long_df[score_col_name], errors="coerce")
    long_df = long_df[long_df[score_col_name].notna()].copy()

    if long_df.empty:
        raise ValueError("После преобразования не осталось числовых значений score.")

    if competency_map is not None:
        long_df["Компетенция"] = long_df["Компетенция"].map(lambda x: competency_map.get(x, x))

    rename_map = {employee_col: "Сотрудник"}
    if role_col is not None:
        rename_map[role_col] = "роль"

    long_df = long_df.rename(columns=rename_map)
    long_df[n_col_name] = 1
    long_df[scale_min_col_name] = scale_min
    long_df[scale_max_col_name] = scale_max

    final_cols = ["Сотрудник"]
    if role_col is not None:
        final_cols.append("роль")
    final_cols += list(group_cols) + ["Компетенция", score_col_name, n_col_name, scale_min_col_name, scale_max_col_name]

    long_df = long_df[final_cols].copy()

    sort_cols = ["Сотрудник"]
    if role_col is not None:
        sort_cols.append("роль")
    sort_cols += list(group_cols) + ["Компетенция"]

    long_df = long_df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return long_df


PALETTE_CYCLE = [
    "#FF474A",
    "#30BAAD",
    "#606E7D",
    "#E0D9CE",
    "#495867",
    "#899AAB",
    "#34CCBE",
    "#B5A798",
    "#CBBFAD",
]


def _wrap_and_truncate_label(
    text: str,
    width: int = 18,
    max_lines: int = 3,
    placeholder: str = "...",
) -> str:
    if pd.isna(text):
        return ""

    text = str(text).strip()
    if not text:
        return ""

    lines = textwrap.wrap(
        text,
        width=width,
        break_long_words=True,
        break_on_hyphens=False,
    )

    if len(lines) <= max_lines:
        return "\n".join(lines)

    clipped = lines[:max_lines]
    last = clipped[-1]

    if len(last) >= len(placeholder):
        last = last[: max(1, width - len(placeholder))].rstrip() + placeholder
    else:
        last = placeholder

    clipped[-1] = last
    return "\n".join(clipped)




def _abbreviate_employee_name(value: object, employee_name_format: str = "keep") -> str:
    if pd.isna(value):
        return ""
    original = str(value).strip()
    if not original:
        return ""
    parts = [part for part in original.split() if part]
    if employee_name_format == "keep":
        return original
    if employee_name_format == "first5":
        return " ".join(parts[:5]) if parts else original
    if employee_name_format == "first3":
        return " ".join(parts[:3]) if parts else original
    if employee_name_format in {"first", "first_initials"}:
        if len(parts) < 2:
            return original
        kept = parts[0]
        abbreviated = [f"{part[0]}." for part in parts[1:] if part]
        return " ".join([kept] + abbreviated)
    if employee_name_format in {"third", "third_initials"}:
        if len(parts) < 3:
            return original
        kept = parts[2]
        prefix = [f"{part[0]}." for part in parts[:2] if part]
        suffix = [f"{part[0]}." for part in parts[3:] if part]
        return " ".join(prefix + [kept] + suffix)
    if employee_name_format == "second":
        if len(parts) < 2:
            return original
        kept = parts[1]
        prefix = [f"{part[0]}." for part in parts[:1] if part]
        suffix = [f"{part[0]}." for part in parts[2:] if part]
        return " ".join(prefix + [kept] + suffix)
    return original
def plot_competency_barplot(
    vis_df: pd.DataFrame,
    employee_col: str = "Сотрудник",
    competency_col: str = "Компетенция",
    role_col: str = "роль",
    score_col: str = "score",
    aggregate_by: str = "competency",
    roles: list[str] | None = None,
    competencies: list[str] | None = None,
    group_filters: dict[str, str | list[str]] | None = None,
    stat: str = "mean",
    show_stat: bool = True,
    show_minmax: str | None = None,
    figsize: tuple[float, float] | None = None,
    sort_by: str = "order",
    title: str | None = None,
    palette: list[str] | tuple[str, ...] | None = None,
    err_color: str = "#121111",
    axis_line_color: str = "#BDBDBD",
    axis_line_width: float = 3.0,
    capsize: int = 4,
    label_y_offset: float = 0.18,
    label_x_pad: float = 0.02,
    label_wrap_width: int = 15,
    label_max_lines: int = 3,
    x_tick_step: float | None = None,
    x_max: float | None = None,
    grid: bool = False,
    title_fontsize: int = 13,
    axis_label_fontsize: int = 11,
    tick_label_fontsize: int = 10,
    value_fontsize: int = 9,
    category_col: str | None = None,
    category_values: list[str] | None = None,
    filters: dict[str, str | list[str]] | None = None,
    category_label: str | None = None,
    facet_col: str | None = None,
    facet_values: list[str] | None = None,
    facet_label: str | None = None,
    agg: str = "weighted",
    employee_name_format: str = "keep",
    highlight_shape: str = "rectangle",
    y_top_n_map: dict[str, int] | None = None,
    reference_enabled: bool = False,
    reference_values: list[str] | None = None,
    reference_label: str = "Референс",
    reference_delta: float | None = None,
    reference_shade: str = "#F3EEEB",
    reference_arrow_fontsize: float | None = None,
):
    if not isinstance(vis_df, pd.DataFrame):
        raise TypeError("vis_df должен быть pandas.DataFrame.")
    if aggregate_by not in {"competency", "role"}:
        raise ValueError("aggregate_by должен быть 'competency' или 'role'.")
    if stat not in {"mean", "median"}:
        raise ValueError("stat должен быть 'mean' или 'median'.")
    if show_minmax not in {None, "errorbar", "text", "both"}:
        raise ValueError("show_minmax должен быть одним из: None, 'errorbar', 'text', 'both'.")
    if sort_by not in {"order", "stat_desc", "stat_asc", "name"}:
        raise ValueError("sort_by должен быть одним из: 'order', 'stat_desc', 'stat_asc', 'name'.")
    if x_tick_step is not None and x_tick_step <= 0:
        raise ValueError("x_tick_step должен быть > 0.")
    if x_max is not None and x_max <= 0:
        raise ValueError("x_max должен быть > 0.")
    if palette is not None and len(palette) == 0:
        raise ValueError("palette не должна быть пустой.")
    if agg not in {"weighted", "mean"}:
        raise ValueError("agg должен быть 'weighted' или 'mean'.")
    if employee_name_format not in {"keep", "first", "second", "third", "first5", "first3", "first_initials", "third_initials"}:
        raise ValueError("employee_name_format должен быть одним из: 'keep', 'first5', 'first3', 'first', 'second', 'third', 'first_initials', 'third_initials'.")
    if highlight_shape not in {"rectangle", "circle"}:
        raise ValueError("highlight_shape должен быть 'rectangle' или 'circle'.")
    y_top_n_map = dict(y_top_n_map or {})

    df = vis_df.copy()
    n_col = "n" if "n" in df.columns else None

    # backward compatibility
    resolved_category_col = category_col
    resolved_category_values = category_values
    resolved_filters = dict(filters or {})
    if resolved_category_col is None:
        if aggregate_by == "competency":
            resolved_category_col = competency_col
            resolved_category_values = competencies
        else:
            if role_col not in df.columns:
                raise ValueError("Для aggregate_by='role' в данных должна быть колонка роли.")
            resolved_category_col = role_col
            resolved_category_values = roles
    if category_label is None:
        category_label = str(resolved_category_col)
    if facet_col is not None and facet_label is None:
        facet_label = str(facet_col)
    if roles is not None and resolved_category_col != role_col and role_col in df.columns:
        resolved_filters[role_col] = roles
    if group_filters:
        resolved_filters.update(group_filters)

    required_cols = [employee_col, resolved_category_col, score_col]
    if resolved_category_col not in df.columns:
        raise ValueError(f"Колонка категории '{resolved_category_col}' не найдена в vis_df.")
    if facet_col is not None:
        required_cols.append(facet_col)
    if n_col is not None:
        required_cols.append(n_col)
    required_cols.extend(list(resolved_filters.keys()))
    missing_cols = [c for c in list(dict.fromkeys(required_cols)) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"В vis_df отсутствуют колонки: {missing_cols}")

    for fcol, fval in resolved_filters.items():
        if fcol not in df.columns:
            raise ValueError(f"Колонка фильтра '{fcol}' не найдена в vis_df.")
        if isinstance(fval, (list, tuple, set)):
            df = df[df[fcol].isin(list(fval))]
        else:
            df = df[df[fcol] == fval]

    if df.empty:
        raise ValueError("После фильтров не осталось данных.")

    df = df.dropna(subset=[score_col, resolved_category_col]).copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df[score_col].notna()].copy()
    if facet_col is not None:
        df = df[df[facet_col].notna()].copy()
    if df.empty:
        raise ValueError("После очистки score_col не осталось числовых данных.")

    df[resolved_category_col] = df[resolved_category_col].astype(str)
    if facet_col is not None:
        df[facet_col] = df[facet_col].astype(str)

    present_categories = set(df[resolved_category_col].dropna().astype(str).tolist())
    if resolved_category_values is None:
        categories = sorted(present_categories, key=lambda x: str(x))
    else:
        categories = [str(x) for x in resolved_category_values if str(x) in present_categories]
    if not categories:
        raise ValueError(f"Нет значений для '{category_label}' после фильтров.")

    if facet_col is None:
        facets = [None]
    else:
        present_facets = set(df[facet_col].dropna().astype(str).tolist())
        if facet_values is None:
            facets = sorted(present_facets, key=lambda x: str(x))
        else:
            facets = [str(x) for x in facet_values if str(x) in present_facets]
        if not facets:
            raise ValueError(f"Нет значений для '{facet_label or facet_col}' после фильтров.")

    def _unique_keep_order(columns: list[str]) -> list[str]:
        seen = set()
        out = []
        for col in columns:
            if col not in seen:
                seen.add(col)
                out.append(col)
        return out

    def collapse_employee_level(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        group_cols_local = _unique_keep_order([employee_col, resolved_category_col] + ([facet_col] if facet_col is not None else []))
        work = frame.copy()
        if n_col is not None:
            work[n_col] = pd.to_numeric(work[n_col], errors="coerce").fillna(0)
        else:
            work["_n_tmp"] = 1.0
        weight_col = n_col if n_col is not None else "_n_tmp"
        if agg == "weighted":
            work["_effective_weight"] = work[weight_col].where(work[weight_col] > 0, 1.0)
            work["_weighted_score"] = work[score_col] * work["_effective_weight"]
            out = (
                work.groupby(group_cols_local, dropna=False)
                .agg(
                    _score_sum=("_weighted_score", "sum"),
                    _weight_sum=("_effective_weight", "sum"),
                    _n_sum=(weight_col, "sum"),
                )
                .reset_index()
            )
            out["employee_score"] = np.where(out["_weight_sum"] > 0, out["_score_sum"] / out["_weight_sum"], np.nan)
            out["employee_n"] = out["_n_sum"]
            return out.drop(columns=["_score_sum", "_weight_sum", "_n_sum"])
        out = (
            work.groupby(group_cols_local, dropna=False)
            .agg(employee_score=(score_col, "mean"), employee_n=(weight_col, "sum"))
            .reset_index()
        )
        return out

    details_df = collapse_employee_level(df)
    if facet_col is None:
        details_df["__facet__"] = ""
        facets_iter = [""]
    else:
        details_df = details_df[details_df[facet_col].isin(facets)].copy()
        facets_iter = facets

    summary_rows = []
    for facet_value in facets_iter:
        facet_df = details_df if facet_col is None else details_df[details_df[facet_col] == facet_value].copy()
        facet_summary = (
            facet_df.groupby(resolved_category_col, as_index=False)
            .agg(
                n=("employee_score", "count"),
                stat_value=("employee_score", "mean" if stat == "mean" else "median"),
                min=("employee_score", "min"),
                max=("employee_score", "max"),
            )
        )
        facet_summary = facet_summary.set_index(resolved_category_col).reindex(categories).reset_index()
        facet_summary = facet_summary[facet_summary["stat_value"].notna()].reset_index(drop=True)
        if facet_summary.empty:
            continue
        if sort_by == "stat_desc":
            facet_summary = facet_summary.sort_values("stat_value", ascending=False).reset_index(drop=True)
        elif sort_by == "stat_asc":
            facet_summary = facet_summary.sort_values("stat_value", ascending=True).reset_index(drop=True)
        elif sort_by == "name":
            facet_summary = facet_summary.sort_values(resolved_category_col, key=lambda s: s.astype(str)).reset_index(drop=True)
        facet_summary["_label"] = facet_summary[resolved_category_col].map(
            lambda x: _wrap_and_truncate_label(
                _abbreviate_employee_name(x, employee_name_format)
                if resolved_category_col == employee_col else x,
                width=label_wrap_width,
                max_lines=label_max_lines,
            )
        )
        facet_summary["facet_value"] = facet_value
        summary_rows.append(facet_summary)

    if not summary_rows:
        raise ValueError("После применения порядка категорий не осталось данных для графика.")

    summary_df = pd.concat(summary_rows, ignore_index=True)
    valid_categories = set(summary_df[resolved_category_col].astype(str).tolist())
    details_df = details_df[details_df[resolved_category_col].astype(str).isin(valid_categories)].copy()

    if figsize is None:
        max_count = max(summary_df.groupby("facet_value").size().max(), 1)
        base_h = max(4, max_count * 0.6 + 1.6)
        figsize = (11, base_h * max(len(summary_df["facet_value"].unique()), 1))

    facet_values_final = summary_df["facet_value"].drop_duplicates().tolist()
    facet_count = len(facet_values_final)
    fig, axes = plt.subplots(
        nrows=facet_count,
        ncols=1,
        figsize=figsize,
        sharex=True if facet_count > 1 else False,
    )
    if facet_count == 1:
        axes = [axes]

    palette_used = list(palette) if palette is not None else PALETTE_CYCLE
    draw_whiskers = show_minmax in ("errorbar", "both")
    show_minmax_text = show_minmax in ("text", "both")
    global_xmax_candidates = []
    for fv in facet_values_final:
        cur = summary_df[summary_df["facet_value"] == fv]
        if not cur.empty:
            vals = cur["max"].to_numpy(dtype=float)
            if vals.size:
                global_xmax_candidates.append(np.nanmax(vals))
            else:
                global_xmax_candidates.append(np.nanmax(cur["stat_value"].to_numpy(dtype=float)))
    xmax_base = np.nanmax(global_xmax_candidates) if global_xmax_candidates else 1.0
    if not np.isfinite(xmax_base) or xmax_base <= 0:
        xmax_base = 1.0
    xlim_right = float(x_max) if x_max is not None else float(xmax_base) * 1.14

    for ax, facet_value in zip(axes, facet_values_final):
        cur = summary_df[summary_df["facet_value"] == facet_value].reset_index(drop=True)
        pos = np.arange(len(cur))
        stat_values = cur["stat_value"].to_numpy(dtype=float)
        mins = cur["min"].to_numpy(dtype=float)
        maxs = cur["max"].to_numpy(dtype=float)
        colors = [palette_used[i % len(palette_used)] for i in range(len(cur))]
        ax.barh(pos, stat_values, color=colors, zorder=2)
        if draw_whiskers:
            xerr = np.vstack([stat_values - mins, maxs - stat_values])
            ax.errorbar(stat_values, pos, xerr=xerr, fmt="none", ecolor=err_color, elinewidth=2, capsize=capsize, zorder=3)
        ax.set_xlim(0, xlim_right)
        x_text_pad = ax.get_xlim()[1] * float(label_x_pad)
        ax.axvline(0, color=axis_line_color, linewidth=axis_line_width, zorder=5)
        for i in range(len(pos)):
            has_whiskers_i = draw_whiskers and np.isfinite(mins[i]) and np.isfinite(maxs[i]) and (maxs[i] > mins[i])
            yi = pos[i] + label_y_offset if has_whiskers_i else pos[i]
            va = "bottom" if has_whiskers_i else "center"
            if show_stat:
                ax.text(stat_values[i] + x_text_pad, yi, f"{stat_values[i]:.2f}", va=va, ha="left", fontsize=value_fontsize, color="#121111", zorder=4)
            if show_minmax_text:
                if np.isfinite(mins[i]):
                    ax.text(mins[i] - x_text_pad, pos[i], f"{mins[i]:.2f}", va="center", ha="right", fontsize=max(8, value_fontsize - 1), color=err_color, zorder=4)
                if np.isfinite(maxs[i]):
                    ax.text(maxs[i] + x_text_pad, pos[i], f"{maxs[i]:.2f}", va="center", ha="left", fontsize=max(8, value_fontsize - 1), color=err_color, zorder=4)
        ax.set_yticks(pos)
        ax.set_yticklabels(cur["_label"].tolist(), fontsize=tick_label_fontsize)
        ax.set_ylabel("")
        if x_tick_step is not None:
            max_tick = ax.get_xlim()[1]
            ax.set_xticks(np.arange(0, max_tick + x_tick_step / 2, x_tick_step))
        ax.tick_params(axis="x", labelsize=tick_label_fontsize)
        ax.tick_params(axis="both", length=0)
        if grid:
            ax.grid(axis="x", alpha=0.3, zorder=0)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)
        if facet_col is not None:
            facet_title_value = _abbreviate_employee_name(facet_value, employee_name_format) if facet_col == employee_col else str(facet_value)
            ax.set_title(facet_title_value, fontsize=title_fontsize)

    axes[-1].set_xlabel("Оценка", fontsize=axis_label_fontsize)
    if title and facet_col is None:
        axes[0].set_title(title, fontsize=title_fontsize)
    elif title and facet_col is not None:
        fig.suptitle(title, fontsize=title_fontsize)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()
    return fig, axes[0] if len(axes) == 1 else axes, summary_df, details_df


def plot_competency_distributions_subplots(
    vis_df: pd.DataFrame,
    competency_col: str = "Компетенция",
    score_col: str = "score",
    role_col: str = "роль",
    employee_col: str = "Сотрудник",
    n_col: str = "n",
    roles: list[str] | None = None,
    group_filters: dict[str, str | list[str]] | None = None,
    competencies: list[str] | None = None,
    category_col: str | None = None,
    category_values: list[str] | None = None,
    filters: dict[str, str | list[str]] | None = None,
    category_label: str | None = None,

    # как агрегировать, если выбрано несколько ролей
    collapse_roles_to_employee: bool = True,
    collapse_role_agg: str = "weighted",   # "weighted" | "mean"

    # биннинг
    vmin: float = 1,
    vmax: float = 4,
    bin_step: float | None = None,
    infer_bin_step: bool = True,
    max_bins_for_wide: int = 20,

    # отсечки
    q1q3: tuple[float, float] | None = None,
    cutoffs_by_comp: dict[str, tuple[float, float]] | None = None,
    cutoffs_by_category: dict[str, tuple[float, float]] | None = None,
    auto_cutoff_mode: str = "within_category",  # "within_category" | "global"
    percentile_span: float = 25,

    # подписи на барах
    show_counts: bool | str = True,
    bar_count_min: int = 1,

    # среднее
    show_mean_line: bool = True,
    mean_color: str = "#000000",

    # оформление
    fig_width: float = 10.8,
    height_per_comp: float = 1.9,
    title: str = None,
    alpha: float = 0.85,
    low_color: str = "#FF474A",
    mid_color: str = "#E0D9CE",
    high_color: str = "#30BAAD",
    low_label: str = "",
    mid_label: str = "",
    high_label: str = "",
    show_xlabel_every_subplot: bool = True,
    show_grid: bool = False,

    # подписи категорий
    comp_labelpad: float = 18,
    comp_wrap_width: int = 22,
    comp_max_lines: int = 3,

    # шрифты
    title_fontsize: float = 14,
    tick_fontsize: float = 9,
    legend_fontsize: float = 9,
    value_fontsize: float = 8,

    # легенда
    legend_x: float = 1.02,
    auto_scale_elements: bool = True,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import textwrap

    q_low = 0.25
    q_high = 0.75
    snap_q_to_bins = True

    color_low = low_color
    color_mid = mid_color
    color_high = high_color
    rwidth = 0.95

    if not isinstance(vis_df, pd.DataFrame):
        raise TypeError("vis_df должен быть pandas.DataFrame.")
    if vmin >= vmax:
        raise ValueError("vmin должен быть меньше vmax.")
    if bin_step is not None and bin_step <= 0:
        raise ValueError("bin_step должен быть > 0.")
    if collapse_role_agg not in {"weighted", "mean"}:
        raise ValueError("collapse_role_agg должен быть 'weighted' или 'mean'.")
    if auto_cutoff_mode not in {"within_category", "global"}:
        raise ValueError("auto_cutoff_mode должен быть 'within_category' или 'global'.")
    percentile_span = _validate_percentile_span_value(percentile_span)

    resolved_category_col = category_col or competency_col
    resolved_category_values = category_values
    resolved_filters = dict(filters or {})
    resolved_cutoffs = cutoffs_by_category if cutoffs_by_category is not None else cutoffs_by_comp

    # backward compatibility
    if category_col is None and category_values is None:
        resolved_category_col = competency_col
        resolved_category_values = competencies
    if category_label is None:
        category_label = str(resolved_category_col)

    if roles is not None and resolved_category_col != role_col:
        resolved_filters[role_col] = roles
    if group_filters:
        resolved_filters.update(group_filters)
    if resolved_category_col == role_col and resolved_category_values is None and roles is not None:
        resolved_category_values = roles

    required_cols = [resolved_category_col, score_col, employee_col]
    if n_col is not None:
        required_cols.append(n_col)
    filter_cols = list(resolved_filters.keys())
    required_cols.extend(filter_cols)
    missing_cols = [c for c in list(dict.fromkeys(required_cols)) if c not in vis_df.columns]
    if missing_cols:
        raise ValueError(f"В датафрейме отсутствуют колонки: {missing_cols}")

    def wrap_category_label(text: str, width: int = 22, max_lines: int = 3) -> str:
        if pd.isna(text):
            return ""
        parts = textwrap.wrap(
            str(text),
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if len(parts) <= max_lines:
            return "\n".join(parts)
        trimmed = parts[:max_lines]
        trimmed[-1] = trimmed[-1].rstrip(" .,;:") + "..."
        return "\n".join(trimmed)

    def infer_pretty_tick_step(vmin_: float, vmax_: float, target_n: int = 8) -> float:
        span = float(vmax_ - vmin_)
        if span <= 0:
            return 1.0
        raw = span / max(target_n, 1)
        magnitude = 10 ** np.floor(np.log10(raw))
        residual = raw / magnitude
        if residual <= 1:
            nice = 1
        elif residual <= 2:
            nice = 2
        elif residual <= 5:
            nice = 5
        else:
            nice = 10
        return float(nice * magnitude)

    def collapse_to_employee_category(sub_df: pd.DataFrame) -> pd.DataFrame:
        """
        После фильтрации всегда агрегирует данные до уровня:
            сотрудник + выбранная категория.

        Это убирает лишние измерения (Компетенция / роль / group_cols),
        которые не являются выбранной характеристикой вывода.

        score:
            - weighted: взвешенное среднее по весу n
            - mean: обычное среднее по строкам
        n:
            - сумма n по строкам сотрудника внутри категории
        """
        group_cols_local = [employee_col, resolved_category_col]

        work = sub_df.copy()
        work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
        work = work[work[score_col].notna()].copy()
        if work.empty:
            return work

        if n_col in work.columns:
            work[n_col] = pd.to_numeric(work[n_col], errors="coerce").fillna(0)
        else:
            work[n_col] = 0

        if collapse_role_agg == "weighted":
            work["_effective_weight"] = work[n_col].where(work[n_col] > 0, 1.0)
            work["_weighted_score"] = work[score_col] * work["_effective_weight"]

            out = (
                work.groupby(group_cols_local, dropna=False)
                .agg(
                    _score_sum=("_weighted_score", "sum"),
                    _weight_sum=("_effective_weight", "sum"),
                    _n_sum=(n_col, "sum"),
                )
                .reset_index()
            )

            out[score_col] = np.where(
                out["_weight_sum"] > 0,
                out["_score_sum"] / out["_weight_sum"],
                np.nan,
            )
            out[n_col] = out["_n_sum"]
            out = out.drop(columns=["_score_sum", "_weight_sum", "_n_sum"])
        else:
            out = (
                work.groupby(group_cols_local, dropna=False)
                .agg(
                    **{
                        score_col: (score_col, "mean"),
                        n_col: (n_col, "sum"),
                    }
                )
                .reset_index()
            )

        return out

    df = vis_df.copy()

    for fcol, fval in resolved_filters.items():
        if fcol == resolved_category_col:
            continue
        if fcol not in df.columns:
            raise ValueError(f"Колонка фильтра '{fcol}' не найдена.")
        if isinstance(fval, (list, tuple, set)):
            df = df[df[fcol].isin(list(fval))]
        else:
            df = df[df[fcol] == fval]

    if resolved_category_values is None:
        resolved_category_values = sorted(df[resolved_category_col].dropna().astype(str).unique().tolist())
    else:
        present = set(df[resolved_category_col].dropna().astype(str).unique().tolist())
        resolved_category_values = [str(c) for c in resolved_category_values if str(c) in present]

    if not resolved_category_values:
        raise ValueError(f"Нет значений для '{category_label}' после фильтров.")

    # Для гистограммы всегда агрегируем до уровня сотрудник × выбранная категория.
    df_for_bins = collapse_to_employee_category(df)
    df_for_bins = df_for_bins[df_for_bins[resolved_category_col].astype(str).isin([str(x) for x in resolved_category_values])].copy()

    global_auto_q1 = np.nan
    global_auto_q3 = np.nan
    global_auto_mean = np.nan
    if auto_cutoff_mode == "global":
        global_scores = pd.to_numeric(df_for_bins[score_col], errors="coerce")
        global_scores = global_scores[np.isfinite(global_scores)]
        if global_scores.shape[0] > 0:
            low_p = 50.0 - float(percentile_span)
            high_p = 50.0 + float(percentile_span)
            global_auto_q1 = float(np.nanpercentile(global_scores, low_p))
            global_auto_q3 = float(np.nanpercentile(global_scores, high_p))
            global_auto_mean = float(np.nanmean(global_scores))

    range_size = vmax - vmin
    xticks = None
    used_discrete_step = None

    if bin_step is not None:
        edges = np.arange(vmin - bin_step / 2, vmax + bin_step, bin_step)
        pad_x = bin_step / 2
        xticks = np.arange(vmin, vmax + 1e-9, bin_step)
        used_discrete_step = bin_step
    else:
        if range_size > 10:
            edges = np.linspace(vmin, vmax, max_bins_for_wide + 1)
            pad_x = 0.0
            tick_step = infer_pretty_tick_step(vmin, vmax, target_n=8)
            xticks = np.arange(vmin, vmax + tick_step * 0.5, tick_step)
        else:
            inferred_step = None

            if infer_bin_step:
                all_vals = np.sort(df_for_bins[score_col].dropna().astype(float).unique())
                diffs = np.diff(all_vals)
                diffs = diffs[diffs > 1e-9]
                if len(diffs) > 0:
                    vals, cnts = np.unique(np.round(diffs, 6), return_counts=True)
                    inferred_step = float(vals[np.argmax(cnts)])

            if inferred_step is None:
                inferred_step = 1.0
            inferred_step = max(float(inferred_step), 0.25)

            edges = np.arange(vmin - inferred_step / 2, vmax + inferred_step, inferred_step)
            pad_x = inferred_step / 2
            xticks = np.arange(vmin, vmax + 1e-9, inferred_step)
            used_discrete_step = inferred_step

    x_left = vmin - pad_x
    x_right = vmax + pad_x

    def snap_to_edge(x: float, edges_arr: np.ndarray) -> float:
        idx = int(np.argmin(np.abs(edges_arr - x)))
        return float(edges_arr[idx])

    def format_axis_value(x: float) -> str:
        x = float(x)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        if abs(x * 10 - round(x * 10)) < 1e-9:
            return f"{x:.1f}"
        return f"{x:.2f}"

    bin_centers = (edges[:-1] + edges[1:]) / 2

    base_fig_width = 10.8
    base_height_per_comp = 1.9
    scale_factor = 1.0
    if auto_scale_elements:
        width_scale = fig_width / base_fig_width if base_fig_width > 0 else 1.0
        height_scale = height_per_comp / base_height_per_comp if base_height_per_comp > 0 else 1.0
        scale_factor = max(0.6, min(1.4, (width_scale * height_scale) ** 0.5))

    legend_fontsize_scaled = max(6.0, float(legend_fontsize) * scale_factor)
    tick_fontsize_scaled = max(6.0, float(tick_fontsize) * scale_factor)
    xlabel_fontsize_scaled = max(8.0, 10.0 * scale_factor)
    ylabel_fontsize_scaled = max(8.0, 10.0 * scale_factor)
    title_fontsize_scaled = max(6.0, float(title_fontsize) * scale_factor)
    comp_labelpad_scaled = comp_labelpad * scale_factor
    mean_linewidth_scaled = max(1.2, 2.0 * scale_factor)
    bar_fontsize_absolute = max(6.0, float(value_fontsize) * scale_factor)
    bar_fontsize_percent = max(6.0, float(value_fontsize) * scale_factor)

    fig, axes = plt.subplots(
        nrows=len(resolved_category_values),
        ncols=1,
        figsize=(fig_width, height_per_comp * len(resolved_category_values)),
        sharex=True,
    )
    if len(resolved_category_values) == 1:
        axes = [axes]

    stats_rows = []

    for ax, category_value in zip(axes, resolved_category_values):
        sub_df = df[df[resolved_category_col].astype(str) == str(category_value)].copy()
        sub_df = sub_df[sub_df[score_col].notna()].copy()

        if sub_df.empty:
            ax.set_visible(False)
            continue

        sub_df = collapse_to_employee_category(sub_df)

        sub = pd.to_numeric(sub_df[score_col], errors="coerce").dropna().astype(float)

        if sub.empty:
            ax.set_visible(False)
            continue

        cutoff_source = "auto_within_category"
        if resolved_cutoffs and str(category_value) in resolved_cutoffs:
            q1, q3 = resolved_cutoffs[str(category_value)]
            cutoff_source = "manual_category"
        elif q1q3 is not None:
            q1, q3 = q1q3
            cutoff_source = "manual_common"
        else:
            if auto_cutoff_mode == "global" and np.isfinite(global_auto_q1) and np.isfinite(global_auto_q3):
                q1 = float(global_auto_q1)
                q3 = float(global_auto_q3)
                cutoff_source = "auto_global"
            else:
                low_p = 50.0 - float(percentile_span)
                high_p = 50.0 + float(percentile_span)
                q1 = float(np.nanpercentile(sub, low_p))
                q3 = float(np.nanpercentile(sub, high_p))
                cutoff_source = "auto_within_category"

        q1_used, q3_used = float(q1), float(q3)

        if snap_q_to_bins:
            q1_used = snap_to_edge(q1_used, edges)
            q3_used = snap_to_edge(q3_used, edges)
            if q3_used <= q1_used:
                i = int(np.searchsorted(edges, q1_used, side="left"))
                if i + 1 < len(edges):
                    q3_used = float(edges[i + 1])
                elif i - 1 >= 0:
                    q1_used = float(edges[i - 1])

        low = sub[sub < q1_used]
        mid = sub[(sub >= q1_used) & (sub <= q3_used)]
        high = sub[sub > q3_used]

        n_low = int(low.shape[0])
        n_mid = int(mid.shape[0])
        n_high = int(high.shape[0])

        c_low, _, _ = ax.hist(low, bins=edges, alpha=alpha, color=color_low, rwidth=rwidth)
        c_mid, _, _ = ax.hist(mid, bins=edges, alpha=alpha, color=color_mid, rwidth=rwidth)
        c_high, _, _ = ax.hist(high, bins=edges, alpha=alpha, color=color_high, rwidth=rwidth)

        mean_val = float(sub.mean())
        if auto_cutoff_mode == "global" and np.isfinite(global_auto_mean):
            mean_line_value = float(global_auto_mean)
        else:
            mean_line_value = float(mean_val)
        mean_used = snap_to_edge(mean_line_value, edges) if snap_q_to_bins else mean_line_value
        if show_mean_line:
            ax.axvline(mean_used, linewidth=mean_linewidth_scaled, color=mean_color, linestyle="-")

        if show_counts in (False, "none", "None", "Ничего"):
            bar_labels_mode = "none"
        elif show_counts in ("percent", "Проценты"):
            bar_labels_mode = "percent"
        else:
            bar_labels_mode = "absolute"

        c_total = c_low + c_mid + c_high
        if bar_labels_mode != "none":
            total_obs = max(len(sub), 1)
            for x, y in zip(bin_centers, c_total):
                y_int = int(round(y))
                if y_int >= bar_count_min and y > 0:
                    if bar_labels_mode == "percent":
                        pct = 100.0 * float(y) / float(total_obs)
                        text_value = f"{pct:.1f}%"
                        if text_value.endswith(".0%"):
                            text_value = text_value.replace(".0%", "%")
                    else:
                        text_value = str(y_int)

                    ax.text(
                        x,
                        y,
                        text_value,
                        ha="center",
                        va="bottom",
                        fontsize=bar_fontsize_percent if bar_labels_mode == "percent" else bar_fontsize_absolute,
                    )

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_low, alpha=alpha),
            plt.Rectangle((0, 0), 1, 1, color=color_mid, alpha=alpha),
            plt.Rectangle((0, 0), 1, 1, color=color_high, alpha=alpha),
        ]
        labels = [
            f"{low_label}: {n_low}" if str(low_label).strip() else str(n_low),
            f"{mid_label}: {n_mid}" if str(mid_label).strip() else str(n_mid),
            f"{high_label}: {n_high}" if str(high_label).strip() else str(n_high),
        ]

        if show_mean_line:
            handles += [plt.Line2D([0], [0], color=mean_color, linewidth=mean_linewidth_scaled, linestyle="-")]
            labels += [f"Среднее={mean_used:.2f}"]

        ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(legend_x, 1.0),
            frameon=False,
            fontsize=legend_fontsize_scaled,
            borderaxespad=0.0,
        )

        ax.set_yticks([])
        ax.set_xlim(x_left, x_right)

        if xticks is not None:
            ax.set_xticks(xticks)
            ax.set_xticklabels([format_axis_value(v) for v in xticks], fontsize=tick_fontsize_scaled)

        is_last_axis = (ax is axes[-1])
        if show_xlabel_every_subplot or is_last_axis:
            ax.set_xlabel("Баллы", fontsize=xlabel_fontsize_scaled)
            ax.tick_params(axis="x", labelbottom=True, bottom=True, labelsize=tick_fontsize_scaled)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False, bottom=False)

        if show_grid:
            ax.grid(axis="x", color="#D0D0D0", linewidth=0.7, alpha=0.5)
        else:
            ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color("#B0B0B0")
        ax.spines["bottom"].set_linewidth(1.0)

        category_text = wrap_category_label(category_value, width=comp_wrap_width, max_lines=comp_max_lines)
        ax.set_ylabel(
            category_text,
            rotation=0,
            ha="right",
            va="center",
            labelpad=comp_labelpad_scaled,
            fontsize=ylabel_fontsize_scaled,
        )

        n_employees = int(sub_df[employee_col].nunique()) if employee_col in sub_df.columns else int(sub.shape[0])
        n_respondents_total = int(pd.to_numeric(sub_df[n_col], errors="coerce").fillna(0).sum()) if n_col in sub_df.columns else int(sub.shape[0])

        stats_rows.append({
            category_label: category_value,
            "n_employees": n_employees,
            "n_respondents_total": n_respondents_total,
            "mean_raw": float(mean_val),
            "mean_used": float(mean_used),
            "q1_input": float(q1),
            "q3_input": float(q3),
            "q1_used": float(q1_used),
            "q3_used": float(q3_used),
            "n_low": n_low,
            "n_mid": n_mid,
            "n_high": n_high,
            "min": float(sub.min()),
            "max": float(sub.max()),
            "bin_step_used": used_discrete_step,
            "cutoff_source": cutoff_source,
            "auto_cutoff_mode": auto_cutoff_mode,
            "percentile_span": float(percentile_span),
            "global_q1": float(global_auto_q1) if np.isfinite(global_auto_q1) else np.nan,
            "global_q3": float(global_auto_q3) if np.isfinite(global_auto_q3) else np.nan,
            "global_mean": float(global_auto_mean) if np.isfinite(global_auto_mean) else np.nan,
        })

    if (not show_xlabel_every_subplot) and (len(axes) > 0):
        axes[-1].set_xlabel("Баллы", fontsize=xlabel_fontsize_scaled)
        axes[-1].tick_params(axis="x", labelbottom=True, bottom=True, labelsize=tick_fontsize_scaled)
    if title is not None:
        fig.suptitle(title, fontsize=title_fontsize_scaled)
    fig.tight_layout(rect=[0, 0, 0.84, 0.96])

    stats_df = pd.DataFrame(stats_rows)
    return fig, axes, stats_df


def plot_ridgeline_by_group(
    vis_df,
    category_col,
    category_values=None,
    filters=None,
    facet_col=None,
    facet_values=None,
    category_label=None,
    facet_label=None,
    score_col="score",
    employee_col="Сотрудник",
    n_col="n",
    agg_mode: str = "weighted",
    vmin=None,
    vmax=None,
    x_tick_step: float | None = None,
    bandwidth=1.0,
    ridge_height=0.8,
    scale=0.9,
    overlay_ridges=True,
    ridge_overlap=0.65,
    ridge_alpha=0.45,
    ridge_palette='default',
    group_label_mode="axis",
    sort_groups_by=None,
    sort_ascending=False,
    show_reference_median=True,
    show_median=True,
    show_mean_line=False,
    median_point_size=42,
    show_intervals=True,
    interval_levels=(0.50, 0.80, 0.95),
    show_n_right=False,
    stat_label_mode="none",
    figsize_per_facet=(11, 4.8),
    ncols=1,
    title_fontsize: float = 13,
    tick_fontsize: float = 10,
    legend_fontsize: float = 9,
    value_fontsize: float = 9,
    show_xlabel_every_subplot: bool = False,
    show_grid: bool = True,
    legend_position: str = "top",
):
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import colors as mcolors

    PALETTE = {
        "background": "#FFFFFF",
        "main": {"black": "#121111", "gray": "#E8E3DE"},
        "grid": {"x": "#B8C2CC", "spine": "#8C99A5"},
    }
    RIDGE_PALETTES = {
        'default': ['#4E79A7', '#E15759', '#76B7B2', '#F28E2B', '#59A14F', '#B07AA1', '#9C755F'],
        'cool': ['#4E79A7', '#76B7B2', '#59A14F', '#5DA5DA', '#60BD68', '#17BECF', '#2F4B7C'],
        'warm': ['#E15759', '#F28E2B', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#D37295'],
        'pastel': ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#DEBB9B', '#FAB0E4'],
    }
    ridge_colors = list(ridge_palette) if isinstance(ridge_palette, (list, tuple)) and len(ridge_palette) > 0 else RIDGE_PALETTES.get(ridge_palette, RIDGE_PALETTES['default'])

    x_pad_left = 0.25
    x_pad_right = 0.65

    if not isinstance(vis_df, pd.DataFrame):
        raise TypeError('vis_df должен быть pandas.DataFrame.')
    if agg_mode not in {'weighted', 'mean'}:
        raise ValueError("agg_mode должен быть 'weighted' или 'mean'.")
    if group_label_mode not in {'axis', 'legend'}:
        raise ValueError("group_label_mode должен быть 'axis' или 'legend'.")
    if stat_label_mode not in {'none', 'median', 'median_quartiles'}:
        raise ValueError("stat_label_mode должен быть 'none', 'median' или 'median_quartiles'.")
    if legend_position not in {'top', 'bottom', 'none'}:
        raise ValueError("legend_position должен быть 'top', 'bottom' или 'none'.")
    if sort_groups_by not in {None, 'mean', 'median'}:
        raise ValueError("sort_groups_by должен быть None, 'mean' или 'median'.")
    if x_tick_step is not None and x_tick_step <= 0:
        raise ValueError('x_tick_step должен быть > 0.')
    if vmin is not None and vmax is not None and vmin >= vmax:
        raise ValueError('vmin должен быть меньше vmax.')
    if bandwidth <= 0:
        raise ValueError('bandwidth должен быть > 0.')
    if ridge_height <= 0 or scale <= 0:
        raise ValueError('ridge_height и scale должны быть > 0.')
    if not (0 < ridge_alpha <= 1):
        raise ValueError('ridge_alpha должен быть в диапазоне (0, 1].')
    if median_point_size <= 0:
        raise ValueError('median_point_size должен быть > 0.')
    if not (0 <= ridge_overlap < 1):
        raise ValueError('ridge_overlap должен быть в диапазоне [0, 1).')
    if ncols < 1:
        raise ValueError('ncols должен быть >= 1.')

    def with_alpha(color, alpha):
        r, g, b = mcolors.to_rgb(color)
        return (r, g, b, alpha)

    def blend_with_white(color, amount=0.2):
        rgb = np.array(mcolors.to_rgb(color))
        white = np.array([1.0, 1.0, 1.0])
        out = rgb * (1 - amount) + white * amount
        return tuple(out)

    def blend_with_black(color, amount=0.15):
        rgb = np.array(mcolors.to_rgb(color))
        black = np.array([0.0, 0.0, 0.0])
        out = rgb * (1 - amount) + black * amount
        return tuple(out)

    def get_median_color(ridge_color):
        return blend_with_black(ridge_color, 0.18)

    def get_interval_color(ridge_color, mass):
        lighten_map = {0.50: 0.00, 0.80: 0.10, 0.95: 0.18}
        return blend_with_white(ridge_color, lighten_map.get(mass, 0.08))

    def get_interval_alpha(mass):
        alpha_map = {0.50: min(1.0, ridge_alpha + 0.55), 0.80: min(1.0, ridge_alpha + 0.40), 0.95: min(1.0, ridge_alpha + 0.28)}
        return alpha_map.get(mass, min(1.0, ridge_alpha + 0.35))

    def get_ridge_step():
        if not overlay_ridges:
            return 1.0
        visible_height = ridge_height * scale
        step = visible_height * (1 - ridge_overlap)
        return max(step, 0.08)

    def _silverman_bandwidth(scores: np.ndarray) -> float:
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        n = len(scores)
        if n < 2:
            return 0.1
        std = np.std(scores, ddof=1)
        q75, q25 = np.percentile(scores, [75, 25])
        iqr = q75 - q25
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(std, 0.1)
        bw = 0.9 * sigma * n ** (-1 / 5)
        return float(max(bw, 1e-3))

    def _numpy_gaussian_kde_1d(scores: np.ndarray, grid: np.ndarray, bandwidth_multiplier=1.0) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        grid = np.asarray(grid, dtype=float)
        if scores.size == 0:
            return np.zeros_like(grid)
        if scores.size == 1:
            mu = float(scores[0])
            sigma = max(0.1, 0.08 * max(float(vmax) - float(vmin), 1.0))
            z = (grid - mu) / sigma
            return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi))
        base_bw = _silverman_bandwidth(scores)
        h = max(base_bw * float(bandwidth_multiplier), 1e-3)
        z = (grid[:, None] - scores[None, :]) / h
        density = np.exp(-0.5 * z ** 2).sum(axis=1) / (scores.size * h * np.sqrt(2 * np.pi))
        return density

    def quantile_interval(arr, mass):
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        lo = (1 - mass) / 2
        hi = 1 - lo
        return np.quantile(arr, [lo, hi])

    def format_stat_value(x: float) -> str:
        x = float(x)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        if abs(x * 10 - round(x * 10)) < 1e-9:
            return f"{x:.1f}"
        return f"{x:.2f}"

    def add_stat_annotations(ax, median_val, q1_val, q3_val, y0, ridge_color):
        if stat_label_mode == 'none':
            return
        text_color = blend_with_black(ridge_color, 0.28)
        ax.annotate(
            format_stat_value(median_val),
            xy=(median_val, y0),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=float(value_fontsize), color=text_color, fontweight='semibold', zorder=8,
        )
        if stat_label_mode == 'median_quartiles':
            ax.annotate(format_stat_value(q1_val), xy=(q1_val, y0), xytext=(-2, 2), textcoords='offset points',
                        ha='right', va='bottom', fontsize=float(value_fontsize), color=text_color, fontweight='semibold', zorder=8)
            ax.annotate(format_stat_value(q3_val), xy=(q3_val, y0), xytext=(2, 2), textcoords='offset points',
                        ha='left', va='bottom', fontsize=float(value_fontsize), color=text_color, fontweight='semibold', zorder=8)

    def collapse_scores(work: pd.DataFrame, group_cols_local: list[str]) -> pd.DataFrame:
        out = work.copy()
        out[score_col] = pd.to_numeric(out[score_col], errors='coerce')
        out = out[out[score_col].notna()].copy()
        if out.empty:
            return out
        if n_col in out.columns:
            out[n_col] = pd.to_numeric(out[n_col], errors='coerce').fillna(0)
        else:
            out[n_col] = 0
        if agg_mode == 'weighted':
            out['_w'] = out[n_col].where(out[n_col] > 0, 1.0)
            out['_ws'] = out[score_col] * out['_w']
            g = out.groupby(group_cols_local, dropna=False).agg(_wsum=('_ws', 'sum'), _wsum2=('_w', 'sum'), _nsum=(n_col, 'sum')).reset_index()
            g[score_col] = np.where(g['_wsum2'] > 0, g['_wsum'] / g['_wsum2'], np.nan)
            g[n_col] = g['_nsum']
            return g[group_cols_local + [score_col, n_col]]
        g = out.groupby(group_cols_local, dropna=False).agg(**{score_col: (score_col, 'mean'), n_col: (n_col, 'sum')}).reset_index()
        return g

    def compute_legend_layout(count: int, fig_width_in: float) -> tuple[int, int]:
        if count <= 0:
            return 1, 0
        # Стараемся растянуть легенду по ширине фигуры, чтобы уменьшить число строк.
        # Одна колонка обычно комфортно занимает ~1.55 дюйма с учетом handle + text.
        max_cols_by_width = max(1, int((fig_width_in - 0.6) / 1.55))
        if count <= 3:
            ncol = count
        else:
            ncol = min(count, max(4, max_cols_by_width))
        nrows = int(math.ceil(count / max(ncol, 1)))
        return ncol, nrows

    df = vis_df.copy()
    required_cols = [category_col, score_col, employee_col]
    if facet_col is not None:
        required_cols.append(facet_col)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f'В датафрейме отсутствуют колонки: {missing}')

    if filters:
        for col, val in filters.items():
            if col not in df.columns:
                raise ValueError(f"Колонка фильтра '{col}' не найдена.")
            if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                df = df[df[col].isin(list(val))]
            else:
                df = df[df[col] == val]

    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df = df[df[score_col].notna()].copy()
    if df.empty:
        raise ValueError('После фильтрации не осталось данных.')

    df[category_col] = df[category_col].astype(str)
    if category_values is None:
        category_values = sorted(df[category_col].dropna().unique().tolist(), key=lambda x: str(x))
    else:
        present = set(df[category_col].dropna().astype(str).unique().tolist())
        category_values = [str(v) for v in category_values if str(v) in present]
    if not category_values:
        raise ValueError('Нет значений выбранной характеристики после фильтрации.')

    if facet_col is not None:
        df[facet_col] = df[facet_col].astype(str)
        if facet_values is None:
            facet_values = sorted(df[facet_col].dropna().unique().tolist(), key=lambda x: str(x))
        else:
            present_f = set(df[facet_col].dropna().astype(str).unique().tolist())
            facet_values = [str(v) for v in facet_values if str(v) in present_f]
        if not facet_values:
            raise ValueError('Нет значений дополнительной разбивки после фильтрации.')
    else:
        facet_values = ['__single_facet__']

    score_min = float(df[score_col].min())
    score_max = float(df[score_col].max())
    if vmin is None:
        vmin = np.floor(score_min * 2) / 2
    if vmax is None:
        vmax = np.ceil(score_max * 2) / 2
    if vmin >= vmax:
        raise ValueError('После расчета границ оказалось, что vmin должен быть меньше vmax.')

    def infer_pretty_tick_step(vmin_, vmax_, target_n=8):
        span = float(vmax_ - vmin_)
        if span <= 0:
            return 1.0
        raw = span / max(target_n, 1)
        magnitude = 10 ** np.floor(np.log10(raw))
        residual = raw / magnitude
        if residual <= 1:
            nice = 1
        elif residual <= 2:
            nice = 2
        elif residual <= 5:
            nice = 5
        else:
            nice = 10
        return float(nice * magnitude)

    x_step = infer_pretty_tick_step(vmin, vmax, target_n=8) if x_tick_step is None else x_tick_step
    x = np.linspace(vmin - x_pad_left, vmax + x_pad_right, 1000)

    facet_count = len(facet_values)
    ncols = int(min(max(1, ncols), facet_count))
    nrows = int(math.ceil(facet_count / ncols))
    fig_w, fig_h = figsize_per_facet
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w * ncols, fig_h * nrows), squeeze=False, sharex=True)
    axes_flat = axes.flatten()
    stats_rows = []
    global_legend_handles = []

    for ax_idx, facet_value in enumerate(facet_values):
        ax = axes_flat[ax_idx]
        facet_df = df.copy() if facet_col is None else df[df[facet_col].astype(str) == str(facet_value)].copy()
        if facet_df.empty:
            ax.set_visible(False)
            continue

        ref_df = collapse_scores(facet_df, [employee_col])
        ref_scores = pd.to_numeric(ref_df[score_col], errors='coerce').dropna().astype(float).to_numpy()
        reference_median = float(np.median(ref_scores)) if ref_scores.size > 0 else np.nan

        ridge_df = collapse_scores(facet_df, [employee_col, category_col])
        ridge_df[category_col] = ridge_df[category_col].astype(str)
        ridge_df = ridge_df[ridge_df[category_col].isin([str(v) for v in category_values])].copy()
        if ridge_df.empty:
            ax.set_visible(False)
            continue

        display_categories = [str(v) for v in category_values if str(v) in set(ridge_df[category_col])]
        if sort_groups_by is not None:
            stat_map = ridge_df.groupby(category_col)[score_col].agg(sort_groups_by)
            display_categories = sorted(display_categories, key=lambda c: stat_map.get(c, np.nan), reverse=not sort_ascending)
        elif sort_ascending:
            display_categories = list(reversed(display_categories))

        plot_categories = list(reversed(display_categories))
        group_colors = {g: ridge_colors[i % len(ridge_colors)] for i, g in enumerate(display_categories)}
        ridge_step = get_ridge_step()
        y_max = 0.0
        yticks, ylabels = [], []

        for i, cat_value in enumerate(plot_categories):
            sub = ridge_df[ridge_df[category_col] == str(cat_value)].copy()
            scores = pd.to_numeric(sub[score_col], errors='coerce').dropna().astype(float).to_numpy()
            scores = scores[np.isfinite(scores)]
            if scores.size == 0:
                continue

            ridge_color = group_colors[cat_value]
            median_color = get_median_color(ridge_color)
            kde = _numpy_gaussian_kde_1d(scores, x, bandwidth_multiplier=float(bandwidth))
            if np.max(kde) > 0:
                kde = kde / np.max(kde)
            y0 = i if not overlay_ridges else i * ridge_step
            y = y0 + kde * ridge_height * scale

            ax.fill_between(x, y0, y, color=with_alpha(ridge_color, ridge_alpha), linewidth=0, zorder=2)
            ax.plot(x, y, color=with_alpha(blend_with_black(ridge_color, 0.10), min(1.0, ridge_alpha + 0.20)), linewidth=1.4, zorder=3)

            med = float(np.median(scores))
            mean_v = float(np.mean(scores))
            q1, q3 = quantile_interval(scores, 0.50)

            if show_intervals:
                for mass in sorted(set(interval_levels)):
                    if mass <= 0 or mass >= 1:
                        continue
                    lo, hi = quantile_interval(scores, mass)
                    interval_color = get_interval_color(ridge_color, mass)
                    ax.hlines(y0, lo, hi, color=with_alpha(interval_color, get_interval_alpha(mass)), linewidth=3.6 if mass == 0.50 else 2.2 if mass == 0.80 else 1.5, zorder=5)

            if show_median:
                ax.scatter(med, y0, s=median_point_size, marker='o', color=median_color,
                           edgecolor=with_alpha(blend_with_white(ridge_color, 0.55), 1.0), linewidth=0.9, zorder=7)

            if show_mean_line:
                ax.scatter(mean_v, y0, s=max(26.0, median_point_size * 0.92), marker='D', color=PALETTE['main']['black'],
                           edgecolor=PALETTE['background'], linewidth=0.8, zorder=7)

            add_stat_annotations(ax, med, q1, q3, y0, ridge_color)

            if show_n_right:
                ax.text(vmax + x_pad_right * 0.28, y0, f'n={len(scores)}', ha='left', va='center', fontsize=float(value_fontsize), color=PALETTE['main']['black'], zorder=8)

            yticks.append(y0)
            ylabels.append(str(cat_value))
            y_max = max(y_max, float(np.max(y)) if y.size else y0)
            stats_rows.append({
                'facet': None if facet_col is None else str(facet_value),
                'category': str(cat_value),
                'n_employees': int(len(scores)),
                'mean': float(mean_v),
                'median': float(med),
                'q25': float(q1),
                'q75': float(q3),
                'reference_median': float(reference_median) if np.isfinite(reference_median) else np.nan,
            })

        if show_reference_median and np.isfinite(reference_median):
            ax.axvline(reference_median, color=PALETTE['main']['black'], linestyle=(0, (4, 4)), linewidth=1.4, zorder=4)

        ax.set_xlim(vmin, vmax + x_pad_right * 0.55)
        ax.set_ylim(-0.18, y_max + ridge_height * scale * 0.55)
        if group_label_mode == 'axis':
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=float(tick_fontsize))
        else:
            ax.set_yticks([])

        if show_grid:
            ax.grid(axis='x', color=PALETTE['grid']['x'], alpha=0.35, linewidth=0.8)
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('#BDBDBD')
        ax.spines['bottom'].set_linewidth(1.0)
        ax.tick_params(axis='x', labelsize=float(tick_fontsize), colors='#7A7A7A')
        ax.tick_params(axis='y', length=0)
        if x_step is not None:
            ax.set_xticks(np.arange(float(vmin), float(vmax) + x_step * 0.5, x_step))
        if facet_col is not None:
            ax.set_title(f'{facet_label or facet_col}: {facet_value}', fontsize=float(title_fontsize))
        if show_xlabel_every_subplot or ax_idx >= facet_count - ncols:
            ax.set_xlabel('Оценка', fontsize=float(tick_fontsize), color='#7A7A7A')
            ax.tick_params(axis='x', labelbottom=True, colors='#7A7A7A')
        else:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False, colors='#BDBDBD')

        if not global_legend_handles:
            if show_median:
                global_legend_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=get_median_color(ridge_colors[0]), markeredgecolor=blend_with_white(ridge_colors[0], 0.55), markersize=7, label='Медиана'))
            if show_intervals and 0.50 in set(interval_levels):
                global_legend_handles.append(Line2D([0], [0], color=get_interval_color(ridge_colors[0], 0.50), linewidth=3.6, label='50% интервал'))
            if show_reference_median:
                global_legend_handles.append(Line2D([0], [0], color=PALETTE['main']['black'], linestyle=(0, (4, 4)), linewidth=1.4, label='Референсная медиана'))
            if show_mean_line:
                global_legend_handles.append(Line2D([0], [0], marker='D', color='none', markerfacecolor=PALETTE['main']['black'], markeredgecolor=PALETTE['background'], markersize=6.5, label='Среднее'))
            if group_label_mode == 'legend':
                for g in display_categories:
                    global_legend_handles.append(Patch(facecolor=with_alpha(group_colors[g], ridge_alpha), edgecolor='none', label=str(g)))

    for ax in axes_flat[facet_count:]:
        ax.set_visible(False)

    if legend_position != 'none' and global_legend_handles:
        fig_width_in = max(float(fig.get_size_inches()[0]), 1.0)
        fig_height_in = max(float(fig.get_size_inches()[1]), 1.0)
        legend_ncol, legend_nrows = compute_legend_layout(len(global_legend_handles), fig_width_in)

        side_pad_in = 0.35
        left_rect = side_pad_in / fig_width_in
        right_rect = 1.0 - left_rect

        # Резервируем место под легенду в фиксированных физических единицах,
        # но учитываем реальное количество строк легенды.
        legend_row_in = max(0.26, float(legend_fontsize) * 0.040)
        legend_block_in = 0.18 + legend_nrows * legend_row_in
        title_gap_in = 0.14 if facet_col is not None else 0.08

        if legend_position == 'top':
            top_pad_in = legend_block_in + title_gap_in
            legend_y = 1.0 - (0.10 / fig_height_in)
            fig.legend(
                handles=global_legend_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, legend_y),
                ncol=legend_ncol,
                frameon=False,
                fontsize=float(legend_fontsize),
                columnspacing=1.2,
                handletextpad=0.5,
                borderaxespad=0.0,
            )
            top_rect = max(0.55, 1.0 - (top_pad_in / fig_height_in))
            fig.tight_layout(rect=[left_rect, 0.05, right_rect, top_rect])
        else:
            bottom_pad_in = legend_block_in + 0.10
            legend_y = 0.10 / fig_height_in
            fig.legend(
                handles=global_legend_handles,
                loc='lower center',
                bbox_to_anchor=(0.5, legend_y),
                ncol=legend_ncol,
                frameon=False,
                fontsize=float(legend_fontsize),
                columnspacing=1.2,
                handletextpad=0.5,
                borderaxespad=0.0,
            )
            bottom_rect = min(0.42, bottom_pad_in / fig_height_in)
            fig.tight_layout(rect=[left_rect, bottom_rect, right_rect, 0.98])
    else:
        fig_width_in = max(float(fig.get_size_inches()[0]), 1.0)
        side_pad_in = 0.35
        left_rect = side_pad_in / fig_width_in
        right_rect = 1.0 - left_rect
        fig.tight_layout(rect=[left_rect, 0.05, right_rect, 0.98])
    return fig, axes, pd.DataFrame(stats_rows)

def plot_ridgeline_roles_for_competency(
    vis_df,
    competency=None,
    competency_col="Компетенция",
    score_col="score",
    role_col="роль",
    emp_col="Сотрудник",
    group_filters=None,
    role_order=None,

    vmin=None,
    vmax=None,
    x_tick_step: float | None = None,

    bandwidth=1.0,
    ridge_height=0.8,
    scale=0.9,

    overlay_ridges=True,
    ridge_overlap=0.65,
    ridge_alpha=0.45,

    group_label_mode="axis",

    show_median=True,
    show_mean_line=False,
    median_point_size=42,

    show_intervals=True,
    interval_levels=(0.50, 0.80, 0.95),

    show_n_right=False,

    show_reference_median=True,
    reference_mode="role",
    reference_value=None,
    stat_label_mode="none",

    figsize=(11, 6.5),
    ridge_palette='default',
    title_fontsize: float = 13,
    tick_fontsize: float = 10,
    legend_fontsize: float = 9,
    value_fontsize: float = 9,
    show_grid: bool = True,
    legend_position: str = "top",
):
    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib import colors as mcolors

    PALETTE = {
        "background": "#FFFFFF",
        "main": {"black": "#121111", "gray": "#E8E3DE"},
        "grid": {"x": "#B8C2CC", "spine": "#8C99A5"},
    }

    RIDGE_PALETTES = {
        'default': ['#4E79A7', '#E15759', '#76B7B2', '#F28E2B', '#59A14F', '#B07AA1', '#9C755F'],
        'cool': ['#4E79A7', '#76B7B2', '#59A14F', '#5DA5DA', '#60BD68', '#17BECF', '#2F4B7C'],
        'warm': ['#E15759', '#F28E2B', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#D37295'],
        'pastel': ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#DEBB9B', '#FAB0E4'],
    }
    if isinstance(ridge_palette, (list, tuple)) and len(ridge_palette) > 0:
        RIDGE_COLORS = list(ridge_palette)
    else:
        RIDGE_COLORS = RIDGE_PALETTES.get(ridge_palette, RIDGE_PALETTES['default'])

    x_pad_left = 0.25
    x_pad_right = 0.65
    interval_y_offset = 0.0

    if bandwidth <= 0: raise ValueError("bandwidth должен быть > 0.")
    if ridge_height <= 0: raise ValueError("ridge_height должен быть > 0.")
    if scale <= 0: raise ValueError("scale должен быть > 0.")
    if not (0 < ridge_alpha <= 1): raise ValueError("ridge_alpha должен быть в диапазоне (0, 1].")
    if median_point_size <= 0: raise ValueError("median_point_size должен быть > 0.")
    if x_tick_step is not None and x_tick_step <= 0: raise ValueError("x_tick_step должен быть > 0.")
    if vmin is not None and vmax is not None and vmin >= vmax: raise ValueError("vmin должен быть меньше vmax.")
    if not (0 <= ridge_overlap < 1): raise ValueError("ridge_overlap должен быть в диапазоне [0, 1).")
    if group_label_mode not in {"axis", "legend"}: raise ValueError("group_label_mode должен быть 'axis' или 'legend'.")
    if figsize is None or len(figsize) != 2: raise ValueError("figsize должен быть кортежем из двух чисел: (width, height).")
    if figsize[0] <= 0 or figsize[1] <= 0: raise ValueError("Оба значения figsize должны быть > 0.")
    if not isinstance(interval_levels, (list, tuple)) or len(interval_levels) == 0: raise ValueError("interval_levels должен быть непустым списком или кортежем.")
    if any((lvl <= 0 or lvl >= 1) for lvl in interval_levels): raise ValueError("Все значения interval_levels должны быть строго между 0 и 1.")

    valid_reference_modes = {None, "role", "roles", "exclude_roles"}
    if reference_mode not in valid_reference_modes: raise ValueError("reference_mode должен быть None, 'role', 'roles' или 'exclude_roles'.")
    if stat_label_mode not in {"none", "median", "median_quartiles"}: raise ValueError("stat_label_mode должен быть 'none', 'median' или 'median_quartiles'.")
    if reference_mode in {"roles", "exclude_roles"} and reference_value is not None and not isinstance(reference_value, (list, tuple, set)):
        raise ValueError("Для reference_mode='roles' или 'exclude_roles' reference_value должен быть списком ролей.")
    if reference_mode == "role" and reference_value is not None and isinstance(reference_value, (list, tuple, set)):
        raise ValueError("Для reference_mode='role' reference_value должен быть строкой.")

    df = vis_df.copy()
    required_cols = [score_col, role_col, emp_col]
    if competency_col not in df.columns: raise ValueError(f"В датафрейме отсутствует колонка '{competency_col}'.")
    missing = [c for c in required_cols if c not in df.columns]
    if missing: raise ValueError(f"В датафрейме отсутствуют колонки: {missing}")

    if group_filters:
        for col, val in group_filters.items():
            if col not in df.columns: raise ValueError(f"Колонка '{col}' не найдена в датафрейме.")
            if isinstance(val, (list, tuple, set, np.ndarray, pd.Series)):
                df = df[df[col].isin(list(val))]
            else:
                df = df[df[col] == val]

    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df = df[df[score_col].notna()].copy()

    if competency is not None:
        df = df[df[competency_col] == competency].copy()
        dedup_cols = [c for c in [emp_col, competency_col, role_col, score_col] if c in df.columns]
        if dedup_cols: df = df.drop_duplicates(subset=dedup_cols)
    else:
        dedup_cols = [c for c in [emp_col, competency_col, role_col, score_col] if c in df.columns]
        if dedup_cols: df = df.drop_duplicates(subset=dedup_cols)
        df = df.groupby([emp_col, role_col], dropna=False).agg(**{score_col: (score_col, 'mean')}).reset_index()

    if df.empty:
        raise ValueError("Нет данных после расчета средней по всем компетенциям." if competency is None else f"Нет данных для компетенции: {competency}")

    if role_order is None:
        present_roles = df[role_col].dropna().unique().tolist()
        role_stats = []
        for role in present_roles:
            vals = df.loc[df[role_col] == role, score_col].dropna().astype(float).to_numpy()
            if len(vals) == 0: continue
            role_stats.append((role, float(np.median(vals))))
        role_stats = sorted(role_stats, key=lambda x: x[1], reverse=True)
        role_order = [r for r, _ in role_stats]
    else:
        present = set(df[role_col].dropna().unique().tolist())
        role_order = [r for r in role_order if r in present]
    if not role_order: raise ValueError('Нет ролей после фильтров.')

    score_min = float(df[score_col].min()); score_max = float(df[score_col].max())
    if vmin is None: vmin = np.floor(score_min * 2) / 2
    if vmax is None: vmax = np.ceil(score_max * 2) / 2
    if vmin >= vmax: raise ValueError('После расчета границ оказалось, что vmin должен быть меньше vmax.')

    def infer_pretty_tick_step(vmin_, vmax_, target_n=8):
        span = float(vmax_ - vmin_)
        if span <= 0: return 1.0
        raw = span / max(target_n, 1)
        magnitude = 10 ** np.floor(np.log10(raw))
        residual = raw / magnitude
        if residual <= 1: nice = 1
        elif residual <= 2: nice = 2
        elif residual <= 5: nice = 5
        else: nice = 10
        return float(nice * magnitude)

    if x_tick_step is None:
        range_size = vmax - vmin
        if range_size <= 4: x_step = 0.5
        elif range_size <= 10: x_step = 1.0
        else: x_step = infer_pretty_tick_step(vmin, vmax, target_n=8)
    else:
        x_step = float(x_tick_step)
    x = np.linspace(vmin - x_pad_left, vmax + x_pad_right, 700)
    role_colors = {role: RIDGE_COLORS[i % len(RIDGE_COLORS)] for i, role in enumerate(role_order)}

    def with_alpha(color, alpha):
        r,g,b = mcolors.to_rgb(color); return (r,g,b,alpha)
    def blend_with_white(color, amount=0.2):
        rgb = np.array(mcolors.to_rgb(color)); white = np.array([1.0,1.0,1.0]); out = rgb*(1-amount)+white*amount; return tuple(out)
    def blend_with_black(color, amount=0.15):
        rgb = np.array(mcolors.to_rgb(color)); black = np.array([0.0,0.0,0.0]); out = rgb*(1-amount)+black*amount; return tuple(out)
    def get_median_color(ridge_color): return blend_with_black(ridge_color, 0.18)
    def get_interval_color(ridge_color, mass):
        lighten_map = {0.50: 0.00, 0.80: 0.10, 0.95: 0.18}; return blend_with_white(ridge_color, lighten_map.get(mass, 0.08))
    def get_interval_alpha(mass):
        alpha_map = {0.50: min(1.0, ridge_alpha + 0.55), 0.80: min(1.0, ridge_alpha + 0.40), 0.95: min(1.0, ridge_alpha + 0.28)}
        return alpha_map.get(mass, min(1.0, ridge_alpha + 0.35))
    def get_ridge_step():
        if not overlay_ridges: return 1.0
        visible_height = ridge_height * scale
        step = visible_height * (1 - ridge_overlap)
        return max(step, 0.08)
    def _silverman_bandwidth(scores):
        scores = np.asarray(scores, dtype=float); scores = scores[np.isfinite(scores)]; n = len(scores)
        if n < 2: return 0.1
        std = np.std(scores, ddof=1); q75, q25 = np.percentile(scores, [75,25]); iqr = q75 - q25
        sigma = min(std, iqr/1.34) if iqr > 0 else std
        if not np.isfinite(sigma) or sigma <= 0: sigma = max(std, 0.1)
        bw = 0.9 * sigma * n ** (-1/5)
        return float(max(bw, 1e-3))
    def _numpy_gaussian_kde_1d(scores, grid, bandwidth_multiplier=1.0):
        scores = np.asarray(scores, dtype=float); scores = scores[np.isfinite(scores)]; grid = np.asarray(grid, dtype=float)
        if scores.size == 0: return np.zeros_like(grid)
        if scores.size == 1:
            mu = float(scores[0]); sigma = 0.1; z = (grid - mu) / sigma
            return np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi))
        base_bw = _silverman_bandwidth(scores); h = max(base_bw * float(bandwidth_multiplier), 1e-3)
        z = (grid[:, None] - scores[None, :]) / h
        density = np.exp(-0.5 * z ** 2).sum(axis=1) / (scores.size * h * np.sqrt(2 * np.pi))
        return density
    def quantile_interval(arr, mass):
        arr = np.asarray(arr, dtype=float); arr = arr[np.isfinite(arr)]
        if arr.size == 0: raise ValueError('Нельзя посчитать интервал для пустого массива.')
        lo = (1 - mass) / 2; hi = 1 - lo
        return np.quantile(arr, [lo, hi])

    def format_stat_value(x: float) -> str:
        x = float(x)
        if abs(x - round(x)) < 1e-9: return str(int(round(x)))
        if abs(x * 10 - round(x * 10)) < 1e-9: return f"{x:.1f}"
        return f"{x:.2f}"
    def add_stat_annotations(ax, scores, y0, ridge_color):
        if stat_label_mode == "none": return
        scores = np.asarray(scores, dtype=float); scores = scores[np.isfinite(scores)]
        if len(scores) == 0: return
        median_val = float(np.median(scores)); q1_val = float(np.quantile(scores, 0.25)); q3_val = float(np.quantile(scores, 0.75))
        text_color = blend_with_black(ridge_color, 0.28)
        quartile_y = y0 + max(0.03, ridge_height * 0.02)
        ax.annotate(format_stat_value(median_val), xy=(median_val, y0), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=float(value_fontsize), color=text_color, zorder=8)
        if stat_label_mode == "median_quartiles":
            ax.annotate(format_stat_value(q1_val), xy=(q1_val, quartile_y), xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom", fontsize=float(value_fontsize), color=text_color, zorder=8)
            ax.annotate(format_stat_value(q3_val), xy=(q3_val, quartile_y), xytext=(2, 2), textcoords="offset points", ha="left", va="bottom", fontsize=float(value_fontsize), color=text_color, zorder=8)
    def make_kde_curve(scores):
        scores = np.asarray(scores, dtype=float); scores = scores[np.isfinite(scores)]
        if scores.size == 0: return np.zeros_like(x)
        unique_scores = np.unique(scores)
        if scores.size < 2 or unique_scores.size == 1:
            mu = float(np.mean(scores)); sigma = max(0.08, 0.12 * float(bandwidth)); kde_y = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        else:
            kde_y = _numpy_gaussian_kde_1d(scores, x, bandwidth_multiplier=bandwidth)
        peak = float(np.max(kde_y))
        return kde_y / peak * ridge_height * scale if peak > 0 else np.zeros_like(x)
    def compute_reference_median(df_ref):
        if not show_reference_median or reference_mode is None: return None
        if reference_mode == 'roles':
            roles_local = list(reference_value) if reference_value is not None else []
            ref_scores = df_ref.loc[df_ref[role_col].isin(roles_local), score_col].dropna().astype(float).to_numpy()
        elif reference_mode == 'exclude_roles':
            roles_local = list(reference_value) if reference_value is not None else []
            ref_scores = df_ref.loc[~df_ref[role_col].isin(roles_local), score_col].dropna().astype(float).to_numpy()
        else:
            role_name = reference_value
            if role_name in set(df_ref[role_col].dropna()):
                ref_scores = df_ref.loc[df_ref[role_col] == role_name, score_col].dropna().astype(float).to_numpy()
            else:
                ref_scores = np.array([])
        if len(ref_scores) == 0: return None
        return float(np.median(ref_scores))
    def draw_ridge(ax, scores, y0, ridge_color, n_label_value=None):
        scores = np.asarray(scores, dtype=float); scores = scores[np.isfinite(scores)]
        if len(scores) == 0: return
        kde_y = make_kde_curve(scores)
        ridge_fill_color = with_alpha(ridge_color, ridge_alpha)
        ridge_line_color = blend_with_black(ridge_color, 0.08)
        median_color = get_median_color(ridge_color)
        ax.fill_between(x, y0, y0 + kde_y, color=ridge_fill_color, linewidth=0, zorder=2)
        ax.plot(x, y0 + kde_y, color=ridge_line_color, linewidth=1.5, zorder=3)
        ax.hlines(y0, vmin - x_pad_left, vmax + x_pad_right * 0.45, color=PALETTE['main']['gray'], linewidth=1.0, zorder=1)
        med = float(np.median(scores))
        if show_intervals:
            level_styles = {0.95: dict(lw=1.8), 0.80: dict(lw=2.4), 0.50: dict(lw=4.2)}
            y_interval = y0 + interval_y_offset
            for mass in sorted(interval_levels, reverse=True):
                left, right = quantile_interval(scores, mass); style = level_styles.get(mass, dict(lw=2.0))
                ax.hlines(y=y_interval, xmin=left, xmax=right, color=get_interval_color(ridge_color, mass), linewidth=style['lw'], alpha=get_interval_alpha(mass), zorder=5)
        if show_median:
            ax.scatter(med, y0, s=median_point_size, marker='o', color=median_color, edgecolor=with_alpha(blend_with_white(ridge_color, 0.55), 1.0), linewidth=0.9, zorder=7)
        if show_mean_line:
            mean_val = float(np.mean(scores))
            ax.vlines(mean_val, y0, y0 + ridge_height * 0.9, color=blend_with_black(ridge_color, 0.28), linewidth=1.2, linestyle='--', zorder=6, alpha=0.95)
        add_stat_annotations(ax, scores, y0, ridge_color)
        if show_n_right and n_label_value is not None:
            right_x = vmax + x_pad_right * 0.18
            ax.text(right_x, y0, f'n={int(n_label_value)}', ha='left', va='center', fontsize=float(value_fontsize), color=PALETTE['main']['black'])
    def style_axis(ax, y_positions, labels):
        ax.set_xlim(vmin - x_pad_left, vmax + x_pad_right)
        ax.set_xticks(np.arange(vmin, vmax + 1e-9, x_step))
        ax.set_xlabel('Баллы', fontsize=float(tick_fontsize))
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=float(tick_fontsize))
        if show_grid:
            ax.grid(axis='x', color=PALETTE['grid']['x'], linewidth=0.7, alpha=0.65)
        else:
            ax.grid(False)
        ax.grid(axis='y', visible=False)
        if group_label_mode == 'axis':
            ax.set_yticks(y_positions); ax.set_yticklabels(labels); ax.tick_params(axis='y', length=0, labelsize=float(tick_fontsize))
        else:
            ax.set_yticks([])
        for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color(PALETTE['grid']['spine'])
        if len(y_positions) > 0:
            y_min = float(np.min(y_positions)) - 0.25; y_max = float(np.max(y_positions)) + ridge_height * scale + 0.25
            ax.set_ylim(y_min, y_max)
    def compute_group_legend_layout(n_items):
        if n_items <= 4: ncol = n_items
        elif n_items <= 8: ncol = 4
        elif n_items <= 12: ncol = 5
        else: ncol = 6
        nrows = int(math.ceil(n_items / max(ncol, 1))) if n_items > 0 else 0
        return ncol, nrows

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PALETTE['background'])
    ax.set_facecolor(PALETTE['background'])
    ridge_step = get_ridge_step()
    y_positions = (np.arange(len(role_order))[::-1] * ridge_step).astype(float)
    reference_median = compute_reference_median(df)

    for y0, role in zip(y_positions, role_order):
        role_sub = df.loc[df[role_col] == role].copy()
        scores = role_sub[score_col].dropna().astype(float).to_numpy()
        if len(scores) == 0: continue
        n_employees = role_sub[emp_col].dropna().nunique()
        draw_ridge(ax, scores, y0, role_colors.get(role, RIDGE_COLORS[0]), n_label_value=n_employees)

    if reference_median is not None and len(y_positions) > 0:
        ref_line_top = ridge_height * 0.92
        for y0 in y_positions:
            ax.vlines(reference_median, y0 - 0.02, y0 + ref_line_top, color=PALETTE['main']['black'], linestyle='-', linewidth=2.0, zorder=6.5, alpha=1.0)

    style_axis(ax, y_positions, role_order)

    top_handles = []
    if group_label_mode == 'legend':
        top_handles = [Patch(facecolor=role_colors[r], edgecolor='none', label=str(r), alpha=ridge_alpha) for r in role_order]
    bottom_handles = []
    if show_intervals:
        lw_map = {0.50: 4.2, 0.80: 2.4, 0.95: 1.8}; demo_color = RIDGE_COLORS[0]
        for mass in interval_levels:
            bottom_handles.append(Line2D([0], [0], color=get_interval_color(demo_color, mass), lw=lw_map.get(mass, 2.0), alpha=get_interval_alpha(mass), label=f'{int(mass * 100)}%'))
    if show_median:
        demo_color = RIDGE_COLORS[0]
        bottom_handles.append(Line2D([0], [0], marker='o', color='none', markerfacecolor=get_median_color(demo_color), markeredgecolor=blend_with_white(demo_color, 0.55), markersize=7, label='Медиана'))
    if show_reference_median:
        bottom_handles.append(Line2D([0], [0], color=PALETTE['main']['black'], lw=2.0, label='Референсная медиана'))

    extra_top_space = 0.0
    extra_bottom_space = 0.0
    if legend_position == 'top':
        if top_handles:
            top_ncol, top_nrows = compute_group_legend_layout(len(top_handles))
            fig.legend(handles=top_handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=top_ncol, frameon=False, fontsize=float(legend_fontsize), columnspacing=1.2, handletextpad=0.5)
            extra_top_space += 0.045 * max(top_nrows, 1)
        if bottom_handles:
            bottom_ncol = min(len(bottom_handles), 5); bottom_y = 0.995 - extra_top_space
            fig.legend(handles=bottom_handles, loc='upper center', bbox_to_anchor=(0.5, bottom_y), ncol=bottom_ncol, frameon=False, fontsize=float(legend_fontsize), columnspacing=1.2, handletextpad=0.5)
            extra_top_space += 0.05
    else:
        if top_handles:
            top_ncol, top_nrows = compute_group_legend_layout(len(top_handles))
            extra_bottom_space += 0.045 * max(top_nrows, 1)
            fig.legend(handles=top_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02 + extra_bottom_space - 0.03), ncol=top_ncol, frameon=False, fontsize=float(legend_fontsize), columnspacing=1.2, handletextpad=0.5)
        if bottom_handles:
            bottom_ncol = min(len(bottom_handles), 5)
            y = 0.02 + extra_bottom_space
            fig.legend(handles=bottom_handles, loc='lower center', bbox_to_anchor=(0.5, y), ncol=bottom_ncol, frameon=False, fontsize=float(legend_fontsize), columnspacing=1.2, handletextpad=0.5)
            extra_bottom_space += 0.06
    rect_top = max(0.82, 0.97 - extra_top_space)
    rect_bottom = min(0.18, 0.04 + extra_bottom_space)
    fig.tight_layout(rect=[0.05, rect_bottom, 0.97, rect_top])
    return fig, ax



def _prepare_emp_df(
    vis_df: pd.DataFrame,
    competency: str,
    employee_col: str,
    competency_col: str,
    score_col: str,
    n_col: str,
    roles: list[str] | None,
    agg: str,
    min_total_n: int,
    group_filters: dict[str, str | list[str]] | None,
    role_col: str | None = "роль",
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    df = vis_df.copy()

    if group_filters:
        for gcol, gval in group_filters.items():
            if gcol not in df.columns:
                raise ValueError(f"Колонка группы '{gcol}' не найдена.")
            if isinstance(gval, (list, tuple, set)):
                df = df[df[gcol].isin(list(gval))]
            else:
                df = df[df[gcol] == gval]

    if competency_col not in df.columns:
        raise ValueError(f"Колонка '{competency_col}' не найдена.")
    df = df[df[competency_col] == competency].copy()

    if df.empty:
        return pd.DataFrame()

    has_role_col = (role_col is not None) and (role_col in df.columns)

    if has_role_col and roles:
        df = df[df[role_col].isin(roles)]

    if df.empty:
        return pd.DataFrame()

    if score_col not in df.columns:
        raise ValueError(f"Колонка '{score_col}' не найдена.")
    df = df.dropna(subset=[score_col]).copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df[df[score_col].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    if n_col in df.columns:
        df[n_col] = pd.to_numeric(df[n_col], errors="coerce").fillna(0.0)
    else:
        df[n_col] = 1.0

    dedup_cols = [employee_col, competency_col, score_col]
    if has_role_col:
        dedup_cols.append(role_col)
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols)

    if agg == "weighted":
        df["_wx"] = df[score_col] * df[n_col]
        emp_df = (
            df.groupby([employee_col], as_index=False)
            .agg(wx_sum=("_wx", "sum"), total_n=(n_col, "sum"))
        )
        emp_df["employee_score"] = emp_df["wx_sum"] / emp_df["total_n"].replace(0, np.nan)
        emp_df = emp_df.drop(columns=["wx_sum"])
    elif agg == "mean":
        emp_df = (
            df.groupby([employee_col], as_index=False)
            .agg(employee_score=(score_col, "mean"), total_n=(n_col, "sum"))
        )
    else:
        raise ValueError("agg должен быть 'weighted' или 'mean'.")

    emp_df = (
        emp_df[emp_df["total_n"] >= int(min_total_n)]
        .dropna(subset=["employee_score"])
        .reset_index(drop=True)
    )

    return emp_df


def plot_competency_employee_rows_scatter(
    vis_df: pd.DataFrame,
    competencies: list[str] | None = None,
    employee_col: str = "Сотрудник",
    competency_col: str = "Компетенция",
    role_col: str | None = "роль",
    score_col: str = "score",
    n_col: str = "n",
    roles: list[str] | None = None,
    agg: str = "weighted",
    min_total_n: int = 1,
    group_filters: dict[str, str | list[str]] | None = None,
    *,
    category_col: str | None = None,
    category_values: list[str] | None = None,
    filters: dict[str, str | list[str]] | None = None,
    facet_col: str | None = None,
    facet_values: list[str] | None = None,
    category_label: str | None = None,
    facet_label: str | None = None,
    figsize: tuple[float, float] = (12, 4),
    row_gap: float = 0.8,
    employee_legend_limit: int = 10,
    x_pad_frac: float = 0.05,
    x_tick_step: float | None = None,
    show_grid: bool = False,
    mean_point_color: str | None = None,
    mean_point_size: float = 100,
    point_size: float = 80,
    alpha: float = 0.9,
    title: bool | str = False,
    employee_colors: list[str] | None = None,
    comp_wrap_width: int = 20,
    comp_max_lines: int = 3,
    show_employee_legend: bool = True,
    show_mean_value_label: bool = False,
    employee_name_format: str = 'keep',
    title_fontsize: float = 14,
    tick_fontsize: float = 10,
    legend_fontsize: float = 9,
    value_fontsize: float = 9,
):
    import math
    import textwrap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    PALETTE = {
        "background": "#FFFFFF",
        "main": {"coral": "#FF474A", "beige": "#CBBFAD", "gray": "#F3EEEB", "black": "#121111"},
        "additional": {"blue_dark": "#495867", "blue_mid": "#606E7D", "blue_light": "#899AAB", "teal_1": "#30BAAD", "teal_2": "#34CCBE", "teal_3": "#59D5C9", "sand": "#B5A798", "cream": "#E0D9CE"}
    }

    if not isinstance(vis_df, pd.DataFrame):
        raise TypeError("vis_df должен быть pandas.DataFrame.")
    if row_gap <= 0:
        raise ValueError("row_gap должен быть > 0.")
    if point_size <= 0:
        raise ValueError("point_size должен быть > 0.")
    if mean_point_size <= 0:
        raise ValueError("mean_point_size должен быть > 0.")
    if not (0 <= alpha <= 1):
        raise ValueError("alpha должен быть в диапазоне [0, 1].")
    if x_pad_frac < 0:
        raise ValueError("x_pad_frac должен быть >= 0.")
    if x_tick_step is not None and x_tick_step <= 0:
        raise ValueError("x_tick_step должен быть > 0.")
    if employee_legend_limit < 0:
        raise ValueError("employee_legend_limit должен быть >= 0.")
    if min_total_n < 1:
        raise ValueError("min_total_n должен быть >= 1.")
    if figsize is None or len(figsize) != 2:
        raise ValueError("figsize должен быть кортежем из двух чисел: (width, height).")
    if figsize[0] <= 0 or figsize[1] <= 0:
        raise ValueError("Оба значения figsize должны быть > 0.")
    if comp_wrap_width <= 0:
        raise ValueError("comp_wrap_width должен быть > 0.")
    if comp_max_lines < 1:
        raise ValueError("comp_max_lines должен быть >= 1.")
    if employee_name_format not in {'keep', 'first', 'second', 'third', 'first5', 'first3', 'first_initials', 'third_initials'}:
        raise ValueError("employee_name_format должен быть одним из: 'keep', 'first5', 'first3', 'first', 'second', 'third', 'first_initials', 'third_initials'.")

    valid_aggs = {"weighted", "mean"}
    if agg not in valid_aggs:
        raise ValueError(f"agg должен быть одним из: {sorted(valid_aggs)}")

    has_role_col = (role_col is not None) and (role_col in vis_df.columns)
    if roles is not None and not has_role_col:
        raise ValueError(
            "Передан параметр roles, но колонка роли отсутствует. "
            "Для неролевого опроса не передавайте roles."
        )

    # backward compatibility
    resolved_category_col = category_col or competency_col
    resolved_category_values = category_values if category_values is not None else competencies
    resolved_filters = dict(filters or {})
    if roles is not None and resolved_category_col != role_col and has_role_col:
        resolved_filters[role_col] = roles
    if group_filters:
        resolved_filters.update(group_filters)
    if resolved_category_col == role_col and resolved_category_values is None and roles is not None:
        resolved_category_values = roles
    if category_label is None:
        category_label = str(resolved_category_col)
    if facet_label is None and facet_col is not None:
        facet_label = str(facet_col)

    required_cols = [employee_col, resolved_category_col, score_col]
    if n_col is not None:
        required_cols.append(n_col)
    if facet_col is not None:
        required_cols.append(facet_col)
    required_cols.extend(list(resolved_filters.keys()))
    missing = [c for c in dict.fromkeys(required_cols) if c not in vis_df.columns]
    if missing:
        raise ValueError(f"В датафрейме отсутствуют колонки: {missing}")

    def wrap_label(text: str, width: int = 20, max_lines: int = 3) -> str:
        if pd.isna(text):
            return ""
        parts = textwrap.wrap(str(text), width=width, break_long_words=False, break_on_hyphens=False)
        if len(parts) <= max_lines:
            return "\n".join(parts)
        trimmed = parts[:max_lines]
        trimmed[-1] = trimmed[-1].rstrip(" .,;:") + "..."
        return "\n".join(trimmed)

    def abbreviate_employee_name(value: object) -> str:
        if pd.isna(value):
            return ""
        original = str(value).strip()
        parts = [part for part in original.split() if part]
        if len(parts) != 3 or employee_name_format == 'keep':
            return original
        keep_idx = {'first': 0, 'second': 1, 'third': 2}.get(employee_name_format, 0)
        kept = parts[keep_idx]
        abbreviated = [f"{part[0]}." for idx, part in enumerate(parts) if idx != keep_idx and part]
        return " ".join([kept] + abbreviated)

    def display_value_for_col(col_name: str | None, value: object) -> str:
        if col_name == employee_col:
            return abbreviate_employee_name(value)
        return "" if pd.isna(value) else str(value)

    def compute_xlim(score_arrays: list[np.ndarray], pad_frac: float) -> tuple[float, float]:
        allx = np.concatenate(score_arrays)
        xmin = float(np.min(allx)); xmax = float(np.max(allx))
        span = xmax - xmin if xmax != xmin else 1.0
        pad = span * float(pad_frac)
        return xmin - pad, xmax + pad

    def infer_pretty_tick_step(xmin: float, xmax: float, target_n: int = 8) -> float:
        span = float(xmax - xmin)
        if span <= 0:
            return 1.0
        raw = span / max(target_n, 1)
        magnitude = 10 ** np.floor(np.log10(raw))
        residual = raw / magnitude
        if residual <= 1: nice = 1
        elif residual <= 2: nice = 2
        elif residual <= 5: nice = 5
        else: nice = 10
        return float(nice * magnitude)

    def build_pretty_ticks(xmin: float, xmax: float, tick_step: float) -> np.ndarray:
        start = math.floor(xmin / tick_step) * tick_step
        stop = math.ceil(xmax / tick_step) * tick_step
        ticks = np.arange(start, stop + tick_step * 0.5, tick_step)
        decimals = 0
        if tick_step < 1:
            decimals = max(0, int(round(-np.log10(tick_step))))
        ticks = np.round(ticks, decimals)
        return ticks

    def style_axis(ax, wrapped_categories: list[str], xlim: tuple[float, float]):
        ax.set_xlim(*xlim)
        xmin, xmax = xlim
        span = xmax - xmin
        if x_tick_step is not None:
            tick_step = float(x_tick_step)
        else:
            if span <= 4: tick_step = 0.5
            elif span <= 10: tick_step = 1.0
            else: tick_step = infer_pretty_tick_step(xmin, xmax, target_n=8)
        xticks = build_pretty_ticks(xmin, xmax, tick_step)
        ax.set_xticks(xticks)
        y_ticks = [i * row_gap for i in range(len(wrapped_categories))]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(wrapped_categories, fontsize=float(tick_fontsize))
        y_min = -0.5
        y_max = (max(0, len(wrapped_categories) - 1) * row_gap) + 0.5
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Средняя оценка сотрудника", fontsize=float(tick_fontsize))
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=float(tick_fontsize))
        if show_grid:
            ax.grid(axis="x", alpha=0.25)
            ax.grid(axis="y", alpha=0.15)
        else:
            ax.grid(False)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(PALETTE["additional"]["blue_mid"])
        ax.tick_params(axis="y", length=0, labelsize=float(tick_fontsize))

    def build_mean_handle(edge_col: str):
        label = "Среднее" if not show_mean_value_label else "Среднее (с подписью)"
        return Line2D([0],[0], marker="D", color="none", markerfacecolor=mean_point_color, markeredgecolor=edge_col, markersize=np.sqrt(mean_point_size), linestyle="None", label=label)

    def build_employee_handles(employee_names: list[str], employee_color_map: dict[str, str], edge_col: str, employee_display_map: dict[str, str]):
        return [Line2D([0],[0], marker="o", color="none", markerfacecolor=employee_color_map[emp], markeredgecolor=edge_col, markersize=np.sqrt(point_size), linestyle="None", label=employee_display_map.get(emp, emp)) for emp in employee_names]

    def collapse_scores(df: pd.DataFrame, group_cols_local: list[str]) -> pd.DataFrame:
        work = df.copy()
        work[score_col] = pd.to_numeric(work[score_col], errors='coerce')
        work = work[work[score_col].notna()].copy()
        if work.empty:
            return work
        if n_col in work.columns:
            work[n_col] = pd.to_numeric(work[n_col], errors='coerce').fillna(0.0)
        else:
            work[n_col] = 1.0
        dedup_cols = [c for c in [employee_col, resolved_category_col, facet_col, score_col, role_col] if c is not None and c in work.columns]
        if dedup_cols:
            work = work.drop_duplicates(subset=dedup_cols)
        if agg == 'weighted':
            work['_effective_weight'] = work[n_col].where(work[n_col] > 0, 1.0)
            work['_wx'] = work[score_col] * work['_effective_weight']
            out = work.groupby(group_cols_local, as_index=False, dropna=False).agg(wx_sum=('_wx','sum'), total_n=('_effective_weight','sum'), raw_n=(n_col,'sum'))
            out['employee_score'] = out['wx_sum'] / out['total_n'].replace(0, np.nan)
            out['total_n'] = out['raw_n']
            out = out.drop(columns=['wx_sum','raw_n'])
        else:
            out = work.groupby(group_cols_local, as_index=False, dropna=False).agg(employee_score=(score_col,'mean'), total_n=(n_col,'sum'))
        out = out[out['total_n'] >= int(min_total_n)].dropna(subset=['employee_score']).reset_index(drop=True)
        return out

    df = vis_df.copy()
    for fcol, fval in resolved_filters.items():
        if fcol in {resolved_category_col, facet_col}:
            continue
        if isinstance(fval, (list, tuple, set)):
            df = df[df[fcol].isin(list(fval))]
        else:
            df = df[df[fcol] == fval]
    if df.empty:
        raise ValueError('После фильтров не осталось данных.')

    if resolved_category_values is None:
        resolved_category_values = sorted(df[resolved_category_col].dropna().astype(str).unique().tolist())
    else:
        present = set(df[resolved_category_col].dropna().astype(str).unique().tolist())
        resolved_category_values = [str(v) for v in resolved_category_values if str(v) in present]
    if not resolved_category_values:
        raise ValueError(f"Нет значений для '{category_label}' после фильтров.")

    if facet_col is None:
        resolved_facet_values = [None]
    else:
        facet_base = df.copy()
        if facet_values is None:
            resolved_facet_values = sorted(facet_base[facet_col].dropna().astype(str).unique().tolist())
        else:
            present = set(facet_base[facet_col].dropna().astype(str).unique().tolist())
            resolved_facet_values = [str(v) for v in facet_values if str(v) in present]
        if not resolved_facet_values:
            raise ValueError(f"Нет значений для '{facet_label or facet_col}' после фильтров.")

    facet_emp_dfs: dict[str, dict[str, pd.DataFrame]] = {}
    all_scores = []
    all_employees = []
    stats_rows = []

    for facet_value in resolved_facet_values:
        facet_df = df.copy() if facet_col is None else df[df[facet_col].astype(str) == str(facet_value)].copy()
        if facet_df.empty:
            continue
        ridge_df = collapse_scores(facet_df, [employee_col, resolved_category_col])
        if ridge_df.empty:
            continue
        ridge_df['_category_key'] = ridge_df[resolved_category_col].astype(str)
        ridge_df['_employee_display'] = ridge_df[employee_col].map(abbreviate_employee_name)
        facet_key = display_value_for_col(facet_col, facet_value) if facet_value is not None else 'Без разбивки'
        facet_dict: dict[str, pd.DataFrame] = {}
        for cat in resolved_category_values:
            cat_key = str(cat)
            emp_df = ridge_df[ridge_df['_category_key'] == cat_key][[employee_col, '_employee_display', 'employee_score', 'total_n']].copy().reset_index(drop=True)
            facet_dict[cat_key] = emp_df
            if not emp_df.empty:
                all_scores.append(emp_df['employee_score'].to_numpy(dtype=float))
                all_employees.extend(emp_df[employee_col].astype(str).tolist())
                stats_rows.append({
                    'facet': facet_key,
                    'category': display_value_for_col(resolved_category_col, cat),
                    'n_employees': int(len(emp_df)),
                    'employee_score_mean': float(emp_df['employee_score'].mean()),
                    'employee_score_median': float(emp_df['employee_score'].median()),
                    'total_n_mean': float(emp_df['total_n'].mean()),
                })
        facet_emp_dfs[facet_key] = facet_dict

    if not all_scores:
        raise ValueError('Нет данных ни по одной категории (после фильтров/min_total_n).')

    xlim = compute_xlim(all_scores, x_pad_frac)
    unique_employees = list(dict.fromkeys(all_employees))
    too_many_employees = len(unique_employees) > employee_legend_limit

    if employee_colors is None:
        employee_colors = ["#FF474A", "#495867", "#30BAAD", "#CBBFAD", "#FA4D6E", "#FF762F", "#1ADC92", "#786EE0", "#8B5E3C", "#F2C94C"]
    employee_color_map = {emp: employee_colors[i % len(employee_colors)] for i, emp in enumerate(unique_employees)}
    employee_display_map = {emp: abbreviate_employee_name(emp) for emp in unique_employees}
    if mean_point_color is None:
        mean_point_color = PALETTE['main']['black']

    n_facets = len(facet_emp_dfs)
    fig_w, fig_h = figsize
    fig, axes = plt.subplots(nrows=n_facets, ncols=1, figsize=(fig_w, fig_h * n_facets), squeeze=False, sharex=True)
    axes_flat = axes.flatten().tolist()
    fig.patch.set_facecolor(PALETTE['background'])
    edge_col = PALETTE['background']
    mean_handle = build_mean_handle(edge_col)

    for ax, (facet_key, cat_map) in zip(axes_flat, facet_emp_dfs.items()):
        ax.set_facecolor(PALETTE['background'])
        for i, cat in enumerate(resolved_category_values):
            emp_df = cat_map.get(str(cat), pd.DataFrame())
            if emp_df is None or emp_df.empty:
                continue
            base_y = i * row_gap
            for _, row in emp_df.iterrows():
                emp_name = str(row[employee_col]); xv = float(row['employee_score']); c = employee_color_map[emp_name]
                ax.scatter([xv], [base_y], s=point_size, color=c, alpha=alpha, edgecolor=edge_col, linewidth=0.7, zorder=3)
            xs = emp_df['employee_score'].to_numpy(dtype=float)
            mean_val = float(np.mean(xs))
            ax.scatter([mean_val], [base_y], s=mean_point_size, color=mean_point_color, edgecolor=edge_col, linewidth=0.9, marker='D', zorder=5)
            if show_mean_value_label:
                ax.annotate(f"{mean_val:.2f}", xy=(mean_val, base_y), xytext=(6, 6), textcoords='offset points', ha='left', va='bottom', fontsize=float(value_fontsize), color=PALETTE['main']['black'], zorder=6)
        wrapped_categories = [wrap_label(display_value_for_col(resolved_category_col, c), width=comp_wrap_width, max_lines=comp_max_lines) for c in resolved_category_values]
        style_axis(ax, wrapped_categories, xlim)
        if facet_col is not None:
            ax.set_title(display_value_for_col(facet_col, facet_key), fontsize=float(title_fontsize))

    for ax in axes_flat[len(facet_emp_dfs):]:
        ax.set_visible(False)

    if show_employee_legend:
        legend_ax = axes_flat[0]
        if too_many_employees:
            legend_ax.legend(handles=[mean_handle], loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=float(legend_fontsize))
            legend_ax.text(1.02, 0.82, 'Сотрудников много.\nВоспользуйтесь другой визуализацией.', transform=legend_ax.transAxes, ha='left', va='top', fontsize=float(value_fontsize), color=PALETTE['main']['black'])
        else:
            employee_handles = build_employee_handles(unique_employees, employee_color_map, edge_col, employee_display_map)
            legend_handles = employee_handles + [mean_handle]
            legend = legend_ax.legend(handles=legend_handles, title='Сотрудники', loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=float(legend_fontsize))
            if legend is not None and legend.get_title() is not None:
                legend.get_title().set_fontsize(float(legend_fontsize))

    if title is True:
        if facet_col is None:
            axes_flat[0].set_title(f'Диаграмма рассеивания по {str(category_label).lower()}', fontsize=float(title_fontsize))
        # with facet, facet titles already shown
    elif isinstance(title, str) and title.strip():
        fig.suptitle(title, fontsize=float(title_fontsize))

    fig.tight_layout(rect=[0, 0, 0.72 if show_employee_legend else 0.96, 1])
    stats_df = pd.DataFrame(stats_rows)
    return fig, axes_flat if n_facets > 1 else axes_flat[0], stats_df, facet_emp_dfs


PALETTE_GROUPS = [
    "#FF474A",
    "#606E7D",
    "#30BAAD",
]


def _normalize_zone_labels(
    zone_labels: tuple[str, str, str] | list[str] | None,
) -> tuple[str, str, str]:
    if zone_labels is None:
        return ("Ниже low", "Между", "Выше high")
    if len(zone_labels) != 3:
        raise ValueError("zone_labels должен содержать ровно 3 подписи: (low, mid, high).")
    return tuple(str(x) for x in zone_labels)


def _normalize_palette_groups(
    palette_groups: tuple[str, str, str] | list[str] | None,
) -> tuple[str, str, str]:
    if palette_groups is None:
        return tuple(PALETTE_GROUPS)
    if len(palette_groups) != 3:
        raise ValueError("palette_groups должен содержать ровно 3 цвета: [low, mid, high].")
    return tuple(str(x) for x in palette_groups)


def _validate_percentile_span_value(span: float, name: str = "percentile_span") -> float:
    span = float(span)
    if not (0 <= span <= 50):
        raise ValueError(f"{name} должен быть в диапазоне [0, 50].")
    return span


def _resolve_percentile_cutoffs_for_columns(
    mat: pd.DataFrame,
    percentile_span: float | dict[str, float] = 25,
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    cols = [str(c) for c in mat.columns.tolist()]

    if isinstance(percentile_span, dict):
        span_map = {
            str(k): _validate_percentile_span_value(v, f"percentile_span['{k}']")
            for k, v in percentile_span.items()
        }
        default_span = 25.0
    else:
        default_span = _validate_percentile_span_value(percentile_span)
        span_map = {}

    for col in cols:
        span = span_map.get(col, default_span)
        low_p = 50.0 - span
        high_p = 50.0 + span
        s = pd.to_numeric(mat[col], errors="coerce")
        finite = s[np.isfinite(s)]
        if finite.shape[0] == 0:
            result[col] = (np.nan, np.nan)
        else:
            result[col] = (
                float(np.nanpercentile(finite, low_p)),
                float(np.nanpercentile(finite, high_p)),
            )
    return result


def _resolve_percentile_cutoffs_global(
    mat: pd.DataFrame,
    percentile_span: float = 25,
) -> dict[str, tuple[float, float]]:
    span = _validate_percentile_span_value(percentile_span)
    low_p = 50.0 - span
    high_p = 50.0 + span
    cols = [str(c) for c in mat.columns.tolist()]
    data = pd.to_numeric(pd.Series(mat.to_numpy(dtype=float).ravel()), errors="coerce")
    finite = data[np.isfinite(data)]
    if finite.shape[0] == 0:
        low, high = np.nan, np.nan
    else:
        low = float(np.nanpercentile(finite, low_p))
        high = float(np.nanpercentile(finite, high_p))
    return {col: (low, high) for col in cols}


def _resolve_explicit_cutoffs_for_columns(
    mat: pd.DataFrame,
    cutoffs: tuple[float, float] | dict[str, tuple[float, float]],
    percentile_span: float | dict[str, float] = 25,
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    cols = [str(c) for c in mat.columns.tolist()]

    if isinstance(cutoffs, tuple):
        if len(cutoffs) != 2:
            raise ValueError("Если cutoffs задан кортежем, он должен быть длины 2: (low, high).")
        low, high = float(cutoffs[0]), float(cutoffs[1])
        if low > high:
            raise ValueError("В cutoffs=(low, high) должно выполняться low <= high.")
        return {col: (low, high) for col in cols}

    if isinstance(cutoffs, dict):
        fallback = _resolve_percentile_cutoffs_for_columns(mat, percentile_span=percentile_span)
        for col in cols:
            if col in cutoffs:
                val = cutoffs[col]
                if not isinstance(val, tuple) or len(val) != 2:
                    raise ValueError(f"Для '{col}' в cutoffs должен быть кортеж (low, high).")
                low, high = float(val[0]), float(val[1])
                if low > high:
                    raise ValueError(f"Для '{col}' в cutoffs должно выполняться low <= high.")
                result[col] = (low, high)
            else:
                result[col] = fallback[col]
        return result

    raise ValueError("cutoffs должен быть tuple(low, high) или dict[column] = (low, high).")


def _resolve_cutoffs_per_column(
    mat: pd.DataFrame,
    cutoffs: None | tuple[float, float] | dict[str, tuple[float, float]],
    percentile_span: float | dict[str, float] = 25,
) -> dict[str, tuple[float, float]]:
    if cutoffs is None:
        return _resolve_percentile_cutoffs_for_columns(mat, percentile_span=percentile_span)
    return _resolve_explicit_cutoffs_for_columns(mat, cutoffs=cutoffs, percentile_span=percentile_span)


def _resolve_cutoffs_global(
    mat: pd.DataFrame,
    cutoffs: None | tuple[float, float],
    percentile_span: float = 25,
) -> dict[str, tuple[float, float]]:
    if cutoffs is None:
        return _resolve_percentile_cutoffs_global(mat, percentile_span=percentile_span)
    if not isinstance(cutoffs, tuple) or len(cutoffs) != 2:
        raise ValueError("При cutoff_mode='global' cutoffs должен быть None или tuple(low, high).")
    low, high = float(cutoffs[0]), float(cutoffs[1])
    if low > high:
        raise ValueError("В cutoffs=(low, high) должно выполняться low <= high.")
    cols = [str(c) for c in mat.columns.tolist()]
    return {col: (low, high) for col in cols}


def _classify_heatmap_values(
    mat: pd.DataFrame,
    cutoffs: None | tuple[float, float] | dict[str, tuple[float, float]],
    cutoff_mode: str = "per_column",
    percentile_span: float | dict[str, float] = 25,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    if cutoff_mode not in {"per_column", "global"}:
        raise ValueError("cutoff_mode должен быть 'per_column' или 'global'.")
    if cutoff_mode == "per_column":
        resolved_cutoffs = _resolve_cutoffs_per_column(mat, cutoffs=cutoffs, percentile_span=percentile_span)
    else:
        if isinstance(cutoffs, dict):
            raise ValueError("При cutoff_mode='global' cutoffs не может быть словарём.")
        if isinstance(percentile_span, dict):
            raise ValueError("При cutoff_mode='global' percentile_span не может быть словарём.")
        resolved_cutoffs = _resolve_cutoffs_global(mat, cutoffs=cutoffs, percentile_span=percentile_span)

    class_data = np.full(mat.shape, np.nan, dtype=float)
    for j, col in enumerate(mat.columns):
        col_str = str(col)
        low, high = resolved_cutoffs[col_str]
        values = pd.to_numeric(mat[col], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            continue
        class_col = np.full(values.shape, np.nan, dtype=float)
        class_col[finite_mask] = 1.0
        class_col[finite_mask & (values < low)] = 0.0
        class_col[finite_mask & (values > high)] = 2.0
        class_data[:, j] = class_col
    return class_data, resolved_cutoffs


def plot_group_heatmap(
    vis_df: pd.DataFrame,
    group_by: str | list[str] | None = None,
    employee_col: str = "Сотрудник",
    competency_col: str = "Компетенция",
    role_col: str = "роль",
    score_col: str = "score",
    heatmap_mode: str = "group",
    x_by: str = "competency",
    stat: str = "mean",
    role_filter: list[str] | None = None,
    competency_filter: list[str] | None = None,
    group_filters: dict[str, str | list[str]] | None = None,
    top_n: int | None = None,
    groups: list[str] | None = None,
    min_employees: int = 1,
    sort_categories: str = "order",
    sort_groups: str = "order",
    cutoff_mode: str = "per_column",
    cutoffs: None | tuple[float, float] | dict[str, tuple[float, float]] = None,
    percentile_span: float | dict[str, float] = 25,
    palette_groups: list[str] | tuple[str, str, str] | None = None,
    low_color: str | None = None,
    mid_color: str | None = None,
    high_color: str | None = None,
    zone_labels: tuple[str, str, str] | list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    cell_fill: float = np.sqrt(0.5),
    show_values: bool = False,
    value_fmt: str = "{:.2f}",
    value_fontsize: int = 8,
    grid: bool = True,
    grid_color: str = "#E0D9CE",
    grid_lw: float = 0.6,
    x_label_rotation: float = 90,
    x_label_ha: str = "center",
    x_label_wrap_width: int = 15,
    x_label_max_lines: int = 3,
    y_label_joiner: str = " • ",
    title_fontsize: int = 13,
    x_tick_label_fontsize: int = 10,
    y_tick_label_fontsize: int = 10,
    legend_fontsize: int = 9,
    value_text_fontsize: int | None = None,
    draw_level_separators: bool = True,
    separator_color: str = "#495867",
    separator_lw: float = 1.6,
    separator_alpha: float = 0.55,
    title: str | None = None,
    x_col: str | None = None,
    x_values: list[str] | None = None,
    y_cols: list[str] | None = None,
    y_values_map: dict[str, list[str]] | None = None,
    filters: dict[str, str | list[str]] | None = None,
    facet_col: str | None = None,
    facet_values: list[str] | None = None,
    facet_label: str | None = None,
    employee_name_format: str = "keep",
    highlight_shape: str = "rectangle",
    y_top_n_map: dict[str, int] | None = None,
    reference_enabled: bool = False,
    reference_values: list[str] | None = None,
    reference_label: str = "Референс",
    reference_delta: float | None = None,
    reference_shade: str = "#F3EEEB",
    reference_arrow_fontsize: float | None = None,
):
    from matplotlib.patches import Rectangle, Patch
    import itertools

    if not isinstance(vis_df, pd.DataFrame):
        raise TypeError("vis_df должен быть pandas.DataFrame.")
    if stat not in {"mean", "median"}:
        raise ValueError("stat должен быть 'mean' или 'median'.")
    if cutoff_mode not in {"per_column", "global"}:
        raise ValueError("cutoff_mode должен быть 'per_column' или 'global'.")
    if sort_categories not in {"order", "name", "none", "overall_stat_desc", "overall_stat_asc"}:
        raise ValueError("sort_categories: 'order'|'name'|'none'|'overall_stat_desc'|'overall_stat_asc'")
    if sort_groups not in {"order", "name", "none", "overall_stat_desc", "overall_stat_asc"}:
        raise ValueError("sort_groups: 'order'|'name'|'none'|'overall_stat_desc'|'overall_stat_asc'")
    if x_label_ha not in {"left", "center", "right"}:
        raise ValueError("x_label_ha должен быть одним из: 'left', 'center', 'right'.")
    if not (0 < cell_fill <= 1):
        raise ValueError("cell_fill должен быть в диапазоне (0, 1].")
    if employee_name_format not in {"keep", "first", "second", "third", "first5", "first3", "first_initials", "third_initials"}:
        raise ValueError("employee_name_format должен быть одним из: 'keep', 'first5', 'first3', 'first', 'second', 'third', 'first_initials', 'third_initials'.")
    if highlight_shape not in {"rectangle", "circle"}:
        raise ValueError("highlight_shape должен быть 'rectangle' или 'circle'.")
    y_top_n_map = dict(y_top_n_map or {})
    if reference_delta is not None and float(reference_delta) < 0:
        raise ValueError("reference_delta должен быть >= 0.")
    if reference_arrow_fontsize is not None and float(reference_arrow_fontsize) <= 0:
        raise ValueError("reference_arrow_fontsize должен быть > 0.")
    if cutoff_mode == "global" and isinstance(percentile_span, dict):
        raise ValueError("При cutoff_mode='global' percentile_span не может быть словарём.")
    if isinstance(percentile_span, dict):
        for k, v in percentile_span.items():
            _validate_percentile_span_value(v, f"percentile_span['{k}']")
    else:
        _validate_percentile_span_value(percentile_span)

    colors = list(_normalize_palette_groups(palette_groups))
    if low_color is not None:
        colors[0] = str(low_color)
    if mid_color is not None:
        colors[1] = str(mid_color)
    if high_color is not None:
        colors[2] = str(high_color)
    zone_low_label, zone_mid_label, zone_high_label = _normalize_zone_labels(zone_labels)
    if value_text_fontsize is None:
        value_text_fontsize = value_fontsize

    df = vis_df.copy()

    # backward compatibility
    if x_col is None:
        if heatmap_mode == "group":
            y_cols = [group_by] if isinstance(group_by, str) else list(group_by or [])
            x_col = competency_col if x_by == "competency" else role_col
        else:
            y_cols = [competency_col]
            x_col = role_col
    else:
        y_cols = list(y_cols or [])
    if not y_cols:
        raise ValueError("Нужно передать хотя бы одну характеристику для оси Y.")
    if facet_col is not None and facet_label is None:
        facet_label = str(facet_col)

    resolved_filters = dict(filters or {})
    if group_filters:
        resolved_filters.update(group_filters)
    if role_filter is not None and role_col not in [x_col, facet_col] + list(y_cols):
        resolved_filters[role_col] = role_filter
    if competency_filter is not None and competency_col not in [x_col, facet_col] + list(y_cols):
        resolved_filters[competency_col] = competency_filter

    needed = [employee_col, x_col, score_col] + list(y_cols) + ([facet_col] if facet_col is not None else []) + list(resolved_filters.keys())
    missing = [c for c in list(dict.fromkeys(needed)) if c not in df.columns]
    if missing:
        raise ValueError(f"В vis_df отсутствуют колонки: {missing}")

    for fcol, fval in resolved_filters.items():
        if fcol in [x_col, facet_col] + list(y_cols):
            continue
        if isinstance(fval, (list, tuple, set)):
            df = df[df[fcol].isin(list(fval))]
        else:
            df = df[df[fcol] == fval]
    if df.empty:
        raise ValueError("После фильтров не осталось данных.")

    work = df.dropna(subset=[employee_col, x_col, score_col] + list(y_cols) + ([facet_col] if facet_col is not None else [])).copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work = work.dropna(subset=[score_col]).copy()
    if work.empty:
        raise ValueError("После очистки score_col не осталось числовых данных.")
    for c in [employee_col, x_col] + list(y_cols) + ([facet_col] if facet_col is not None else []):
        work[c] = work[c].astype(str)

    if x_values is None:
        x_values = sorted(work[x_col].dropna().unique().tolist(), key=lambda x: str(x))
    x_values = [str(x) for x in x_values if str(x) in set(work[x_col].dropna().tolist())]
    if top_n is not None:
        x_values = x_values[:int(top_n)]
    if not x_values:
        raise ValueError(f"Нет значений для '{x_col}' после фильтров.")
    work = work[work[x_col].isin(x_values)].copy()

    y_values_map = dict(y_values_map or {})
    if groups is not None and len(y_cols) == 1 and y_cols[0] not in y_values_map:
        y_values_map[y_cols[0]] = [str(g) for g in groups]
    for yc, vals in y_values_map.items():
        if vals is not None and yc in work.columns:
            allowed = [str(v) for v in vals]
            work = work[work[yc].isin(allowed)].copy()
    if work.empty:
        raise ValueError("После фильтров осей не осталось данных.")

    if facet_col is None:
        facets = [None]
    else:
        if facet_values is None:
            facet_values = sorted(work[facet_col].dropna().unique().tolist(), key=lambda x: str(x))
        facets = [str(x) for x in facet_values if str(x) in set(work[facet_col].dropna().tolist())]
        if not facets:
            raise ValueError(f"Нет значений для '{facet_label or facet_col}' после фильтров.")

    def _unique_keep_order(columns: list[str]) -> list[str]:
        seen = set()
        out = []
        for col in columns:
            if col not in seen:
                seen.add(col)
                out.append(col)
        return out

    stat_func = "mean" if stat == "mean" else "median"
    summary_frames = []
    mats = {}
    resolved_cutoffs_map = {}

    for facet_value in facets:
        facet_work = work if facet_col is None else work[work[facet_col] == str(facet_value)].copy()
        if facet_work.empty:
            continue
        emp_scores = (
            facet_work.groupby(_unique_keep_order([employee_col] + list(y_cols) + [x_col]), as_index=False)
            .agg(employee_score=(score_col, stat_func))
        )
        summary_df = (
            emp_scores.groupby(_unique_keep_order(list(y_cols) + [x_col]), as_index=False)
            .agg(n_employees=("employee_score", "count"), stat_value=("employee_score", stat_func))
        )
        summary_df.loc[summary_df["n_employees"] < int(min_employees), "stat_value"] = np.nan
        mat = summary_df.pivot_table(index=y_cols, columns=x_col, values="stat_value", aggfunc="mean")
        mat = mat.reindex(columns=[c for c in x_values if c in mat.columns])
        if sort_categories == "name":
            mat = mat.reindex(sorted(mat.columns, key=lambda x: str(x)), axis=1)
        elif sort_categories in ("overall_stat_desc", "overall_stat_asc"):
            overall_c = mat.mean(axis=0, skipna=True)
            mat = mat.loc[:, overall_c.sort_values(ascending=(sort_categories == "overall_stat_asc")).index]
        if sort_groups == "order":
            levels = []
            for yc in y_cols:
                vals = y_values_map.get(yc)
                if vals is None:
                    vals = sorted(facet_work[yc].dropna().unique().tolist(), key=lambda x: str(x))
                else:
                    vals = [str(v) for v in vals if str(v) in set(facet_work[yc].dropna().tolist())]
                top_n_y = int(y_top_n_map.get(yc, 0) or 0)
                if top_n_y > 0:
                    vals = vals[:top_n_y]
                levels.append(vals)
            if isinstance(mat.index, pd.MultiIndex):
                desired = [idx for idx in itertools.product(*levels) if idx in mat.index]
                mat = mat.reindex(desired)
            else:
                desired = [idx for idx in levels[0] if idx in mat.index]
                mat = mat.reindex(desired)
        elif sort_groups == "name":
            if isinstance(mat.index, pd.MultiIndex):
                mat = mat.sort_index(level=list(range(mat.index.nlevels)))
            else:
                mat = mat.sort_index()
        elif sort_groups in ("overall_stat_desc", "overall_stat_asc"):
            overall_r = mat.mean(axis=1, skipna=True)
            mat = mat.loc[overall_r.sort_values(ascending=(sort_groups == "overall_stat_asc")).index]

        if y_top_n_map:
            if isinstance(mat.index, pd.MultiIndex):
                keep_mask = np.ones(len(mat.index), dtype=bool)
                for level_idx, yc in enumerate(y_cols):
                    top_n_y = int(y_top_n_map.get(yc, 0) or 0)
                    if top_n_y > 0:
                        allowed_vals = pd.Index(mat.index.get_level_values(level_idx)).drop_duplicates().tolist()[:top_n_y]
                        keep_mask &= pd.Index(mat.index.get_level_values(level_idx)).isin(allowed_vals)
                mat = mat.loc[keep_mask]
            else:
                top_n_y = int(y_top_n_map.get(y_cols[0], 0) or 0)
                if top_n_y > 0:
                    mat = mat.iloc[:top_n_y]

        if mat.shape[0] == 0 or mat.shape[1] == 0:
            continue

        reference_col_name = None
        if reference_enabled:
            ref_candidates = [str(v) for v in (reference_values or []) if str(v) in mat.columns]
            if not ref_candidates:
                ref_candidates = [str(v) for v in x_values if str(v) in mat.columns]
            if ref_candidates:
                reference_series = mat[ref_candidates].mean(axis=1, skipna=True)
                reference_col_name = str(reference_label or "Референс")
                if reference_col_name in mat.columns:
                    reference_col_name = f"{reference_col_name} (ref)"
                mat[reference_col_name] = reference_series
                ordered_cols = [reference_col_name] + [c for c in mat.columns.tolist() if c != reference_col_name]
                mat = mat.loc[:, ordered_cols]

        class_data, resolved_cutoffs = _classify_heatmap_values(mat, cutoffs=cutoffs, cutoff_mode=cutoff_mode, percentile_span=percentile_span)
        if reference_enabled and reference_col_name is not None and reference_col_name in mat.columns and cutoff_mode == "global":
            ref_j = list(mat.columns).index(reference_col_name)
            ref_vals = pd.to_numeric(mat[reference_col_name], errors="coerce").to_numpy(dtype=float)
            ref_mask = np.isfinite(ref_vals)
            class_data[:, ref_j] = np.where(ref_mask, 1.0, np.nan)
            if reference_col_name in resolved_cutoffs:
                resolved_cutoffs[reference_col_name] = (np.nan, np.nan)
        facet_key = str(facet_value) if facet_value is not None else "__single__"
        mats[facet_key] = (mat, class_data)
        resolved_cutoffs_map[facet_key] = resolved_cutoffs
        summary_df["facet_value"] = "" if facet_value is None else str(facet_value)
        summary_frames.append(summary_df)

    if not mats:
        raise ValueError("После фильтров/порогов не осталось данных для heatmap.")

    facet_keys = list(mats.keys())
    max_rows = max(mat.shape[0] for mat, _ in mats.values())
    max_cols = max(mat.shape[1] for mat, _ in mats.values())
    if figsize is None:
        figsize = (max(8.5, 0.55 * max_cols + 4.0), max(5.5, 0.35 * max_rows + 2.8) * len(facet_keys))
    fig, axes = plt.subplots(nrows=len(facet_keys), ncols=1, figsize=figsize)
    if len(facet_keys) == 1:
        axes = [axes]

    half = cell_fill / 2.0
    for ax, facet_key in zip(axes, facet_keys):
        mat, class_data = mats[facet_key]
        reference_col_name = str(reference_label or "Референс") if reference_enabled else None
        if reference_col_name not in mat.columns:
            alt_candidates = [c for c in mat.columns if str(c).startswith(str(reference_label or "Референс"))]
            reference_col_name = alt_candidates[0] if alt_candidates else None
        ax.set_facecolor("white")
        if reference_col_name is not None and reference_col_name in mat.columns:
            ref_j = list(mat.columns).index(reference_col_name)
            ax.add_patch(Rectangle((ref_j - 0.5, -0.5), 1.0, mat.shape[0], facecolor=reference_shade, edgecolor="none", zorder=0))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                cls = class_data[i, j]
                if np.isfinite(cls):
                    if highlight_shape == "circle":
                        patch = Circle((j, i), radius=cell_fill / 2.0, facecolor=colors[int(cls)], edgecolor="none", zorder=2)
                    else:
                        patch = Rectangle((j - half, i - half), cell_fill, cell_fill, facecolor=colors[int(cls)], edgecolor="none", zorder=2)
                    ax.add_patch(patch)
        ax.set_xlim(-0.5, mat.shape[1] - 0.5)
        ax.set_ylim(mat.shape[0] - 0.5, -0.5)
        ax.set_aspect("auto")
        xlabels = [_wrap_and_truncate_label(_abbreviate_employee_name(c, employee_name_format) if x_col == employee_col else str(c), width=x_label_wrap_width, max_lines=x_label_max_lines) for c in mat.columns.tolist()]
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(xlabels, rotation=x_label_rotation, ha=x_label_ha, fontsize=x_tick_label_fontsize)
        ax.xaxis.tick_top()
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        if isinstance(mat.index, pd.MultiIndex):
            ylabels = [y_label_joiner.join([_abbreviate_employee_name(v, employee_name_format) if y_cols[i] == employee_col else str(v) for i, v in enumerate(t)]) for t in mat.index.tolist()]
        else:
            ylabels = [_abbreviate_employee_name(v, employee_name_format) if y_cols[0] == employee_col else str(v) for v in mat.index.tolist()]
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(ylabels, fontsize=y_tick_label_fontsize)
        if grid:
            ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
            ax.grid(which="minor", color=grid_color, linewidth=grid_lw)
            ax.tick_params(which="minor", bottom=False, left=False, top=False)
        if draw_level_separators and isinstance(mat.index, pd.MultiIndex) and mat.shape[0] > 1:
            vals = mat.index.get_level_values(0).to_numpy()
            for i in range(1, mat.shape[0]):
                if vals[i] != vals[i - 1]:
                    ax.hlines(i - 0.5, -0.5, mat.shape[1] - 0.5, colors=separator_color, linewidth=separator_lw, alpha=separator_alpha, zorder=4)
        if show_values:
            real_data = mat.to_numpy(dtype=float)
            ref_vals = pd.to_numeric(mat[reference_col_name], errors="coerce").to_numpy(dtype=float) if reference_col_name is not None and reference_col_name in mat.columns else None
            delta_threshold = float(reference_delta) if reference_delta is not None else None
            arrow_fontsize = float(reference_arrow_fontsize) if reference_arrow_fontsize is not None else max(9.0, float(value_text_fontsize) + 2.0)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = real_data[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, value_fmt.format(v), ha="center", va="center", fontsize=value_text_fontsize, zorder=5)
                        if ref_vals is not None and delta_threshold is not None and reference_col_name is not None and mat.columns[j] != reference_col_name and np.isfinite(ref_vals[i]):
                            delta = float(v - ref_vals[i])
                            if delta > delta_threshold:
                                ax.text(j + 0.27, i, "▲", ha="center", va="center", fontsize=arrow_fontsize, color="#121111", zorder=6)
                            elif delta < -delta_threshold:
                                ax.text(j + 0.27, i, "▼", ha="center", va="center", fontsize=arrow_fontsize, color="#121111", zorder=6)
        if facet_col is not None:
            ax.set_title(_abbreviate_employee_name(facet_key, employee_name_format) if facet_col == employee_col else str(facet_key), pad=18, fontsize=title_fontsize)

    legend_handles = [
        Patch(facecolor=colors[0], edgecolor="none", label=zone_low_label),
        Patch(facecolor=colors[1], edgecolor="none", label=zone_mid_label),
        Patch(facecolor=colors[2], edgecolor="none", label=zone_high_label),
    ]
    if reference_enabled:
        legend_handles.extend([
            Line2D([0], [0], marker="^", color="#121111", linestyle="None", markersize=max(8, (float(reference_arrow_fontsize) if reference_arrow_fontsize is not None else max(9.0, float(value_text_fontsize) + 2.0)) * 0.9), label="Выше референса"),
            Line2D([0], [0], marker="v", color="#121111", linestyle="None", markersize=max(8, (float(reference_arrow_fontsize) if reference_arrow_fontsize is not None else max(9.0, float(value_text_fontsize) + 2.0)) * 0.9), label="Ниже референса"),
        ])
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(legend_handles), 5), frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=legend_fontsize)
    if title:
        if len(facet_keys) == 1:
            axes[0].set_title(title, pad=18, fontsize=title_fontsize)
            fig.tight_layout(rect=(0, 0.06, 1, 1))
        else:
            fig.suptitle(title, fontsize=title_fontsize)
            fig.tight_layout(rect=(0, 0.06, 1, 0.98))
    else:
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    summary_df_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    if len(facet_keys) == 1:
        key = facet_keys[0]
        return fig, axes[0], summary_df_all, mats[key][0], resolved_cutoffs_map[key]
    return fig, axes, summary_df_all, {k: mats[k][0] for k in facet_keys}, resolved_cutoffs_map



def plot_9box(
    df: pd.DataFrame,
    x_col: str,
    x_values: list[str],
    y_col: str,
    y_values: list[str],
    vmin: float,
    vmax: float,
    employee_col: str = "Сотрудник",
    score_col: str = "score",
    n_col: str = "n",
    filters: dict[str, str | list[str]] | None = None,
    facet_col: str | None = None,
    facet_values: list[str] | None = None,
    cutpoint_method: Literal["percent", "fixed"] = "percent",
    cutpoint_scope: Literal["all_data", "filtered_data"] = "filtered_data",
    percent_span: float = 25,
    cutpoints: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
    agg: str = "weighted",
    jitter: bool = True,
    jitter_scale: float = 0.10,
    random_state: int = 42,
    show_cell_counts: bool = True,
    label_cells: list[str] | None = None,
    label_fontsize: int = 8,
    label_dx: int = 6,
    label_dy: int = 6,
    point_size: int = 65,
    point_alpha: float = 0.85,
    point_color: str = "#121111",
    zone_colors: dict[str, str] | None = None,
    figsize: tuple[float, float] = (11, 9.5),
    title: str | None = None,
    axis_label_fontsize: int = 11,
    axis_tick_fontsize: int = 10,
    legend_fontsize: int = 9,
    legend_title_fontsize: int = 10,
    title_fontsize: int = 13,
    axis_tick_mode: Literal["cutpoints", "percentiles", "none"] = "cutpoints",
    axis_tick_decimals: int = 2,
    employee_name_format: str = "keep",
    x_title: str | None = None,
    y_title: str | None = None,
    highlight: dict[str, str | list[str]] | None = None,
    highlight_color: str = "#2563EB",
    zone_level_labels: dict[str, str] | None = None,
    zone_cell_colors: dict[str, str] | None = None,
    zone_cell_labels: dict[str, str] | None = None,
    legend_position: Literal['top', 'bottom', 'none'] = 'top',
    group_by: str | None = None,
    point_mode: Literal['employee', 'competency', 'role'] = 'employee',
    point_col: str | None = None,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df должен быть pandas.DataFrame.")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Колонки для осей X/Y не найдены в df.")
    if score_col not in df.columns:
        raise ValueError("В df должна быть колонка score.")
    if agg not in {"weighted", "mean"}:
        raise ValueError("agg должен быть 'weighted' или 'mean'.")
    if employee_name_format not in {"keep", "first", "second", "third", "first5", "first3", "first_initials", "third_initials"}:
        raise ValueError("employee_name_format должен быть одним из: 'keep', 'first5', 'first3', 'first', 'second', 'third', 'first_initials', 'third_initials'.")
    if not x_values or not y_values:
        raise ValueError("Для осей X и Y нужно выбрать хотя бы по одному значению.")
    if float(vmax) <= float(vmin):
        raise ValueError("vmax должен быть больше vmin.")
    if legend_position not in {'top', 'bottom', 'none'}:
        raise ValueError("legend_position должен быть 'top', 'bottom' или 'none'.")
    if point_mode not in {'employee', 'competency', 'role'}:
        raise ValueError("point_mode должен быть 'employee', 'competency' или 'role'.")

    resolved_point_col = point_col or {
        'employee': employee_col,
        'competency': 'Компетенция',
        'role': 'роль',
    }[point_mode]
    if resolved_point_col not in df.columns:
        raise ValueError(f"Колонка точки '{resolved_point_col}' не найдена в df.")
    if facet_col is not None and facet_col not in df.columns:
        raise ValueError(f"facet_col='{facet_col}' не найдена в df.")

    def _abbr(value: object) -> str:
        if pd.isna(value):
            return ""
        return _abbreviate_employee_name(value, employee_name_format)

    def _uniq(cols):
        out = []
        seen = set()
        for c in cols:
            if c is None or c in seen:
                continue
            seen.add(c)
            out.append(c)
        return out

    def _filter_df_local(src: pd.DataFrame, skip_axis_facet: bool = True) -> pd.DataFrame:
        out = src.copy()
        for col, val in (filters or {}).items():
            if col not in out.columns:
                continue
            if skip_axis_facet and col in {x_col, y_col, facet_col}:
                continue
            if isinstance(val, (list, tuple, set)):
                out = out[out[col].astype(str).isin([str(v) for v in val])]
            else:
                out = out[out[col].astype(str) == str(val)]
        return out

    def _build_axis_scores(src: pd.DataFrame, dim_col: str, values: list[str], out_name: str, facet_for_grouping: str | None = None) -> pd.DataFrame:
        work = src.copy()
        work[score_col] = pd.to_numeric(work[score_col], errors='coerce')
        work = work[work[score_col].notna()].copy()
        work[dim_col] = work[dim_col].astype(str)
        work[resolved_point_col] = work[resolved_point_col].astype(str)
        if facet_for_grouping is not None:
            work[facet_for_grouping] = work[facet_for_grouping].astype(str)
        work = work[work[dim_col].isin([str(v) for v in values])].copy()
        if work.empty:
            cols = _uniq([resolved_point_col] + ([facet_for_grouping] if facet_for_grouping is not None else [])) + [out_name]
            return pd.DataFrame(columns=cols)
        grp_cols = _uniq([resolved_point_col, dim_col] + ([facet_for_grouping] if facet_for_grouping is not None else []))
        if n_col in work.columns and agg == 'weighted':
            work[n_col] = pd.to_numeric(work[n_col], errors='coerce').fillna(0)
            work['_w'] = work[n_col].where(work[n_col] > 0, 1.0)
            work['_ws'] = work[score_col] * work['_w']
            dim_scores = (
                work.groupby(grp_cols, dropna=False)
                .agg(_score_sum=('_ws', 'sum'), _weight_sum=('_w', 'sum'))
                .reset_index()
            )
            dim_scores[out_name] = np.where(dim_scores['_weight_sum'] > 0, dim_scores['_score_sum'] / dim_scores['_weight_sum'], np.nan)
            dim_scores = dim_scores.drop(columns=['_score_sum', '_weight_sum'])
        else:
            dim_scores = work.groupby(grp_cols, dropna=False)[score_col].mean().reset_index().rename(columns={score_col: out_name})
        point_grp = _uniq([resolved_point_col] + ([facet_for_grouping] if facet_for_grouping is not None else []))
        return dim_scores.groupby(point_grp, dropna=False)[out_name].mean().reset_index()

    d_plot = _filter_df_local(df, skip_axis_facet=True)
    if d_plot.empty:
        raise ValueError('После фильтров не осталось данных.')

    effective_facet_col = facet_col
    x_scores = _build_axis_scores(d_plot, x_col, x_values, 'x_raw', facet_for_grouping=effective_facet_col)
    y_scores = _build_axis_scores(d_plot, y_col, y_values, 'y_raw', facet_for_grouping=effective_facet_col)
    merge_cols = _uniq([resolved_point_col] + ([effective_facet_col] if effective_facet_col is not None else []))
    points_df = x_scores.merge(y_scores, on=merge_cols, how='inner')
    points_df = points_df.dropna(subset=['x_raw', 'y_raw']).copy()

    if effective_facet_col is not None:
        points_df[effective_facet_col] = points_df[effective_facet_col].astype(str)
        if facet_values is None:
            facets = sorted(points_df[effective_facet_col].dropna().astype(str).unique().tolist(), key=lambda x: str(x))
        else:
            present = set(points_df[effective_facet_col].dropna().astype(str).unique().tolist())
            facets = [str(v) for v in facet_values if str(v) in present]
        points_df = points_df[points_df[effective_facet_col].astype(str).isin(facets)].copy()
        points_df['facet'] = points_df[effective_facet_col].astype(str)
        facet_title_col = effective_facet_col
        unique_facets = [f for f in facets if f in set(points_df['facet'].astype(str))]
    else:
        points_df['facet'] = ''
        facet_title_col = None
        unique_facets = [''] if not points_df.empty else []

    if points_df.empty or not unique_facets:
        raise ValueError('Нет точек с данными одновременно по осям X и Y после фильтров.')

    if cutpoint_scope == 'all_data':
        d_cut = df.copy()
        x_cut_scores = _build_axis_scores(d_cut, x_col, x_values, 'x_raw', facet_for_grouping=None)
        y_cut_scores = _build_axis_scores(d_cut, y_col, y_values, 'y_raw', facet_for_grouping=None)
        cut_df = x_cut_scores.merge(y_cut_scores, on=[resolved_point_col], how='inner')
    else:
        d_cut = d_plot.copy()
        x_cut_scores = _build_axis_scores(d_cut, x_col, x_values, 'x_raw', facet_for_grouping=effective_facet_col)
        y_cut_scores = _build_axis_scores(d_cut, y_col, y_values, 'y_raw', facet_for_grouping=effective_facet_col)
        cut_merge_cols = _uniq([resolved_point_col] + ([effective_facet_col] if effective_facet_col is not None else []))
        cut_df = x_cut_scores.merge(y_cut_scores, on=cut_merge_cols, how='inner')
    if cut_df.empty:
        cut_df = points_df[[resolved_point_col, 'x_raw', 'y_raw']].copy()

    if cutpoint_method == 'percent':
        if not (0 < float(percent_span) < 50):
            raise ValueError('percent_span должен быть > 0 и < 50.')
        low_q = (50.0 - float(percent_span)) / 100.0
        high_q = (50.0 + float(percent_span)) / 100.0
        x_arr = cut_df['x_raw'].to_numpy(dtype=float)
        y_arr = cut_df['y_raw'].to_numpy(dtype=float)
        x_arr = x_arr[np.isfinite(x_arr)]
        y_arr = y_arr[np.isfinite(y_arr)]
        def _default_cut(arr):
            if len(arr) == 0:
                span = float(vmax) - float(vmin)
                return (float(vmin) + span/3.0, float(vmin) + 2*span/3.0)
            return float(np.quantile(arr, low_q)), float(np.quantile(arr, high_q))
        t1x, t2x = _default_cut(x_arr)
        t1y, t2y = _default_cut(y_arr)
    else:
        if cutpoints is None:
            raise ValueError("Для cutpoint_method='fixed' нужно задать cutpoints.")
        if isinstance(cutpoints, dict):
            if 'x' not in cutpoints or 'y' not in cutpoints:
                raise ValueError("cutpoints dict должен быть {'x': (t1,t2), 'y': (t1,t2)}")
            t1x, t2x = cutpoints['x']
            t1y, t2y = cutpoints['y']
        else:
            t1x, t2x = cutpoints
            t1y, t2y = cutpoints
        t1x, t2x, t1y, t2y = map(float, (t1x, t2x, t1y, t2y))

    if not (float(vmin) <= float(t1x) <= float(t2x) <= float(vmax)):
        raise ValueError('X cutpoints должны удовлетворять vmin <= t1 <= t2 <= vmax.')
    if not (float(vmin) <= float(t1y) <= float(t2y) <= float(vmax)):
        raise ValueError('Y cutpoints должны удовлетворять vmin <= t1 <= t2 <= vmax.')

    def _map_to_0_3(v: float, t1: float, t2: float) -> float:
        if not np.isfinite(v):
            return np.nan
        v = float(np.clip(v, vmin, vmax))
        eps = 1e-9
        if v <= t1:
            return (v - vmin) / max(t1 - vmin, eps)
        if v <= t2:
            return 1.0 + (v - t1) / max(t2 - t1, eps)
        return 2.0 + (v - t2) / max(vmax - t2, eps)

    points_df['x_plot'] = points_df['x_raw'].map(lambda v: _map_to_0_3(v, t1x, t2x))
    points_df['y_plot'] = points_df['y_raw'].map(lambda v: _map_to_0_3(v, t1y, t2y))
    points_df['x_zone'] = np.where(points_df['x_plot'] < 1, 0, np.where(points_df['x_plot'] < 2, 1, 2))
    points_df['y_zone'] = np.where(points_df['y_plot'] < 1, 0, np.where(points_df['y_plot'] < 2, 1, 2))
    points_df['cell_key'] = points_df['x_zone'].astype(str) + '_' + points_df['y_zone'].astype(str)
    points_df['_point_label'] = points_df[resolved_point_col].map(_abbr)
    points_df['_x'] = points_df['x_plot']
    points_df['_y'] = points_df['y_plot']

    zone_colors = {'low': '#FF474A', 'mid': '#E0D9CE', 'high': '#30BAAD', **(zone_colors or {})}
    zone_level_labels = {'low': 'Низкий', 'mid': 'Средний', 'high': 'Высокий', **(zone_level_labels or {})}
    default_zone_cell_colors = {
        '0_2': zone_colors['low'], '1_2': zone_colors['mid'], '2_2': zone_colors['high'],
        '0_1': zone_colors['low'], '1_1': zone_colors['mid'], '2_1': zone_colors['high'],
        '0_0': zone_colors['low'], '1_0': zone_colors['mid'], '2_0': zone_colors['high'],
    }
    zone_cell_colors = {**default_zone_cell_colors, **(zone_cell_colors or {})}
    default_zone_cell_labels = {
        '0_2': f"{zone_level_labels['low']}-{zone_level_labels['high']}",
        '1_2': f"{zone_level_labels['mid']}-{zone_level_labels['high']}",
        '2_2': f"{zone_level_labels['high']}-{zone_level_labels['high']}",
        '0_1': f"{zone_level_labels['low']}-{zone_level_labels['mid']}",
        '1_1': f"{zone_level_labels['mid']}-{zone_level_labels['mid']}",
        '2_1': f"{zone_level_labels['high']}-{zone_level_labels['mid']}",
        '0_0': f"{zone_level_labels['low']}-{zone_level_labels['low']}",
        '1_0': f"{zone_level_labels['mid']}-{zone_level_labels['low']}",
        '2_0': f"{zone_level_labels['high']}-{zone_level_labels['low']}",
    }
    zone_cell_labels = {**default_zone_cell_labels, **(zone_cell_labels or {})}

    facet_count = max(len(unique_facets), 1)
    fig, axes = plt.subplots(facet_count, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten().tolist()

    for ax, facet_value in zip(axes, unique_facets):
        sub = points_df if facet_title_col is None else points_df[points_df['facet'] == str(facet_value)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        cell_counts = sub['cell_key'].value_counts().to_dict()
        for xi in range(3):
            for yi in range(3):
                color = zone_cell_colors.get(f'{xi}_{yi}', zone_colors['mid'])
                ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color, alpha=0.5, edgecolor='none', zorder=0))
        for g in [1, 2]:
            ax.axvline(g, color='#121111', linewidth=0.8, alpha=0.35, zorder=1)
            ax.axhline(g, color='#121111', linewidth=0.8, alpha=0.35, zorder=1)
        ax.scatter(sub['_x'], sub['_y'], s=point_size, alpha=point_alpha, color=point_color, marker='o', zorder=2)
        if show_cell_counts:
            for xi in range(3):
                for yi in range(3):
                    n = cell_counts.get(f'{xi}_{yi}', 0)
                    ax.text(xi + 0.95, yi + 0.95, str(n), ha='right', va='top', fontsize=max(8, axis_tick_fontsize), fontweight='bold', color='#121111', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1.5), zorder=4)
        if label_cells:
            labels_df = sub[sub['cell_key'].isin([str(x) for x in label_cells])].copy()
            for _, row in labels_df.iterrows():
                txt = ax.annotate(
                    str(row['_point_label']),
                    xy=(row['_x'], row['_y']),
                    xytext=(label_dx, label_dy),
                    textcoords='offset points',
                    fontsize=label_fontsize,
                    color='#121111',
                    ha='left' if label_dx >= 0 else 'right',
                    va='bottom' if label_dy >= 0 else 'top',
                    zorder=10,
                    annotation_clip=False,
                )
                txt.set_path_effects([
                    path_effects.Stroke(linewidth=2.0, foreground='white'),
                    path_effects.Normal(),
                ])
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        if axis_tick_mode == 'none':
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xticks([0, 1, 2, 3])
            ax.set_yticks([0, 1, 2, 3])
            if axis_tick_mode == 'cutpoints' or cutpoint_method == 'fixed':
                fmt = f'{{:.{axis_tick_decimals}f}}'
                ax.set_xticklabels([f'{vmin:g}', fmt.format(t1x), fmt.format(t2x), f'{vmax:g}'])
                ax.set_yticklabels([f'{vmin:g}', fmt.format(t1y), fmt.format(t2y), f'{vmax:g}'])
            else:
                p1 = 50 - percent_span
                p2 = 50 + percent_span
                ax.set_xticklabels(['0', f'{p1:g}', f'{p2:g}', '100'])
                ax.set_yticklabels(['0', f'{p1:g}', f'{p2:g}', '100'])
            ax.tick_params(axis='both', labelsize=axis_tick_fontsize)
        ax.set_xlabel(x_title or (f'{x_col}: ' + ', '.join([str(v) for v in x_values])), fontsize=axis_label_fontsize)
        ax.set_ylabel(y_title or (f'{y_col}: ' + ', '.join([str(v) for v in y_values])), fontsize=axis_label_fontsize)
        if facet_title_col is not None:
            ax.set_title(str(facet_value), fontsize=title_fontsize)
        elif title:
            ax.set_title(title, fontsize=title_fontsize)
        ax.set_facecolor('white')

    zone_handles = []
    ordered_cells = [(0, 2), (1, 2), (2, 2), (0, 1), (1, 1), (2, 1), (0, 0), (1, 0), (2, 0)]
    for cell in ordered_cells:
        cell_key = f'{cell[0]}_{cell[1]}'
        zone_handles.append(Patch(facecolor=zone_cell_colors.get(cell_key, zone_colors['mid']), edgecolor='none', alpha=0.5, label=zone_cell_labels.get(cell_key, cell_key)))
    legend_handles = list(zone_handles)
    if legend_position != 'none':
        fig_w_in = max(float(fig.get_size_inches()[0]), 1.0)
        fig_h_in = max(float(fig.get_size_inches()[1]), 1.0)
        ncol = min(max(len(legend_handles), 1), 6)
        legend_loc = 'upper center' if legend_position == 'top' else 'lower center'
        if legend_position == 'top':
            legend_y = 1.0 - (0.18 / fig_h_in)
            top_pad_in = 0.70
            bottom_pad_in = 0.45
            fig.legend(handles=legend_handles, frameon=False, loc=legend_loc, bbox_to_anchor=(0.5, legend_y), bbox_transform=fig.transFigure, ncol=ncol, fontsize=legend_fontsize, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
            fig.tight_layout(rect=[0.03, bottom_pad_in / fig_h_in, 0.97, 1.0 - (top_pad_in / fig_h_in)])
        else:
            legend_y = 0.18 / fig_h_in
            top_pad_in = 0.35
            bottom_pad_in = 0.90
            fig.legend(handles=legend_handles, frameon=False, loc=legend_loc, bbox_to_anchor=(0.5, legend_y), bbox_transform=fig.transFigure, ncol=ncol, fontsize=legend_fontsize, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.0)
            fig.tight_layout(rect=[0.03, bottom_pad_in / fig_h_in, 0.97, 1.0 - (top_pad_in / fig_h_in)])
    else:
        fig.tight_layout(rect=[0.03, 0.04, 0.97, 0.97])

    cell_tables = {}
    for cell_key, sub in points_df.groupby('cell_key', dropna=False):
        cell_table_cols = [resolved_point_col, '_point_label', 'x_raw', 'y_raw', 'cell_key'] + ([effective_facet_col] if effective_facet_col is not None and effective_facet_col not in {resolved_point_col} else [])
        cell_tables[str(cell_key)] = sub[cell_table_cols].copy().reset_index(drop=True)
    cutpoints_used = {'x': (float(t1x), float(t2x)), 'y': (float(t1y), float(t2y))}
    return fig, axes[0] if len(axes) == 1 else axes, points_df, cutpoints_used, cell_tables

