from __future__ import annotations

import io
import zipfile
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from lib_charts import (
    make_assessment_long_format,
    make_assessment_from_aggregated_wide,
    plot_competency_distributions_subplots,
    plot_ridgeline_by_group,
    plot_competency_employee_rows_scatter,
    plot_competency_barplot,
    plot_group_heatmap,
    plot_9box,
)

st.set_page_config(page_title="MVP графиков", layout="wide")
st.title("MVP: загрузка данных и построение графиков")
st.caption("Локальный Streamlit MVP поверх функций из ноутбука")


# ---------- helpers ----------
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded_file)
    raise ValueError('Поддерживаются только CSV и Excel файлы.')


@st.cache_data(show_spinner=False)
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8-sig')


@st.cache_data(show_spinner=False)
def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    buffer.seek(0)
    return buffer.getvalue()


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=180, bbox_inches='tight')
    buffer.seek(0)
    return buffer.getvalue()


def build_png_zip(fig_bytes_by_name: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in fig_bytes_by_name.items():
            zf.writestr(name, data)
    buffer.seek(0)
    return buffer.getvalue()


def sorted_unique_values(series: pd.Series) -> list[Any]:
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda x: str(x))


def abbreviate_employee_name(value: object, mode: str = 'keep') -> str:
    if pd.isna(value):
        return ''
    original = str(value).strip()
    parts = [part for part in original.split() if part]
    if len(parts) != 3 or mode == 'keep':
        return original
    keep_idx = {'first': 0, 'second': 1, 'third': 2}.get(mode, 0)
    kept = parts[keep_idx]
    abbreviated = [f"{part[0]}." for idx, part in enumerate(parts) if idx != keep_idx and part]
    return ' '.join([kept] + abbreviated)


def get_available_values_after_filters(
    df: pd.DataFrame,
    target_col: str,
    filters: dict[str, Any] | None = None,
    target_values: list[Any] | None = None,
) -> list[Any]:
    work = df.copy()
    for fcol, fvals in (filters or {}).items():
        if fcol == target_col or fcol not in work.columns:
            continue
        if isinstance(fvals, (list, tuple, set)):
            work = work[work[fcol].isin(list(fvals))]
        else:
            work = work[work[fcol] == fvals]
    if target_col not in work.columns:
        return []
    present = set(work[target_col].dropna().astype(str).tolist())
    candidates = target_values if target_values is not None else sorted_unique_values(work[target_col])
    return [x for x in candidates if str(x) in present]


def role_column_index(columns: list[str]) -> int:
    for i, col in enumerate(columns):
        if str(col).strip().lower() == 'роль':
            return i
    return 0


def initialize_plot_state() -> None:
    st.session_state.setdefault('plot_done', False)
    st.session_state.setdefault('plot_fig_bytes', None)
    st.session_state.setdefault('plot_stats_df', None)
    st.session_state.setdefault('plot_error', None)
    st.session_state.setdefault('plot_zip_bytes', None)
    st.session_state.setdefault('plot_zip_name', None)


def clear_plot_state() -> None:
    st.session_state.plot_done = False
    st.session_state.plot_fig_bytes = None
    st.session_state.plot_stats_df = None
    st.session_state.plot_error = None
    st.session_state.plot_zip_bytes = None
    st.session_state.plot_zip_name = None


def reset_widget_state() -> None:
    prefixes = ('wide_', 'long_', 'aggregated_', 'plot_', 'group_filter::', 'hist_filter::', 'cutoff_', 'setup_', 'ridge_', 'role_ridge_')
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            del st.session_state[key]


def uploaded_file_signature(uploaded_file) -> tuple[str, int] | None:
    if uploaded_file is None:
        return None
    return (uploaded_file.name, int(getattr(uploaded_file, 'size', 0) or 0))


def reset_for_new_data() -> None:
    st.session_state.prepared_df = None
    st.session_state.group_cols = []
    clear_plot_state()
    reset_widget_state()


def get_default_scale_from_prepared_df(
    df: pd.DataFrame | None,
    competencies: list[str] | None = None,
) -> tuple[float | None, float | None]:
    if df is None or df.empty:
        return None, None
    if 'scale_min' not in df.columns or 'scale_max' not in df.columns:
        return None, None

    work = df.copy()
    if competencies:
        work = work[work['Компетенция'].isin(competencies)].copy()

    mins = pd.to_numeric(work['scale_min'], errors='coerce').dropna()
    maxs = pd.to_numeric(work['scale_max'], errors='coerce').dropna()

    if mins.empty or maxs.empty:
        return None, None

    return float(mins.min()), float(maxs.max())


def ensure_session_defaults() -> None:
    st.session_state.setdefault('raw_df', None)
    st.session_state.setdefault('prepared_df', None)
    st.session_state.setdefault('group_cols', [])
    st.session_state.setdefault('uploaded_file_signature', None)
    initialize_plot_state()


ensure_session_defaults()


# ---------- upload ----------
st.subheader('1. Загрузка файла')
uploaded_file = st.file_uploader('CSV / XLSX', type=['csv', 'xlsx', 'xls'])
current_signature = uploaded_file_signature(uploaded_file)

if current_signature != st.session_state.get('uploaded_file_signature'):
    st.session_state.uploaded_file_signature = current_signature
    st.session_state.raw_df = None
    reset_for_new_data()

    if uploaded_file is not None:
        try:
            raw_df = read_uploaded_file(uploaded_file)
            st.session_state.raw_df = raw_df
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')
            st.stop()

    st.rerun()

raw_df: pd.DataFrame | None = st.session_state.raw_df

if raw_df is None:
    st.info('Загрузи CSV или XLSX, чтобы продолжить.')
    st.stop()

st.write(f'Строк: **{len(raw_df):,}**, колонок: **{len(raw_df.columns)}**')
st.dataframe(raw_df.head(20), width='stretch')

columns = list(raw_df.columns)


# ---------- setup ----------
st.subheader('2. Настройка данных')
data_type = st.radio(
    'Тип входных данных',
    options=['wide', 'long', 'aggregated'],
    horizontal=True,
    help='wide = исходные/сырые данные; long = данные уже в строковом формате; aggregated = одна строка на сотрудника, а компетенции разложены по отдельным колонкам.',
)

left, right = st.columns(2)

with left:
    if data_type == 'wide':
        st.markdown('**Маппинг для преобразования в long**')
        respondent_col = st.selectbox('Колонка респондента', columns, index=0, key='wide_respondent_col')
        competency_col = st.selectbox('Колонка компетенции', columns, index=min(1, len(columns)-1), key='wide_competency_col')
        employee_col = st.selectbox('Колонка сотрудника', columns, index=min(2, len(columns)-1), key='wide_employee_col')
        value_col = st.selectbox('Колонка значения / score', columns, index=min(3, len(columns)-1), key='wide_value_col')
        role_enabled = st.checkbox('Есть колонка роли', value=('роль' in [str(c).strip().lower() for c in columns]), key='wide_role_enabled')
        role_col = st.selectbox(
            'Колонка роли',
            columns,
            index=role_column_index(columns),
            disabled=not role_enabled,
            key='wide_role_col',
        )

        excluded_cols = {respondent_col, competency_col, employee_col, value_col}
        if role_enabled and role_col:
            excluded_cols.add(role_col)

        available_group_cols = [c for c in columns if c not in excluded_cols]
        group_cols = st.multiselect(
            'Группирующие колонки (необязательно)',
            available_group_cols,
            default=[c for c in st.session_state.get('group_cols', []) if c in available_group_cols],
            key='wide_group_cols',
        )
        agg_label = st.selectbox('Аггрегация значения', ['Среднее', 'Медиана', 'Сумма'], index=0, key='wide_agg_label')
        agg = {'Среднее': 'mean', 'Медиана': 'median', 'Сумма': 'sum'}[agg_label]
        n_col = None
        aggregated_competency_cols = []
    elif data_type == 'aggregated':
        st.markdown('**Маппинг агрегированного wide-формата**')
        employee_col = st.selectbox('Колонка сотрудника', columns, index=0, key='aggregated_employee_col')
        role_enabled = st.checkbox('Есть колонка роли', value=('роль' in [str(c).strip().lower() for c in columns]), key='aggregated_role_enabled')
        role_col = st.selectbox(
            'Колонка роли',
            columns,
            index=role_column_index(columns),
            disabled=not role_enabled,
            key='aggregated_role_col',
        )

        excluded_for_groups = {employee_col}
        if role_enabled and role_col:
            excluded_for_groups.add(role_col)

        available_group_cols = [c for c in columns if c not in excluded_for_groups]
        group_cols = st.multiselect(
            'Группирующие колонки',
            available_group_cols,
            default=[c for c in st.session_state.get('group_cols', []) if c in available_group_cols],
            key='aggregated_group_cols',
        )

        excluded_for_competencies = set(excluded_for_groups) | set(group_cols)
        available_competency_value_cols = [c for c in columns if c not in excluded_for_competencies]
        aggregated_competency_cols = st.multiselect(
            'Колонки компетенций',
            options=available_competency_value_cols,
            default=available_competency_value_cols,
            key='aggregated_competency_value_cols',
        )
        competency_col = None
        value_col = None
        respondent_col = None
        agg = 'mean'
        n_col = None
    else:
        st.markdown('**Маппинг уже готового long-формата**')
        employee_col = st.selectbox('Колонка сотрудника', columns, index=columns.index('Сотрудник') if 'Сотрудник' in columns else 0, key='long_employee_col')
        competency_col = st.selectbox('Колонка компетенции', columns, index=columns.index('Компетенция') if 'Компетенция' in columns else min(1, len(columns)-1), key='long_competency_col')
        value_col = st.selectbox('Колонка score', columns, index=columns.index('score') if 'score' in columns else min(2, len(columns)-1), key='long_value_col')
        n_options = ['<создать n=1>'] + columns
        n_col = st.selectbox('Колонка n', n_options, index=n_options.index('n') if 'n' in columns else 0, key='long_n_col')
        role_enabled = st.checkbox('Есть колонка роли', value=('роль' in [str(c).strip().lower() for c in columns]), key='long_role_enabled')
        role_col = st.selectbox(
            'Колонка роли',
            columns,
            index=role_column_index(columns),
            disabled=not role_enabled,
            key='long_role_col',
        )

        excluded_cols = {employee_col, competency_col, value_col}
        if role_enabled and role_col:
            excluded_cols.add(role_col)

        available_group_cols = [c for c in columns if c not in excluded_cols]
        group_cols = st.multiselect(
            'Дополнительные групповые колонки',
            available_group_cols,
            default=[c for c in st.session_state.get('group_cols', []) if c in available_group_cols],
            key='long_group_cols',
        )
        agg = 'mean'
        aggregated_competency_cols = []

with right:
    st.markdown('**Пояснение**')
    if data_type == 'wide':
        st.write(
            'Для перевода из wide-формата в long-формат используется агрегация исходных строк '
            'по сотруднику / роли / группам / компетенции.'
        )
    elif data_type == 'aggregated':
        st.write(
            'В агрегированном wide-режиме каждая строка уже соответствует одному сотруднику, '
            'а компетенции хранятся в отдельных колонках. Преобразование выполняется через перенос этих колонок в long-формат, n создается как 1.'
        )
    else:
        st.write(
            'В long-режиме приложение просто приводит названия колонок к стандарту: '
            '`Сотрудник`, `роль`, `Компетенция`, `score`, `n`.'
        )

    with st.expander('Параметры шкалы для prepared long', expanded=True):
        st.caption('Глобальные границы шкалы будут записаны в prepared long и автоматически подставляться в гистограмму и риджплоты.')
        scale_c1, scale_c2 = st.columns(2)
        with scale_c1:
            prepared_scale_min = st.number_input(
                'Глобальный минимум шкалы',
                value=float(st.session_state.get('setup_scale_min', 0.0)),
                step=0.1,
                key='setup_scale_min',
            )
        with scale_c2:
            prepared_scale_max = st.number_input(
                'Глобальный максимум шкалы',
                value=float(st.session_state.get('setup_scale_max', 0.0)),
                step=0.1,
                key='setup_scale_max',
            )

        if float(prepared_scale_max) <= float(prepared_scale_min):
            st.error('Укажите корректную шкалу на этапе подготовки данных: максимум должен быть больше минимума.')

    with st.expander('Что получится после подготовки'):
        st.code('Сотрудник | роль | <group_cols> | Компетенция | score | n | scale_min | scale_max')

if st.button('Сохранить настройки / Подготовить данные', width='stretch'):
    try:
        if float(prepared_scale_max) <= float(prepared_scale_min):
            raise ValueError('Глобальный максимум шкалы должен быть больше минимума.')

        if data_type == 'wide':
            prepared_df = make_assessment_long_format(
                df=raw_df,
                group_cols=group_cols,
                respondent_col=respondent_col,
                competency_col=competency_col,
                employee_col=employee_col,
                value_col=value_col,
                role_col=role_col if role_enabled else None,
                agg=agg,
                scale_min=float(prepared_scale_min),
                scale_max=float(prepared_scale_max),
            )
        elif data_type == 'aggregated':
            prepared_df = make_assessment_from_aggregated_wide(
                df=raw_df,
                employee_col=employee_col,
                competency_value_cols=aggregated_competency_cols,
                group_cols=group_cols,
                role_col=role_col if role_enabled else None,
                scale_min=float(prepared_scale_min),
                scale_max=float(prepared_scale_max),
            )
        else:
            rename_map = {
                employee_col: 'Сотрудник',
                competency_col: 'Компетенция',
                value_col: 'score',
            }
            if role_enabled:
                rename_map[role_col] = 'роль'

            keep_cols = [employee_col, competency_col, value_col, *group_cols]
            if role_enabled:
                keep_cols.append(role_col)
            keep_cols = list(dict.fromkeys(keep_cols))

            prepared_df = raw_df[keep_cols].copy().rename(columns=rename_map)
            prepared_df['score'] = pd.to_numeric(prepared_df['score'], errors='coerce')
            prepared_df = prepared_df[prepared_df['score'].notna()].copy()

            if n_col == '<создать n=1>':
                prepared_df['n'] = 1
            else:
                prepared_df['n'] = pd.to_numeric(raw_df[n_col], errors='coerce').fillna(1).astype(int)

            prepared_df['scale_min'] = float(prepared_scale_min)
            prepared_df['scale_max'] = float(prepared_scale_max)

            ordered_cols = ['Сотрудник']
            if role_enabled:
                ordered_cols.append('роль')
            ordered_cols += group_cols + ['Компетенция', 'score', 'n', 'scale_min', 'scale_max']
            prepared_df = prepared_df[ordered_cols].reset_index(drop=True)

        st.session_state.prepared_df = prepared_df
        st.session_state.group_cols = group_cols
        clear_plot_state()
        st.success('Данные подготовлены.')
    except Exception as e:
        st.session_state.prepared_df = None
        clear_plot_state()
        st.error(f'Ошибка подготовки данных: {e}')

prepared_df: pd.DataFrame | None = st.session_state.prepared_df

if prepared_df is None:
    st.stop()

st.subheader('3. Рабочий long-формат')
st.write(f'Строк: **{len(prepared_df):,}**')
st.dataframe(prepared_df.head(50), width='stretch')

with st.expander('Скачать подготовленные данные'):
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            'Скачать CSV',
            data=df_to_csv_bytes(prepared_df),
            file_name='prepared_long.csv',
            mime='text/csv',
            width='stretch',
        )
    with c2:
        st.download_button(
            'Скачать XLSX',
            data=df_to_xlsx_bytes(prepared_df),
            file_name='prepared_long.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            width='stretch',
        )


# ---------- plotting ----------
st.subheader('4. Построение графика')
chart_name = st.selectbox(
    'Визуализация',
    options=['Гистограмма', 'Риджплот', 'Диаграмма рассеивания', 'Столбчатая диаграмма', 'Тепловая карта', '9-box'],
    help='Можно выбрать гистограмму, риджплот, диаграмму рассеивания, столбчатую диаграмму или тепловую карту.',
    key='plot_chart_name',
)

plot_group_cols = st.session_state.group_cols
has_role = 'роль' in prepared_df.columns
available_roles = sorted(prepared_df['роль'].dropna().astype(str).unique().tolist()) if has_role else []
available_competencies = sorted(prepared_df['Компетенция'].dropna().astype(str).unique().tolist())

base_group_candidates = [c for c in plot_group_cols if c in prepared_df.columns]
if not base_group_candidates:
    protected_cols = {'Сотрудник', 'роль', 'Компетенция', 'score', 'n'}
    base_group_candidates = [c for c in prepared_df.columns if c not in protected_cols]

hist_dimension_registry: dict[str, dict[str, Any]] = {
    'Компетенция': {
        'col': 'Компетенция',
        'values': available_competencies,
        'label': 'Компетенция',
    },
}
if has_role:
    hist_dimension_registry['роль'] = {
        'col': 'роль',
        'values': available_roles,
        'label': 'Роль',
    }
for gcol in plot_group_cols:
    if gcol in prepared_df.columns:
        hist_dimension_registry[gcol] = {
            'col': gcol,
            'values': sorted_unique_values(prepared_df[gcol]),
            'label': str(gcol),
        }

hist_default_vmin, hist_default_vmax = get_default_scale_from_prepared_df(prepared_df)
ridge_default_vmin, ridge_default_vmax = get_default_scale_from_prepared_df(prepared_df)
role_ridge_default_vmin, role_ridge_default_vmax = get_default_scale_from_prepared_df(prepared_df)

if chart_name == 'Гистограмма':
    p1, p2 = st.columns(2)
    with p1:
        hist_dimension_name = st.selectbox(
            'Характеристика для вывода',
            options=list(hist_dimension_registry.keys()),
            index=0,
            key='plot_hist_dimension_name',
        )
        hist_dimension_meta = hist_dimension_registry[hist_dimension_name]
        hist_category_col = hist_dimension_meta['col']
        hist_category_label = hist_dimension_meta['label']
        hist_category_options = hist_dimension_meta['values']

        default_hist_values = hist_category_options[: min(8, len(hist_category_options))] if hist_category_col == 'Компетенция' else hist_category_options
        selected_hist_values = st.multiselect(
            f'{hist_category_label}: порядок вывода',
            options=hist_category_options,
            default=default_hist_values,
            key='plot_hist_category_values',
        )

        hist_filters: dict[str, Any] = {}
        with st.expander('Фильтры', expanded=True):
            for dim_name, dim_meta in hist_dimension_registry.items():
                dim_col = dim_meta['col']
                if dim_col == hist_category_col:
                    continue
                options = dim_meta['values']
                selected = st.multiselect(
                    f'{dim_meta["label"]}',
                    options=options,
                    default=options,
                    format_func=lambda x: str(x),
                    key=f'hist_filter::{dim_col}',
                )
                if 0 < len(selected) < len(options):
                    hist_filters[dim_col] = selected

        st.caption('После фильтрации гистограмма всегда агрегирует данные до уровня: сотрудник × выбранная характеристика.')
        collapse_role_agg_label = st.selectbox(
            'Способ агрегации повторных строк сотрудника',
            ['Прямая', 'Последовательная'],
            index=0,
            key='plot_collapse_role_agg_label',
            help='Прямая = взвешенное среднее по n; Последовательная = простое среднее строк после фильтрации.',
        )
        collapse_role_agg = {'Прямая': 'weighted', 'Последовательная': 'mean'}[collapse_role_agg_label]

    with p2:
        with st.container(border=True):
            st.markdown('**Границы шкалы**')
            st.caption('По умолчанию используются scale_min / scale_max из prepared long, но их можно переопределить вручную.')
            scale_c1, scale_c2 = st.columns(2)
            with scale_c1:
                vmin = st.number_input('Минимум шкалы', value=float(hist_default_vmin) if hist_default_vmin is not None else 0.0, step=0.1, key='plot_vmin')
            with scale_c2:
                vmax = st.number_input('Максимум шкалы', value=float(hist_default_vmax) if hist_default_vmax is not None else 0.0, step=0.1, key='plot_vmax')
        infer_bin_step = st.checkbox(
            'Авто ширина шага',
            value=True,
            help='Автоматически определить ширину шага по данным. Если включено, ручная ширина шага не используется.',
            key='plot_infer_bin_step',
        )
        bin_step_input = st.number_input(
            'Ширина шага',
            min_value=0.0,
            value=0.0,
            step=0.1,
            disabled=infer_bin_step,
            key='plot_bin_step',
        )
        auto_cutoff_mode_label = st.selectbox(
            'Режим авто-отсечек',
            ['Внутри каждой категории', 'По всем сотрудникам'],
            index=0,
            help='Внутри каждой категории — зоны считаются по распределению выбранной категории. По всем сотрудникам — одна общая зона для всех выбранных категорий после фильтрации.',
            key='plot_auto_cutoff_mode_label',
        )
        auto_cutoff_mode = {
            'Внутри каждой категории': 'within_category',
            'По всем сотрудникам': 'global',
        }[auto_cutoff_mode_label]
        percentile_span = st.number_input(
            'Span от медианы, %',
            min_value=0.0,
            max_value=50.0,
            value=25.0,
            step=1.0,
            help='Авто-отсечки считаются как percentiles от медианы: 50%-span и 50%+span. Значение 25% соответствует прежним квартилям.',
            key='plot_percentile_span',
        )
        use_custom_q1q3 = st.checkbox(
            'Установить общие отсечки вручную',
            value=False,
            help='Если включено, график использует заданные вручную общие границы вместо автоматического расчёта по span от медианы.',
            key='plot_use_custom_q1q3',
        )
        q1_value = st.number_input(
            'Общая нижняя отсечка',
            value=0.0,
            step=0.1,
            disabled=not use_custom_q1q3,
            key='plot_q1',
        )
        q3_value = st.number_input(
            'Общая верхняя отсечка',
            value=0.0,
            step=0.1,
            disabled=not use_custom_q1q3,
            key='plot_q3',
        )
        bar_labels_mode = st.selectbox(
            'Показывать на столбиках',
            ['Абсолютные значения', 'Проценты', 'Ничего'],
            index=0,
            key='plot_bar_labels_mode',
        )
        show_mean_line = st.checkbox('Показывать среднюю линию', value=True, key='plot_show_mean_line')
        title = st.text_input('Заголовок', value=f'Распределение оценок по: {hist_dimension_label if False else hist_category_label}', key='plot_title')
        hist_show_xlabel_every_subplot = st.checkbox(
            'Указать подписи оси X на всех графиках',
            value=False,
            key='plot_show_xlabel_every_subplot',
        )

    extra_params: dict[str, Any] = {}
    with st.expander('Дополнительные параметры'):
        e1, e2 = st.columns(2)
        with e1:
            st.markdown('**Размер картинки, см**')
            size_c1, size_c2, size_c3 = st.columns([1, 1, 1])
            with size_c1:
                extra_params['fig_width_cm'] = st.number_input(
                    'Ширина',
                    min_value=5.0,
                    value=27.4,
                    step=0.1,
                    key='plot_fig_width_cm',
                )
            with size_c2:
                extra_params['height_per_comp_cm'] = st.number_input(
                    'Высота одного сабплота',
                    min_value=1.0,
                    value=4.8,
                    step=0.1,
                    key='plot_height_per_comp_cm',
                )
            with size_c3:
                total_height_cm = float(extra_params['height_per_comp_cm']) * max(len(selected_hist_values), 1)
                st.metric('Общая высота, см', f'{total_height_cm:.1f}')
            extra_params['alpha'] = st.number_input(
                'Прозрачность столбика',
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                key='plot_alpha',
            )
            extra_params['mean_color'] = st.color_picker(
                'Цвет линии среднего',
                value='#000000',
                key='plot_mean_color',
            )
        with e2:
            extra_params['low_color'] = st.color_picker('Цвет низкой зоны', value='#FF474A', key='plot_low_color')
            extra_params['mid_color'] = st.color_picker('Цвет средней зоны', value='#E0D9CE', key='plot_mid_color')
            extra_params['high_color'] = st.color_picker('Цвет высокой зоны', value='#30BAAD', key='plot_high_color')

        l1, l2, l3 = st.columns(3)
        with l1:
            extra_params['low_label'] = st.text_input('Подпись низкой зоны', value='', key='plot_low_label')
        with l2:
            extra_params['mid_label'] = st.text_input('Подпись средней зоны', value='', key='plot_mid_label')
        with l3:
            extra_params['high_label'] = st.text_input('Подпись высокой зоны', value='', key='plot_high_label')

        hist_show_grid = st.checkbox('Сетка', value=False, key='plot_show_grid')

        st.markdown('**Размеры шрифтов**')
        hf1, hf2, hf3, hf4 = st.columns(4)
        with hf1:
            extra_params['title_fontsize'] = st.number_input('Шрифт заголовка', min_value=6.0, value=14.0, step=1.0, key='plot_title_fontsize')
        with hf2:
            extra_params['tick_fontsize'] = st.number_input('Шрифт тиков', min_value=6.0, value=9.0, step=1.0, key='plot_tick_fontsize')
        with hf3:
            extra_params['legend_fontsize'] = st.number_input('Шрифт легенды', min_value=6.0, value=9.0, step=1.0, key='plot_legend_fontsize')
        with hf4:
            extra_params['value_fontsize'] = st.number_input('Шрифт значений внутри графика', min_value=6.0, value=8.0, step=1.0, key='plot_value_fontsize')

        st.markdown(f'**Отсечки вручную для характеристики: {hist_category_label}**')
        cutoffs_by_category: dict[str, tuple[float, float]] = {}
        if selected_hist_values:
            for cat_value in selected_hist_values:
                with st.container():
                    use_cat_cutoff = st.checkbox(
                        f'{cat_value}',
                        value=False,
                        key=f'cutoff_enabled::{hist_category_col}::{cat_value}',
                    )
                    c_q1, c_q3 = st.columns(2)
                    with c_q1:
                        cat_q1 = st.number_input(
                            f'Нижняя отсечка: {cat_value}',
                            value=0.0,
                            step=0.1,
                            disabled=not use_cat_cutoff,
                            key=f'cutoff_q1::{hist_category_col}::{cat_value}',
                        )
                    with c_q3:
                        cat_q3 = st.number_input(
                            f'Верхняя отсечка: {cat_value}',
                            value=0.0,
                            step=0.1,
                            disabled=not use_cat_cutoff,
                            key=f'cutoff_q3::{hist_category_col}::{cat_value}',
                        )
                    if use_cat_cutoff:
                        cutoffs_by_category[str(cat_value)] = (float(cat_q1), float(cat_q3))
        else:
            st.caption(f'Сначала выбери хотя бы одно значение для: {hist_category_label}.')

    if st.button('Построить', width='stretch', key='plot_build_hist'):
        if not selected_hist_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {hist_category_label}.'
        else:
            try:
                if float(vmax) <= float(vmin):
                    raise ValueError('Заполни корректные границы шкалы: максимум должен быть больше минимума.')

                manual_bin_step = None if infer_bin_step else (float(bin_step_input) if float(bin_step_input) > 0 else None)
                manual_q1q3 = (float(q1_value), float(q3_value)) if use_custom_q1q3 else None
                if manual_q1q3 is not None and manual_q1q3[1] <= manual_q1q3[0]:
                    raise ValueError('Верхняя отсечка должна быть больше нижней.')

                for cat_name, (cat_q1, cat_q3) in cutoffs_by_category.items():
                    if float(cat_q3) <= float(cat_q1):
                        raise ValueError(f'Для значения "{cat_name}" верхняя отсечка должна быть больше нижней.')

                fig_width_inches = float(extra_params['fig_width_cm']) / 2.54
                height_per_comp_inches = float(extra_params['height_per_comp_cm']) / 2.54

                fig, axes, stats_df = plot_competency_distributions_subplots(
                    vis_df=prepared_df,
                    category_col=hist_category_col,
                    category_values=[str(x) for x in selected_hist_values],
                    filters=hist_filters or None,
                    category_label=hist_category_label,
                    collapse_roles_to_employee=True,
                    collapse_role_agg=collapse_role_agg,
                    vmin=float(vmin),
                    vmax=float(vmax),
                    bin_step=manual_bin_step,
                    infer_bin_step=bool(infer_bin_step),
                    q1q3=manual_q1q3,
                    cutoffs_by_category=cutoffs_by_category or None,
                    auto_cutoff_mode=auto_cutoff_mode,
                    percentile_span=float(percentile_span),
                    show_counts=bar_labels_mode,
                    show_mean_line=show_mean_line,
                    title=title,
                    fig_width=fig_width_inches,
                    height_per_comp=height_per_comp_inches,
                    alpha=float(extra_params['alpha']),
                    mean_color=extra_params['mean_color'],
                    low_color=extra_params['low_color'],
                    mid_color=extra_params['mid_color'],
                    high_color=extra_params['high_color'],
                    low_label=extra_params['low_label'],
                    mid_label=extra_params['mid_label'],
                    high_label=extra_params['high_label'],
                    show_xlabel_every_subplot=bool(hist_show_xlabel_every_subplot),
                    title_fontsize=float(extra_params['title_fontsize']),
                    tick_fontsize=float(extra_params['tick_fontsize']),
                    legend_fontsize=float(extra_params['legend_fontsize']),
                    value_fontsize=float(extra_params['value_fontsize']),
                    show_grid=bool(hist_show_grid),
                )

                st.session_state.plot_fig_bytes = fig_to_png_bytes(fig)
                st.session_state.plot_stats_df = stats_df
                st.session_state.plot_done = True
                st.session_state.plot_error = None
                plt.close(fig)
            except Exception as e:
                clear_plot_state()
                st.session_state.plot_error = f'Ошибка построения графика: {e}'
elif chart_name == 'Риджплот':
    ridge_dimension_registry = hist_dimension_registry

    rp1, rp2 = st.columns(2)
    with rp1:
        ridge_dimension_name = st.selectbox(
            'Характеристика для вывода',
            options=list(ridge_dimension_registry.keys()),
            index=0,
            key='ridge_dimension_name',
        )
        ridge_dimension_meta = ridge_dimension_registry[ridge_dimension_name]
        ridge_category_col = ridge_dimension_meta['col']
        ridge_category_label = ridge_dimension_meta['label']
        ridge_category_options = ridge_dimension_meta['values']
        ridge_default_values = ridge_category_options[: min(8, len(ridge_category_options))] if ridge_category_col == 'Компетенция' else ridge_category_options
        ridge_category_values = st.multiselect(
            f'{ridge_category_label}: порядок вывода',
            options=ridge_category_options,
            default=ridge_default_values,
            key='ridge_category_values',
        )

        facet_options = ['<без разбивки>'] + [name for name in ridge_dimension_registry.keys() if ridge_dimension_registry[name]['col'] != ridge_category_col]
        ridge_facet_name = st.selectbox(
            'Дополнительная разбивка на графики',
            options=facet_options,
            index=0,
            key='ridge_facet_name',
        )
        ridge_facet_col = None
        ridge_facet_label = None
        ridge_facet_values = None
        if ridge_facet_name != '<без разбивки>':
            ridge_facet_meta = ridge_dimension_registry[ridge_facet_name]
            ridge_facet_col = ridge_facet_meta['col']
            ridge_facet_label = ridge_facet_meta['label']
            ridge_facet_values = st.multiselect(
                f'{ridge_facet_label}: порядок графиков',
                options=ridge_facet_meta['values'],
                default=ridge_facet_meta['values'],
                key='ridge_facet_values',
            )

        ridge_filters: dict[str, Any] = {}
        with st.expander('Фильтры', expanded=True):
            for dim_name, dim_meta in ridge_dimension_registry.items():
                dim_col = dim_meta['col']
                if dim_col == ridge_category_col or dim_col == ridge_facet_col:
                    continue
                options = dim_meta['values']
                selected = st.multiselect(
                    f'{dim_meta["label"]}',
                    options=options,
                    default=options,
                    format_func=lambda x: str(x),
                    key=f'ridge_filter::{dim_col}',
                )
                if 0 < len(selected) < len(options):
                    ridge_filters[dim_col] = selected

        st.caption('После фильтрации риджплот агрегирует данные до уровня: сотрудник × выбранная характеристика. Если задана дополнительная разбивка — до сотрудник × выбранная характеристика × значение разбивки.')
        ridge_agg_label = st.selectbox(
            'Способ агрегации повторных строк сотрудника',
            ['Прямая', 'Последовательная'],
            index=0,
            key='ridge_agg_label',
            help='Прямая = взвешенное среднее по n; Последовательная = простое среднее строк после фильтрации.',
        )
        ridge_agg_mode = {'Прямая': 'weighted', 'Последовательная': 'mean'}[ridge_agg_label]

    with rp2:
        with st.container(border=True):
            st.markdown('**Границы шкалы**')
            st.caption('По умолчанию используются scale_min / scale_max из prepared long, но их можно переопределить вручную.')
            rv1, rv2 = st.columns(2)
            with rv1:
                ridge_vmin = st.number_input('Минимум шкалы', value=float(ridge_default_vmin) if ridge_default_vmin is not None else 0.0, step=0.1, key='ridge_vmin')
            with rv2:
                ridge_vmax = st.number_input('Максимум шкалы', value=float(ridge_default_vmax) if ridge_default_vmax is not None else 0.0, step=0.1, key='ridge_vmax')
        ridge_show_reference_median = st.checkbox('Показывать референсную медиану', value=True, key='ridge_show_reference_median')
        ridge_show_median = st.checkbox('Показывать медиану', value=True, key='ridge_show_median')
        ridge_show_mean_line = st.checkbox('Показывать точку среднего', value=False, key='ridge_show_mean_line')
        ridge_show_intervals = st.checkbox('Показывать 50% интервал', value=True, key='ridge_show_intervals')
        ridge_show_xlabel_every_subplot = st.checkbox(
            'Указать подписи оси X на всех графиках',
            value=False,
            key='ridge_show_xlabel_every_subplot',
        )

    ridge_extra: dict[str, Any] = {}
    with st.expander('Дополнительные параметры'):
        re1, re2 = st.columns(2)
        with re1:
            st.markdown('**Размер картинки, см**')
            rs1, rs2, rs3 = st.columns(3)
            with rs1:
                ridge_extra['fig_width_cm'] = st.number_input('Ширина', min_value=5.0, value=27.4, step=0.1, key='ridge_fig_width_cm')
            with rs2:
                ridge_extra['height_per_facet_cm'] = st.number_input('Высота одного графика', min_value=2.0, value=12.2, step=0.1, key='ridge_height_per_facet_cm')
            with rs3:
                ridge_total_facets = max(len(ridge_facet_values), 1) if ridge_facet_col else 1
                ridge_total_height_cm = float(ridge_extra['height_per_facet_cm']) * max(ridge_total_facets, 1)
                st.metric('Общая высота, см', f'{ridge_total_height_cm:.1f}')
            ridge_extra['palette'] = st.selectbox('Палитра гребней', ['Стандартная', 'Холодная', 'Теплая', 'Пастельная', 'Своя'], index=0, key='ridge_palette')
            ridge_extra['bandwidth'] = st.number_input('Сглаживание', min_value=0.05, value=0.5, step=0.05, key='ridge_bandwidth')
            ridge_extra['ridge_height'] = st.number_input('Высота гребня', min_value=0.1, value=0.8, step=0.1, key='ridge_height')
            ridge_extra['scale'] = st.number_input('Масштаб', min_value=0.1, value=0.9, step=0.1, key='ridge_scale')
            ridge_extra['x_tick_step'] = st.number_input('Шаг по оси X', min_value=0.0, value=0.0, step=0.1, key='ridge_x_tick_step')
            ridge_extra['ncols'] = st.number_input('Колонок графиков', min_value=1, value=1, step=1, key='ridge_ncols')
        with re2:
            ridge_extra['overlay_ridges'] = st.checkbox('Накладывать гребни', value=True, key='ridge_overlay_ridges')
            ridge_extra['ridge_overlap'] = st.number_input('Перекрытие гребней', min_value=0.0, max_value=0.95, value=0.65, step=0.05, key='ridge_overlap')
            ridge_extra['ridge_alpha'] = st.number_input('Прозрачность гребней', min_value=0.05, max_value=1.0, value=0.45, step=0.05, key='ridge_alpha')
            group_label_mode_label = st.selectbox('Подписи категорий', ['Ось Y', 'Легенда'], index=0, key='ridge_group_label_mode')
            ridge_extra['group_label_mode'] = 'axis' if group_label_mode_label == 'Ось Y' else 'legend'
            sort_groups_label = st.selectbox('Сортировка категорий', ['Как в списке', 'По медиане', 'По среднему'], index=0, key='ridge_sort_groups_by')
            ridge_extra['sort_groups_by'] = {'Как в списке': None, 'По медиане': 'median', 'По среднему': 'mean'}[sort_groups_label]
            ridge_extra['sort_ascending'] = st.checkbox('Сортировать по возрастанию', value=False, key='ridge_sort_ascending')
            ridge_extra['show_n_right'] = st.checkbox('Показывать n справа', value=False, key='ridge_show_n_right')
            ridge_stat_label_mode_label = st.selectbox('Подписи значения', ['Отсутствуют', 'Медиана', 'Медиана и квартили'], index=0, key='ridge_stat_label_mode')
            ridge_extra['stat_label_mode'] = {'Отсутствуют': 'none', 'Медиана': 'median', 'Медиана и квартили': 'median_quartiles'}[ridge_stat_label_mode_label]
            ridge_extra['show_interval_80'] = st.checkbox('Показывать 80% интервал', value=False, key='ridge_show_interval_80')
            ridge_extra['show_interval_95'] = st.checkbox('Показывать 95% интервал', value=False, key='ridge_show_interval_95')
            ridge_extra['show_grid'] = st.checkbox('Сетка', value=True, key='ridge_show_grid')
            ridge_legend_position_label = st.selectbox('Размещение легенды', ['Вверху', 'Внизу', 'Отсутствует'], index=0, key='ridge_legend_position')
            ridge_extra['legend_position'] = {'Вверху': 'top', 'Внизу': 'bottom', 'Отсутствует': 'none'}[ridge_legend_position_label]

        if ridge_extra['palette'] == 'Своя':
            st.markdown('**Цвета палитры**')
            custom_palette_cols = st.columns(4)
            default_group_palette = ['#4E79A7', '#E15759', '#76B7B2', '#F28E2B', '#59A14F', '#B07AA1', '#9C755F']
            ridge_extra['custom_palette'] = []
            for i, default_color in enumerate(default_group_palette):
                with custom_palette_cols[i % 4]:
                    ridge_extra['custom_palette'].append(
                        st.color_picker(f'Цвет {i+1}', value=default_color, key=f'ridge_custom_color_{i}')
                    )
        else:
            ridge_extra['custom_palette'] = None

        st.markdown('**Размеры шрифтов**')
        rgf1, rgf2, rgf3, rgf4 = st.columns(4)
        with rgf1:
            ridge_extra['title_fontsize'] = st.number_input('Шрифт заголовка', min_value=6.0, value=13.0, step=1.0, key='ridge_title_fontsize')
        with rgf2:
            ridge_extra['tick_fontsize'] = st.number_input('Шрифт тиков', min_value=6.0, value=10.0, step=1.0, key='ridge_tick_fontsize')
        with rgf3:
            ridge_extra['legend_fontsize'] = st.number_input('Шрифт легенды', min_value=6.0, value=9.0, step=1.0, key='ridge_legend_fontsize')
        with rgf4:
            ridge_extra['value_fontsize'] = st.number_input('Шрифт значений внутри графика', min_value=6.0, value=9.0, step=1.0, key='ridge_value_fontsize')

    if st.button('Построить', width='stretch', key='plot_build_ridge'):
        if not ridge_category_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {ridge_category_label}.'
        elif ridge_facet_col and not ridge_facet_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {ridge_facet_label}.'
        else:
            try:
                if float(ridge_vmax) <= float(ridge_vmin):
                    raise ValueError('Заполни корректные границы шкалы: максимум должен быть больше минимума.')

                ridge_fig_width_inches = float(ridge_extra['fig_width_cm']) / 2.54
                ridge_height_per_facet_inches = float(ridge_extra['height_per_facet_cm']) / 2.54
                ridge_x_tick_step = None if float(ridge_extra['x_tick_step']) <= 0 else float(ridge_extra['x_tick_step'])
                interval_levels = []
                if ridge_show_intervals:
                    interval_levels.append(0.50)
                if ridge_extra['show_interval_80']:
                    interval_levels.append(0.80)
                if ridge_extra['show_interval_95']:
                    interval_levels.append(0.95)
                if not interval_levels:
                    interval_levels = [0.50]

                palette_map = {
                    'Стандартная': 'default',
                    'Холодная': 'cool',
                    'Теплая': 'warm',
                    'Пастельная': 'pastel',
                    'Своя': ridge_extra.get('custom_palette') or 'default',
                }

                zip_bytes = None
                zip_name = None
                display_facet_values = ridge_facet_values if ridge_facet_col else None
                ridge_stats_df = None

                filtered_for_facet = prepared_df.copy()
                if ridge_filters:
                    for fcol, fval in ridge_filters.items():
                        if fcol in {ridge_category_col, ridge_facet_col}:
                            continue
                        if isinstance(fval, (list, tuple, set)):
                            filtered_for_facet = filtered_for_facet[filtered_for_facet[fcol].isin(list(fval))]
                        else:
                            filtered_for_facet = filtered_for_facet[filtered_for_facet[fcol] == fval]

                available_ridge_facet_values = None
                if ridge_facet_col is not None:
                    present_facet_values = set(filtered_for_facet[ridge_facet_col].dropna().astype(str).unique().tolist())
                    requested_facet_values = ridge_facet_values if ridge_facet_values is not None else []
                    available_ridge_facet_values = [v for v in requested_facet_values if str(v) in present_facet_values]
                    display_facet_values = available_ridge_facet_values

                if ridge_facet_col is not None and display_facet_values is not None and len(display_facet_values) > 5:
                    display_facet_values = list(display_facet_values)[:5]
                    pngs: dict[str, bytes] = {}
                    ridge_stats_frames = []
                    for facet_value in available_ridge_facet_values:
                        one_filters = dict(ridge_filters or {})
                        one_filters[ridge_facet_col] = [facet_value]
                        one_fig, _, one_stats_df = plot_ridgeline_by_group(
                            vis_df=prepared_df,
                            category_col=ridge_category_col,
                            category_values=ridge_category_values,
                            filters=one_filters or None,
                            facet_col=None,
                            facet_values=None,
                            category_label=ridge_category_label,
                            facet_label=None,
                            score_col='score',
                            employee_col='Сотрудник',
                            n_col='n',
                            agg_mode=ridge_agg_mode,
                            vmin=float(ridge_vmin),
                            vmax=float(ridge_vmax),
                            x_tick_step=ridge_x_tick_step,
                            bandwidth=float(ridge_extra['bandwidth']),
                            ridge_height=float(ridge_extra['ridge_height']),
                            scale=float(ridge_extra['scale']),
                            overlay_ridges=bool(ridge_extra['overlay_ridges']),
                            ridge_overlap=float(ridge_extra['ridge_overlap']),
                            ridge_alpha=float(ridge_extra['ridge_alpha']),
                            group_label_mode=ridge_extra['group_label_mode'],
                            sort_groups_by=ridge_extra['sort_groups_by'],
                            sort_ascending=bool(ridge_extra['sort_ascending']),
                            show_reference_median=bool(ridge_show_reference_median),
                            show_median=bool(ridge_show_median),
                            show_mean_line=bool(ridge_show_mean_line),
                            show_intervals=bool(ridge_show_intervals),
                            interval_levels=tuple(interval_levels),
                            show_n_right=bool(ridge_extra['show_n_right']),
                            stat_label_mode=ridge_extra['stat_label_mode'],
                            figsize_per_facet=(ridge_fig_width_inches, ridge_height_per_facet_inches),
                            ncols=1,
                            ridge_palette=palette_map[ridge_extra['palette']],
                            title_fontsize=float(ridge_extra['title_fontsize']),
                            tick_fontsize=float(ridge_extra['tick_fontsize']),
                            legend_fontsize=float(ridge_extra['legend_fontsize']),
                            value_fontsize=float(ridge_extra['value_fontsize']),
                            show_xlabel_every_subplot=bool(ridge_show_xlabel_every_subplot),
                            show_grid=bool(ridge_extra['show_grid']),
                            legend_position=ridge_extra['legend_position'],
                        )
                        safe_name = str(facet_value).replace('/', '-').replace('\\', '-').strip() or 'facet'
                        pngs[f'{safe_name}.png'] = fig_to_png_bytes(one_fig)
                        if isinstance(one_stats_df, pd.DataFrame) and not one_stats_df.empty:
                            one_stats_df = one_stats_df.copy()
                            one_stats_df['export_facet'] = str(facet_value)
                            ridge_stats_frames.append(one_stats_df)
                        plt.close(one_fig)
                    zip_bytes = build_png_zip(pngs)
                    facet_slug = str(ridge_facet_label or ridge_facet_col).strip().replace(' ', '_').replace('/', '-').replace('\\', '-').lower() or 'facet'
                    zip_name = f'ridgeline_facets_by_{facet_slug}.zip'
                    if ridge_stats_frames:
                        ridge_stats_df = pd.concat(ridge_stats_frames, ignore_index=True)

                fig, axes, preview_stats_df = plot_ridgeline_by_group(
                    vis_df=prepared_df,
                    category_col=ridge_category_col,
                    category_values=ridge_category_values,
                    filters=ridge_filters or None,
                    facet_col=ridge_facet_col,
                    facet_values=display_facet_values,
                    category_label=ridge_category_label,
                    facet_label=ridge_facet_label,
                    score_col='score',
                    employee_col='Сотрудник',
                    n_col='n',
                    agg_mode=ridge_agg_mode,
                    vmin=float(ridge_vmin),
                    vmax=float(ridge_vmax),
                    x_tick_step=ridge_x_tick_step,
                    bandwidth=float(ridge_extra['bandwidth']),
                    ridge_height=float(ridge_extra['ridge_height']),
                    scale=float(ridge_extra['scale']),
                    overlay_ridges=bool(ridge_extra['overlay_ridges']),
                    ridge_overlap=float(ridge_extra['ridge_overlap']),
                    ridge_alpha=float(ridge_extra['ridge_alpha']),
                    group_label_mode=ridge_extra['group_label_mode'],
                    sort_groups_by=ridge_extra['sort_groups_by'],
                    sort_ascending=bool(ridge_extra['sort_ascending']),
                    show_reference_median=bool(ridge_show_reference_median),
                    show_median=bool(ridge_show_median),
                    show_mean_line=bool(ridge_show_mean_line),
                    show_intervals=bool(ridge_show_intervals),
                    interval_levels=tuple(interval_levels),
                    show_n_right=bool(ridge_extra['show_n_right']),
                    stat_label_mode=ridge_extra['stat_label_mode'],
                    figsize_per_facet=(ridge_fig_width_inches, ridge_height_per_facet_inches),
                    ncols=int(ridge_extra['ncols']),
                    ridge_palette=palette_map[ridge_extra['palette']],
                    title_fontsize=float(ridge_extra['title_fontsize']),
                    tick_fontsize=float(ridge_extra['tick_fontsize']),
                    legend_fontsize=float(ridge_extra['legend_fontsize']),
                    value_fontsize=float(ridge_extra['value_fontsize']),
                    show_xlabel_every_subplot=bool(ridge_show_xlabel_every_subplot),
                    show_grid=bool(ridge_extra['show_grid']),
                    legend_position=ridge_extra['legend_position'],
                )

                st.session_state.plot_fig_bytes = fig_to_png_bytes(fig)
                st.session_state.plot_stats_df = ridge_stats_df if ridge_stats_df is not None else preview_stats_df
                st.session_state.plot_done = True
                st.session_state.plot_error = None
                st.session_state.plot_zip_bytes = zip_bytes
                st.session_state.plot_zip_name = zip_name
                plt.close(fig)
            except Exception as e:
                clear_plot_state()
                st.session_state.plot_error = f'Ошибка построения графика: {e}'
elif chart_name == 'Диаграмма рассеивания':
    scatter_dimension_registry: dict[str, dict[str, Any]] = dict(hist_dimension_registry)
    scatter_filter_registry: dict[str, dict[str, Any]] = dict(scatter_dimension_registry)
    if 'Сотрудник' in prepared_df.columns:
        scatter_filter_registry['Сотрудник'] = {
            'col': 'Сотрудник',
            'values': sorted_unique_values(prepared_df['Сотрудник']),
            'label': 'Сотрудник',
        }
    scatter_facet_registry: dict[str, dict[str, Any]] = dict(scatter_filter_registry)

    sp1, sp2 = st.columns(2)
    with sp1:
        scatter_dimension_name = st.selectbox(
            'Характеристика для вывода',
            options=list(scatter_dimension_registry.keys()),
            index=0,
            key='scatter_dimension_name',
        )
        scatter_dimension_meta = scatter_dimension_registry[scatter_dimension_name]
        scatter_category_col = scatter_dimension_meta['col']
        scatter_category_label = scatter_dimension_meta['label']
        scatter_category_options = scatter_dimension_meta['values']
        scatter_default_values = scatter_category_options[: min(8, len(scatter_category_options))] if scatter_category_col == 'Компетенция' else scatter_category_options
        scatter_category_values = st.multiselect(
            f'{scatter_category_label}: порядок вывода',
            options=scatter_category_options,
            default=scatter_default_values,
            key='scatter_category_values',
        )

        facet_candidates = ['Без разбивки'] + [name for name in scatter_facet_registry.keys() if scatter_facet_registry[name]['col'] != scatter_category_col]
        scatter_facet_name = st.selectbox(
            'Дополнительная разбивка на графики',
            options=facet_candidates,
            index=0,
            key='scatter_facet_name',
        )
        if scatter_facet_name == 'Без разбивки':
            scatter_facet_col = None
            scatter_facet_label = None
            scatter_facet_values = None
        else:
            scatter_facet_meta = scatter_facet_registry[scatter_facet_name]
            scatter_facet_col = scatter_facet_meta['col']
            scatter_facet_label = scatter_facet_meta['label']
            scatter_facet_values = st.multiselect(
                f'{scatter_facet_label}: порядок вывода графиков',
                options=scatter_facet_meta['values'],
                default=scatter_facet_meta['values'],
                key='scatter_facet_values',
            )

        scatter_filters: dict[str, Any] = {}
        with st.expander('Фильтры', expanded=True):
            for dim_name, dim_meta in scatter_filter_registry.items():
                dim_col = dim_meta['col']
                if dim_col in {scatter_category_col, scatter_facet_col}:
                    continue
                options = dim_meta['values']
                selected = st.multiselect(
                    f'{dim_meta["label"]}',
                    options=options,
                    default=options,
                    format_func=lambda x: str(x),
                    key=f'scatter_filter::{dim_col}',
                )
                if 0 < len(selected) < len(options):
                    scatter_filters[dim_col] = selected

        st.caption('После фильтрации диаграмма рассеивания агрегирует данные до уровня: сотрудник × выбранная характеристика. Если задана дополнительная разбивка — графики делятся по ее значениям.')
        if scatter_facet_col is not None:
            st.caption('Если по дополнительной разбивке получается больше 5 графиков, на экране показываются первые 5, а полный набор можно скачать ZIP-архивом.')
        scatter_agg_label = st.selectbox(
            'Способ агрегации повторных строк сотрудника',
            ['Прямая', 'Последовательная'],
            index=0,
            key='scatter_agg_label',
        )
        scatter_agg = {'Прямая': 'weighted', 'Последовательная': 'mean'}[scatter_agg_label]

    with sp2:
        scatter_min_total_n = st.number_input('Минимальное количество оценок для сотрудника', min_value=1, value=1, step=1, key='scatter_min_total_n')
        scatter_employee_legend_limit = st.number_input('Лимит легенды сотрудников', min_value=0, value=10, step=1, key='scatter_employee_legend_limit')
        scatter_employee_name_mode = st.selectbox(
            'Формат вывода сотрудников',
            ['Не менять', 'Первое слово', 'Второе слово', 'Третье слово'],
            index=0,
            key='scatter_employee_name_mode',
        )
        scatter_show_grid = st.checkbox('Показывать сетку', value=False, key='scatter_show_grid')
        scatter_show_employee_legend = st.checkbox('Показывать легенду сотрудников', value=True, key='scatter_show_employee_legend')
        scatter_show_mean_value_label = st.checkbox('Подпись значения среднего', value=False, key='scatter_show_mean_value_label')
        scatter_title_mode = st.selectbox(
            'Заголовок',
            ['Без заголовка', 'Автоматический'],
            index=0,
            key='scatter_title_mode',
        )

    scatter_extra: dict[str, Any] = {}
    with st.expander('Дополнительные параметры'):
        se1, se2 = st.columns(2)
        with se1:
            st.markdown('**Размер картинки, см**')
            ss1, ss2 = st.columns(2)
            with ss1:
                scatter_extra['fig_width_cm'] = st.number_input('Ширина', min_value=5.0, value=30.5, step=0.1, key='scatter_fig_width_cm')
            with ss2:
                scatter_extra['fig_height_cm'] = st.number_input('Высота одного графика', min_value=3.0, value=10.2, step=0.1, key='scatter_fig_height_cm')
            scatter_extra['row_gap'] = st.number_input('Расстояние между строками', min_value=0.1, value=0.8, step=0.1, key='scatter_row_gap')
            scatter_extra['x_tick_step'] = st.number_input('Шаг по оси X', min_value=0.0, value=0.0, step=0.1, key='scatter_x_tick_step')
            scatter_extra['x_pad_frac'] = st.number_input('Отступ по оси X', min_value=0.0, value=0.05, step=0.01, key='scatter_x_pad_frac')
        with se2:
            scatter_extra['point_size'] = st.number_input('Размер точек сотрудников', min_value=1.0, value=80.0, step=1.0, key='scatter_point_size')
            scatter_extra['mean_point_size'] = st.number_input('Размер точки среднего', min_value=1.0, value=100.0, step=1.0, key='scatter_mean_point_size')
            scatter_extra['alpha'] = st.number_input('Прозрачность точек', min_value=0.0, max_value=1.0, value=0.9, step=0.05, key='scatter_alpha')
            scatter_extra['mean_point_color'] = st.color_picker('Цвет точки среднего', value='#121111', key='scatter_mean_point_color')
            scatter_extra['comp_wrap_width'] = st.number_input('Ширина переноса названий категорий', min_value=5, value=20, step=1, key='scatter_comp_wrap_width')
            scatter_extra['comp_max_lines'] = st.number_input('Максимум строк названия', min_value=1, value=3, step=1, key='scatter_comp_max_lines')

        st.markdown('**Размеры шрифтов**')
        sf1, sf2, sf3, sf4 = st.columns(4)
        with sf1:
            scatter_extra['title_fontsize'] = st.number_input('Шрифт заголовка', min_value=6.0, value=14.0, step=1.0, key='scatter_title_fontsize')
        with sf2:
            scatter_extra['tick_fontsize'] = st.number_input('Шрифт тиков', min_value=6.0, value=10.0, step=1.0, key='scatter_tick_fontsize')
        with sf3:
            scatter_extra['legend_fontsize'] = st.number_input('Шрифт легенды', min_value=6.0, value=9.0, step=1.0, key='scatter_legend_fontsize')
        with sf4:
            scatter_extra['value_fontsize'] = st.number_input('Шрифт значений внутри графика', min_value=6.0, value=9.0, step=1.0, key='scatter_value_fontsize')

        scatter_palette_mode = st.selectbox('Палитра сотрудников', ['Стандартная', 'Своя'], index=0, key='scatter_palette_mode')
        if scatter_palette_mode == 'Своя':
            st.markdown('**Цвета сотрудников**')
            default_scatter_palette = ['#FF474A', '#495867', '#30BAAD', '#CBBFAD', '#FA4D6E', '#FF762F', '#1ADC92', '#786EE0', '#8B5E3C', '#F2C94C']
            palette_cols = st.columns(5)
            scatter_extra['employee_colors'] = []
            for i, default_color in enumerate(default_scatter_palette):
                with palette_cols[i % 5]:
                    scatter_extra['employee_colors'].append(
                        st.color_picker(f'Цвет {i+1}', value=default_color, key=f'scatter_custom_color_{i}')
                    )
        else:
            scatter_extra['employee_colors'] = None

    if st.button('Построить', width='stretch', key='plot_build_scatter'):
        if not scatter_category_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {scatter_category_label}.'
        elif scatter_facet_col is not None and not scatter_facet_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {scatter_facet_label}.'
        else:
            try:
                scatter_fig_width_inches = float(scatter_extra['fig_width_cm']) / 2.54
                scatter_fig_height_inches = float(scatter_extra['fig_height_cm']) / 2.54
                scatter_x_tick_step = None if float(scatter_extra['x_tick_step']) <= 0 else float(scatter_extra['x_tick_step'])
                plot_title = (False if scatter_title_mode == 'Без заголовка' else True)
                scatter_name_mode = {
                    'Не менять': 'keep',
                    'Первое слово': 'first',
                    'Второе слово': 'second',
                    'Третье слово': 'third',
                }[scatter_employee_name_mode]

                zip_bytes = None
                zip_name = None
                display_facet_values = scatter_facet_values

                filtered_for_facet = prepared_df.copy()
                if scatter_filters:
                    for fcol, fval in scatter_filters.items():
                        if fcol in {scatter_category_col, scatter_facet_col}:
                            continue
                        if isinstance(fval, (list, tuple, set)):
                            filtered_for_facet = filtered_for_facet[filtered_for_facet[fcol].isin(list(fval))]
                        else:
                            filtered_for_facet = filtered_for_facet[filtered_for_facet[fcol] == fval]

                available_facet_values_after_filters = None
                if scatter_facet_col is not None:
                    present_facet_values = set(filtered_for_facet[scatter_facet_col].dropna().astype(str).unique().tolist())
                    requested_facet_values = scatter_facet_values if scatter_facet_values is not None else []
                    available_facet_values_after_filters = [v for v in requested_facet_values if str(v) in present_facet_values]
                    display_facet_values = available_facet_values_after_filters

                if scatter_facet_col is not None and display_facet_values is not None and len(display_facet_values) > 5:
                    display_facet_values = list(display_facet_values)[:5]
                    pngs: dict[str, bytes] = {}
                    for facet_value in available_facet_values_after_filters:
                        one_fig, _, _, _ = plot_competency_employee_rows_scatter(
                            vis_df=prepared_df,
                            category_col=scatter_category_col,
                            category_values=scatter_category_values,
                            category_label=scatter_category_label,
                            facet_col=scatter_facet_col,
                            facet_values=[facet_value],
                            facet_label=scatter_facet_label,
                            filters=scatter_filters or None,
                            agg=scatter_agg,
                            min_total_n=int(scatter_min_total_n),
                            figsize=(scatter_fig_width_inches, scatter_fig_height_inches),
                            row_gap=float(scatter_extra['row_gap']),
                            employee_legend_limit=int(scatter_employee_legend_limit),
                            x_pad_frac=float(scatter_extra['x_pad_frac']),
                            x_tick_step=scatter_x_tick_step,
                            show_grid=bool(scatter_show_grid),
                            mean_point_color=scatter_extra['mean_point_color'],
                            mean_point_size=float(scatter_extra['mean_point_size']),
                            point_size=float(scatter_extra['point_size']),
                            alpha=float(scatter_extra['alpha']),
                            title=plot_title,
                            employee_colors=scatter_extra['employee_colors'],
                            comp_wrap_width=int(scatter_extra['comp_wrap_width']),
                            comp_max_lines=int(scatter_extra['comp_max_lines']),
                            show_employee_legend=bool(scatter_show_employee_legend),
                            show_mean_value_label=bool(scatter_show_mean_value_label),
                            title_fontsize=float(scatter_extra['title_fontsize']),
                            tick_fontsize=float(scatter_extra['tick_fontsize']),
                            legend_fontsize=float(scatter_extra['legend_fontsize']),
                            value_fontsize=float(scatter_extra['value_fontsize']),
                            employee_name_format=scatter_name_mode,
                        )
                        safe_name = str(facet_value).replace('/', '-').replace('\\', '-').strip() or 'facet'
                        pngs[f'{safe_name}.png'] = fig_to_png_bytes(one_fig)
                        plt.close(one_fig)
                    zip_bytes = build_png_zip(pngs)
                    facet_slug = str(scatter_facet_label or scatter_facet_col).strip().replace(' ', '_').replace('/', '-').replace('\\', '-').lower() or 'facet'
                    zip_name = f'scatter_facets_by_{facet_slug}.zip'
                fig, ax, scatter_stats_df, emp_dfs = plot_competency_employee_rows_scatter(
                    vis_df=prepared_df,
                    category_col=scatter_category_col,
                    category_values=scatter_category_values,
                    category_label=scatter_category_label,
                    facet_col=scatter_facet_col,
                    facet_values=display_facet_values,
                    facet_label=scatter_facet_label,
                    filters=scatter_filters or None,
                    agg=scatter_agg,
                    min_total_n=int(scatter_min_total_n),
                    figsize=(scatter_fig_width_inches, scatter_fig_height_inches),
                    row_gap=float(scatter_extra['row_gap']),
                    employee_legend_limit=int(scatter_employee_legend_limit),
                    x_pad_frac=float(scatter_extra['x_pad_frac']),
                    x_tick_step=scatter_x_tick_step,
                    show_grid=bool(scatter_show_grid),
                    mean_point_color=scatter_extra['mean_point_color'],
                    mean_point_size=float(scatter_extra['mean_point_size']),
                    point_size=float(scatter_extra['point_size']),
                    alpha=float(scatter_extra['alpha']),
                    title=plot_title,
                    employee_colors=scatter_extra['employee_colors'],
                    comp_wrap_width=int(scatter_extra['comp_wrap_width']),
                    comp_max_lines=int(scatter_extra['comp_max_lines']),
                    show_employee_legend=bool(scatter_show_employee_legend),
                    show_mean_value_label=bool(scatter_show_mean_value_label),
                    title_fontsize=float(scatter_extra['title_fontsize']),
                    tick_fontsize=float(scatter_extra['tick_fontsize']),
                    legend_fontsize=float(scatter_extra['legend_fontsize']),
                    value_fontsize=float(scatter_extra['value_fontsize']),
                    employee_name_format=scatter_name_mode,
                )

                st.session_state.plot_fig_bytes = fig_to_png_bytes(fig)
                st.session_state.plot_stats_df = scatter_stats_df
                st.session_state.plot_done = True
                st.session_state.plot_error = None
                st.session_state.plot_zip_bytes = zip_bytes
                st.session_state.plot_zip_name = zip_name
                plt.close(fig)
            except Exception as e:
                clear_plot_state()
                st.session_state.plot_error = f'Ошибка построения графика: {e}'

elif chart_name == 'Столбчатая диаграмма':
    bar_dimension_registry: dict[str, dict[str, Any]] = dict(hist_dimension_registry)
    if 'Сотрудник' in prepared_df.columns:
        bar_dimension_registry['Сотрудник'] = {
            'col': 'Сотрудник',
            'values': sorted_unique_values(prepared_df['Сотрудник']),
            'label': 'Сотрудник',
        }

    bp1, bp2 = st.columns(2)
    with bp1:
        bar_dimension_name = st.selectbox(
            'Характеристика для вывода',
            options=list(bar_dimension_registry.keys()),
            index=0,
            key='bar_dimension_name',
        )
        bar_dimension_meta = bar_dimension_registry[bar_dimension_name]
        bar_category_col = bar_dimension_meta['col']
        bar_category_label = bar_dimension_meta['label']
        bar_category_options = bar_dimension_meta['values']
        bar_default_values = bar_category_options[: min(8, len(bar_category_options))] if bar_category_col == 'Компетенция' else bar_category_options
        bar_category_values = st.multiselect(
            f'{bar_category_label}: порядок вывода',
            options=bar_category_options,
            default=bar_default_values,
            key='bar_category_values',
        )

        bar_facet_candidates = ['Без разбивки'] + [name for name in bar_dimension_registry.keys() if bar_dimension_registry[name]['col'] != bar_category_col]
        bar_facet_name = st.selectbox(
            'Дополнительная разбивка на графики',
            options=bar_facet_candidates,
            index=0,
            key='bar_facet_name',
        )
        if bar_facet_name == 'Без разбивки':
            bar_facet_col = None
            bar_facet_label = None
            bar_facet_values = None
        else:
            bar_facet_meta = bar_dimension_registry[bar_facet_name]
            bar_facet_col = bar_facet_meta['col']
            bar_facet_label = bar_facet_meta['label']
            bar_facet_values = st.multiselect(
                f'{bar_facet_label}: порядок графиков',
                options=bar_facet_meta['values'],
                default=bar_facet_meta['values'],
                key='bar_facet_values',
            )

        bar_filters: dict[str, Any] = {}
        with st.expander('Фильтры', expanded=True):
            for dim_name, dim_meta in bar_dimension_registry.items():
                dim_col = dim_meta['col']
                if dim_col in {bar_category_col, bar_facet_col}:
                    continue
                options = dim_meta['values']
                selected = st.multiselect(
                    f'{dim_meta["label"]}',
                    options=options,
                    default=options,
                    format_func=lambda x: str(x),
                    key=f'bar_filter::{dim_col}',
                )
                if 0 < len(selected) < len(options):
                    bar_filters[dim_col] = selected

        st.caption('После фильтрации столбчатая диаграмма агрегирует данные до уровня: сотрудник × выбранная характеристика. Если задана дополнительная разбивка — графики делятся по ее значениям.')
        if bar_facet_col is not None:
            st.caption('Если по дополнительной разбивке получается больше 5 графиков, на экране показываются первые 5, а полный набор можно скачать ZIP-архивом.')
        bar_agg_label = st.selectbox(
            'Способ агрегации повторных строк сотрудника',
            ['Прямая', 'Последовательная'],
            index=0,
            key='bar_agg_label',
        )
        bar_agg = {'Прямая': 'weighted', 'Последовательная': 'mean'}[bar_agg_label]
        bar_stat_label = st.selectbox('Статистика', ['Среднее', 'Медиана'], index=0, key='bar_stat_label')
        bar_stat = {'Среднее': 'mean', 'Медиана': 'median'}[bar_stat_label]
        bar_sort_label = st.selectbox(
            'Сортировка',
            ['Как в списке', 'По значению ↓', 'По значению ↑', 'По названию'],
            index=0,
            key='bar_sort_label',
        )
        bar_sort_by = {
            'Как в списке': 'order',
            'По значению ↓': 'stat_desc',
            'По значению ↑': 'stat_asc',
            'По названию': 'name',
        }[bar_sort_label]
        bar_show_stat = st.checkbox('Показывать значение статистики', value=True, key='bar_show_stat')
        bar_show_minmax_label = st.selectbox(
            'Минимум / максимум',
            ['Не показывать', 'Усы', 'Подписи', 'Усы и подписи'],
            index=0,
            key='bar_show_minmax_label',
        )
        bar_show_minmax = {
            'Не показывать': None,
            'Усы': 'errorbar',
            'Подписи': 'text',
            'Усы и подписи': 'both',
        }[bar_show_minmax_label]

    with bp2:
        bar_title = st.text_input('Заголовок', value='Столбчатая диаграмма', key='bar_title')
        bar_grid = st.checkbox('Сетка', value=False, key='bar_grid')
        bar_x_tick_step_raw = st.number_input('Шаг тиков по оси X', min_value=0.0, value=0.0, step=0.1, key='bar_x_tick_step')
        if bar_category_col == 'Сотрудник' or bar_facet_col == 'Сотрудник':
            bar_employee_name_mode = st.selectbox(
                'Формат вывода сотрудников',
                ['Не менять', 'Первое слово', 'Второе слово', 'Третье слово'],
                index=0,
                key='bar_employee_name_mode',
            )
        else:
            bar_employee_name_mode = 'Не менять'

    bar_extra: dict[str, Any] = {}
    with st.expander('Дополнительные параметры'):
        be1, be2 = st.columns(2)
        with be1:
            st.markdown('**Размер картинки, см**')
            bs1, bs2, bs3 = st.columns(3)
            with bs1:
                bar_extra['fig_width_cm'] = st.number_input('Ширина', min_value=5.0, value=27.0, step=0.1, key='bar_fig_width_cm')
            with bs2:
                bar_extra['height_per_category_cm'] = st.number_input('Высота на категорию', min_value=0.3, value=1.5, step=0.1, key='bar_height_per_category_cm')
            with bs3:
                bar_category_count = len(bar_category_values)
                bar_facet_count = len(bar_facet_values) if bar_facet_values else 1
                total_height_cm = max(10.2, (float(st.session_state.get('bar_height_per_category_cm', 1.5)) * max(bar_category_count, 1) + 4.0) * max(bar_facet_count, 1))
                st.metric('Общая высота, см', f'{total_height_cm:.1f}')
            bar_extra['x_max_raw'] = st.number_input('Максимум оси X', min_value=0.0, value=0.0, step=0.1, key='bar_x_max')
            bar_extra['label_wrap_width'] = st.number_input('Ширина переноса названий', min_value=5, value=15, step=1, key='bar_label_wrap_width')
            bar_extra['label_max_lines'] = st.number_input('Максимум строк названия', min_value=1, value=3, step=1, key='bar_label_max_lines')
        with be2:
            bar_extra['err_color'] = st.color_picker('Цвет усов / min-max', value='#121111', key='bar_err_color')
            bar_extra['axis_line_color'] = st.color_picker('Цвет осевой линии', value='#BDBDBD', key='bar_axis_line_color')
            bar_extra['axis_line_width'] = st.number_input('Толщина осевой линии', min_value=0.1, value=3.0, step=0.1, key='bar_axis_line_width')
            bar_palette_mode = st.selectbox('Палитра баров', ['Стандартная', 'Своя'], index=0, key='bar_palette_mode')
            if bar_palette_mode == 'Своя':
                custom_bar_palette_cols = st.columns(4)
                default_bar_palette = ['#FF474A', '#30BAAD', '#606E7D', '#E0D9CE', '#495867', '#899AAB', '#34CCBE', '#B5A798', '#CBBFAD']
                bar_extra['custom_palette'] = []
                for i, default_color in enumerate(default_bar_palette):
                    with custom_bar_palette_cols[i % 4]:
                        bar_extra['custom_palette'].append(
                            st.color_picker(f'Цвет {i+1}', value=default_color, key=f'bar_custom_color_{i}')
                        )
            else:
                bar_extra['custom_palette'] = None

        st.markdown('**Размеры шрифтов**')
        bf1, bf2, bf3, bf4 = st.columns(4)
        with bf1:
            bar_extra['title_fontsize'] = st.number_input('Шрифт заголовка', min_value=6.0, value=13.0, step=1.0, key='bar_title_fontsize')
        with bf2:
            bar_extra['axis_label_fontsize'] = st.number_input('Шрифт названия оси', min_value=6.0, value=11.0, step=1.0, key='bar_axis_label_fontsize')
        with bf3:
            bar_extra['tick_label_fontsize'] = st.number_input('Шрифт тиков', min_value=6.0, value=10.0, step=1.0, key='bar_tick_label_fontsize')
        with bf4:
            bar_extra['value_fontsize'] = st.number_input('Шрифт значений', min_value=6.0, value=9.0, step=1.0, key='bar_value_fontsize')

    if st.button('Построить', width='stretch', key='plot_build_barplot'):
        if not bar_category_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {bar_category_label}.'
        else:
            try:
                filtered_bar_df = prepared_df.copy()
                for fcol, fvals in bar_filters.items():
                    if isinstance(fvals, (list, tuple, set)):
                        filtered_bar_df = filtered_bar_df[filtered_bar_df[fcol].isin(list(fvals))]
                    else:
                        filtered_bar_df = filtered_bar_df[filtered_bar_df[fcol] == fvals]
                available_category_values = [
                    x for x in bar_category_values
                    if str(x) in set(filtered_bar_df[bar_category_col].dropna().astype(str).tolist())
                ]
                if not available_category_values:
                    raise ValueError(f"Нет значений для '{bar_category_label}' после фильтров.")
                filtered_bar_df = filtered_bar_df[
                    filtered_bar_df[bar_category_col].astype(str).isin([str(x) for x in available_category_values])
                ].copy()
                if bar_facet_col is not None:
                    candidate_facets = bar_facet_values if bar_facet_values is not None else sorted_unique_values(filtered_bar_df[bar_facet_col])
                    available_facet_values = [x for x in candidate_facets if str(x) in set(filtered_bar_df[bar_facet_col].dropna().astype(str).tolist())]
                else:
                    available_facet_values = None

                bar_fig_width_inches = float(bar_extra['fig_width_cm']) / 2.54
                if bar_facet_col is not None:
                    facet_category_counts = []
                    for facet_value in available_facet_values or []:
                        facet_df = filtered_bar_df[filtered_bar_df[bar_facet_col].astype(str) == str(facet_value)].copy()
                        facet_category_counts.append(int(facet_df[bar_category_col].astype(str).nunique()))
                    effective_category_count = max(facet_category_counts) if facet_category_counts else len(available_category_values)
                else:
                    effective_category_count = len(available_category_values)
                if bar_category_col == 'Сотрудник' or bar_facet_col == 'Сотрудник':
                    per_facet_height_inches = max(2.2, (float(bar_extra['height_per_category_cm']) * max(effective_category_count, 1) + 1.4) / 2.54)
                else:
                    per_facet_height_inches = max(4.0, (float(bar_extra['height_per_category_cm']) * max(effective_category_count, 1) + 4.0) / 2.54)
                bar_x_tick_step = None if float(bar_x_tick_step_raw) <= 0 else float(bar_x_tick_step_raw)
                bar_x_max = None if float(bar_extra['x_max_raw']) <= 0 else float(bar_extra['x_max_raw'])
                zip_bytes = None
                zip_name = None
                display_facet_values = available_facet_values

                if bar_facet_col is not None and available_facet_values and len(available_facet_values) > 5:
                    display_facet_values = available_facet_values[:5]
                    fig_bytes_by_name: dict[str, bytes] = {}
                    for facet_value in available_facet_values:
                        facet_title = (bar_title.strip() or 'Столбчатая диаграмма') + f' — {bar_facet_label}: {facet_value}'
                        facet_fig, _, _, _ = plot_competency_barplot(
                            vis_df=prepared_df,
                            category_col=bar_category_col,
                            category_values=[str(x) for x in available_category_values],
                            category_label=bar_category_label,
                            facet_col=bar_facet_col,
                            facet_values=[str(facet_value)],
                            facet_label=bar_facet_label,
                            filters=bar_filters or None,
                            agg=bar_agg,
                            stat=bar_stat,
                            show_stat=bool(bar_show_stat),
                            show_minmax=bar_show_minmax,
                            figsize=(bar_fig_width_inches, per_facet_height_inches),
                            sort_by=bar_sort_by,
                            title=facet_title,
                            err_color=bar_extra['err_color'],
                            axis_line_color=bar_extra['axis_line_color'],
                            axis_line_width=float(bar_extra['axis_line_width']),
                            palette=(bar_extra['custom_palette'] if bar_extra.get('custom_palette') else None),
                            label_wrap_width=int(bar_extra['label_wrap_width']),
                            label_max_lines=int(bar_extra['label_max_lines']),
                            x_tick_step=bar_x_tick_step,
                            x_max=bar_x_max,
                            grid=bool(bar_grid),
                            title_fontsize=int(bar_extra['title_fontsize']),
                            axis_label_fontsize=int(bar_extra['axis_label_fontsize']),
                            tick_label_fontsize=int(bar_extra['tick_label_fontsize']),
                            value_fontsize=int(bar_extra['value_fontsize']),
                            employee_name_format={
                                'Не менять': 'keep',
                                'Первое слово': 'first',
                                'Второе слово': 'second',
                                'Третье слово': 'third',
                            }[bar_employee_name_mode],
                        )
                        facet_slug = str(facet_value).replace('/', '-').replace('\
', ' ').strip() or 'facet'
                        fig_bytes_by_name[f'bar_{facet_slug}.png'] = fig_to_png_bytes(facet_fig)
                        plt.close(facet_fig)
                    zip_bytes = build_png_zip(fig_bytes_by_name)
                    facet_slug = str(bar_facet_label or 'facet').strip().lower().replace(' ', '_') or 'facet'
                    zip_name = f'bar_facets_by_{facet_slug}.zip'

                total_height_inches = per_facet_height_inches * max(len(display_facet_values), 1) if display_facet_values else per_facet_height_inches
                fig, ax, summary_df, details_df = plot_competency_barplot(
                    vis_df=prepared_df,
                    category_col=bar_category_col,
                    category_values=[str(x) for x in available_category_values],
                    category_label=bar_category_label,
                    facet_col=bar_facet_col,
                    facet_values=[str(x) for x in display_facet_values] if display_facet_values is not None else None,
                    facet_label=bar_facet_label,
                    filters=bar_filters or None,
                    agg=bar_agg,
                    stat=bar_stat,
                    show_stat=bool(bar_show_stat),
                    show_minmax=bar_show_minmax,
                    figsize=(bar_fig_width_inches, total_height_inches),
                    sort_by=bar_sort_by,
                    title=bar_title.strip() or None,
                    err_color=bar_extra['err_color'],
                    axis_line_color=bar_extra['axis_line_color'],
                    axis_line_width=float(bar_extra['axis_line_width']),
                    palette=(bar_extra['custom_palette'] if bar_extra.get('custom_palette') else None),
                    label_wrap_width=int(bar_extra['label_wrap_width']),
                    label_max_lines=int(bar_extra['label_max_lines']),
                    x_tick_step=bar_x_tick_step,
                    x_max=bar_x_max,
                    grid=bool(bar_grid),
                    title_fontsize=int(bar_extra['title_fontsize']),
                    axis_label_fontsize=int(bar_extra['axis_label_fontsize']),
                    tick_label_fontsize=int(bar_extra['tick_label_fontsize']),
                    value_fontsize=int(bar_extra['value_fontsize']),
                    employee_name_format={
                        'Не менять': 'keep',
                        'Первое слово': 'first',
                        'Второе слово': 'second',
                        'Третье слово': 'third',
                    }[bar_employee_name_mode],
                )

                st.session_state.plot_fig_bytes = fig_to_png_bytes(fig)
                st.session_state.plot_stats_df = summary_df.copy()
                st.session_state.plot_done = True
                st.session_state.plot_error = None
                st.session_state.plot_zip_bytes = zip_bytes
                st.session_state.plot_zip_name = zip_name
                plt.close(fig)
            except Exception as e:
                clear_plot_state()
                st.session_state.plot_error = f'Ошибка построения графика: {e}'


elif chart_name == '9-box':
    ninebox_dimension_registry: dict[str, dict[str, Any]] = {}
    service_cols = {'score', 'n', 'scale_min', 'scale_max'}
    ordered_candidates = []
    if 'Сотрудник' in prepared_df.columns:
        ordered_candidates.append('Сотрудник')
    if has_role:
        ordered_candidates.append('роль')
    if 'Компетенция' in prepared_df.columns:
        ordered_candidates.append('Компетенция')
    ordered_candidates.extend([c for c in plot_group_cols if c in prepared_df.columns])
    ordered_candidates.extend([c for c in prepared_df.columns if c not in service_cols and c not in ordered_candidates])
    for col in ordered_candidates:
        if col in service_cols or col not in prepared_df.columns:
            continue
        label = 'Роль' if col == 'роль' else str(col)
        ninebox_dimension_registry[label] = {
            'col': col,
            'values': sorted_unique_values(prepared_df[col]),
            'label': label,
        }

    point_mode_options = ['Сотрудник', 'Компетенция'] + (['Роль'] if has_role else [])
    point_mode_to_col = {'Сотрудник': 'Сотрудник', 'Компетенция': 'Компетенция', 'Роль': 'роль'}

    nb1, nb2 = st.columns(2)
    with nb1:
        ninebox_point_mode = st.selectbox(
            'Построение по',
            options=point_mode_options,
            index=0,
            key='ninebox_point_mode',
        )
        ninebox_point_col = point_mode_to_col[ninebox_point_mode]
        if ninebox_point_mode == 'Сотрудник':
            axis_excluded_cols = {'Сотрудник', *[c for c in plot_group_cols if c in prepared_df.columns]}
        elif ninebox_point_mode == 'Компетенция':
            axis_excluded_cols = {'Компетенция'}
        else:
            axis_excluded_cols = {'роль'}

        ninebox_axis_registry = {
            name: meta for name, meta in ninebox_dimension_registry.items()
            if meta['col'] not in axis_excluded_cols
        }
        if not ninebox_axis_registry:
            st.error('Нет доступных колонок для осей 9-box при выбранном режиме построения.')
            st.stop()

        axis_names = list(ninebox_axis_registry.keys())
        ninebox_x_name = st.selectbox(
            'Характеристика для оси X',
            options=axis_names,
            index=0,
            key='ninebox_x_name',
        )
        ninebox_x_meta = ninebox_axis_registry[ninebox_x_name]
        ninebox_x_col = ninebox_x_meta['col']
        ninebox_x_label = ninebox_x_meta['label']
        ninebox_x_options = ninebox_x_meta['values']
        ninebox_x_values = st.multiselect(
            f'{ninebox_x_label}: значения для оси X',
            options=ninebox_x_options,
            default=ninebox_x_options[:min(3, len(ninebox_x_options))],
            key='ninebox_x_values',
        )

        default_y_index = 1 if len(axis_names) > 1 else 0
        ninebox_y_name = st.selectbox(
            'Характеристика для оси Y',
            options=axis_names,
            index=default_y_index,
            key='ninebox_y_name',
        )
        ninebox_y_meta = ninebox_axis_registry[ninebox_y_name]
        ninebox_y_col = ninebox_y_meta['col']
        ninebox_y_label = ninebox_y_meta['label']
        ninebox_y_options = ninebox_y_meta['values']
        ninebox_y_values = st.multiselect(
            f'{ninebox_y_label}: значения для оси Y',
            options=ninebox_y_options,
            default=ninebox_y_options[:min(3, len(ninebox_y_options))],
            key='ninebox_y_values',
        )
        ninebox_x_title = st.text_input('Название оси X', value='', key='ninebox_x_title')
        ninebox_y_title = st.text_input('Название оси Y', value='', key='ninebox_y_title')

        ninebox_facet_candidates = ['Без разбивки'] + [
            name for name, meta in ninebox_dimension_registry.items()
            if meta['col'] not in {ninebox_x_col, ninebox_y_col}
        ]
        ninebox_facet_name = st.selectbox(
            'Дополнительная разбивка на графики',
            options=ninebox_facet_candidates,
            index=0,
            key='ninebox_facet_name',
        )
        if ninebox_facet_name == 'Без разбивки':
            ninebox_facet_col = None
            ninebox_facet_label = None
            ninebox_facet_values = None
        else:
            ninebox_facet_meta = ninebox_dimension_registry[ninebox_facet_name]
            ninebox_facet_col = ninebox_facet_meta['col']
            ninebox_facet_label = ninebox_facet_meta['label']
            ninebox_facet_values = st.multiselect(
                f'{ninebox_facet_label}: порядок графиков',
                options=ninebox_facet_meta['values'],
                default=ninebox_facet_meta['values'],
                key='ninebox_facet_values',
            )

        ninebox_filters: dict[str, Any] = {}
        with st.expander('Фильтры', expanded=True):
            for dim_name, dim_meta in ninebox_dimension_registry.items():
                dim_col = dim_meta['col']
                if dim_col in {ninebox_x_col, ninebox_y_col, ninebox_facet_col}:
                    continue
                options = dim_meta['values']
                selected = st.multiselect(
                    dim_meta['label'],
                    options=options,
                    default=options,
                    key=f'ninebox_filter::{dim_col}',
                )
                if 0 < len(selected) < len(options):
                    ninebox_filters[dim_col] = selected

        if ninebox_facet_col is not None:
            st.caption('Если по дополнительной разбивке получается больше 5 графиков, на экране показываются первые 5, а полный набор можно скачать ZIP-архивом.')

    with nb2:
        with st.container(border=True):
            st.markdown('**Границы шкалы**')
            nb_s1, nb_s2 = st.columns(2)
            with nb_s1:
                ninebox_vmin = st.number_input('Минимум шкалы', value=float(hist_default_vmin) if hist_default_vmin is not None else 0.0, step=0.1, key='ninebox_vmin')
            with nb_s2:
                ninebox_vmax = st.number_input('Максимум шкалы', value=float(hist_default_vmax) if hist_default_vmax is not None else 0.0, step=0.1, key='ninebox_vmax')
        ninebox_cutpoint_method_label = st.selectbox('Способ расчета отсечек', ['Percentile span', 'Фиксированные'], index=0, key='ninebox_cutpoint_method_label')
        ninebox_cutpoint_method = 'percent' if ninebox_cutpoint_method_label == 'Percentile span' else 'fixed'
        ninebox_cutpoint_scope_label = st.selectbox('База для отсечек', ['По отфильтрованным данным', 'По всем данным'], index=1, key='ninebox_cutpoint_scope_label')
        ninebox_cutpoint_scope = 'filtered_data' if ninebox_cutpoint_scope_label == 'По отфильтрованным данным' else 'all_data'
        ninebox_percent_span = st.number_input('Span от медианы, %', min_value=1.0, max_value=49.0, value=25.0, step=1.0, key='ninebox_percent_span')
        if ninebox_cutpoint_method == 'fixed':
            fx1, fx2 = st.columns(2)
            with fx1:
                ninebox_x_t1 = st.number_input('X: нижняя отсечка', value=0.0, step=0.1, key='ninebox_x_t1')
                ninebox_y_t1 = st.number_input('Y: нижняя отсечка', value=0.0, step=0.1, key='ninebox_y_t1')
            with fx2:
                ninebox_x_t2 = st.number_input('X: верхняя отсечка', value=0.0, step=0.1, key='ninebox_x_t2')
                ninebox_y_t2 = st.number_input('Y: верхняя отсечка', value=0.0, step=0.1, key='ninebox_y_t2')
        else:
            ninebox_x_t1 = ninebox_x_t2 = ninebox_y_t1 = ninebox_y_t2 = 0.0
        ninebox_show_cell_counts = st.checkbox('Показывать количество в ячейках', value=True, key='ninebox_show_cell_counts')
        ninebox_label_cells = st.multiselect(
            'Подписывать точки в ячейках',
            options=['0_0', '0_1', '0_2', '1_0', '1_1', '1_2', '2_0', '2_1', '2_2'],
            default=[],
            key='ninebox_label_cells',
        )
        ninebox_title = st.text_input('Подзаголовок', value='', key='ninebox_title')

    ninebox_extra: dict[str, Any] = {}
    with st.expander('Дополнительные параметры'):
        nbe1, nbe2 = st.columns(2)
        with nbe1:
            st.markdown('**Размер картинки, см**')
            nbf1, nbf2 = st.columns(2)
            with nbf1:
                ninebox_extra['fig_width_cm'] = st.number_input('Ширина', min_value=5.0, value=24.0, step=0.1, key='ninebox_fig_width_cm')
            with nbf2:
                ninebox_extra['fig_height_cm'] = st.number_input('Высота одного графика', min_value=5.0, value=20.0, step=0.1, key='ninebox_fig_height_cm')
            ninebox_extra['point_size'] = st.number_input('Размер точек', min_value=1.0, value=65.0, step=1.0, key='ninebox_point_size')
            ninebox_extra['point_alpha'] = st.number_input('Прозрачность точек', min_value=0.0, max_value=1.0, value=0.85, step=0.05, key='ninebox_point_alpha')
        with nbe2:
            ninebox_extra['point_color'] = st.color_picker('Цвет точек', value='#121111', key='ninebox_point_color')
            ninebox_extra['label_fontsize'] = st.number_input('Шрифт подписей точек', min_value=6, value=8, step=1, key='ninebox_label_fontsize')
            ninebox_label_mode = st.selectbox(
                'Формат подписей точек',
                ['Без изменений', 'Сократить до 5 слов', 'Сократить до 3 слов', 'Первое слово и инициалы', 'Третье слово и инициалы'],
                index=0,
                key='ninebox_label_mode',
            )
            ninebox_extra['employee_name_format'] = {
                'Без изменений': 'keep',
                'Сократить до 5 слов': 'first5',
                'Сократить до 3 слов': 'first3',
                'Первое слово и инициалы': 'first_initials',
                'Третье слово и инициалы': 'third_initials',
            }[ninebox_label_mode]
            ninebox_extra['axis_tick_mode'] = st.selectbox('Подписи тиков осей', ['cutpoints', 'percentiles', 'none'], index=0, key='ninebox_axis_tick_mode')
        st.markdown('**Размеры шрифтов**')
        nbff1, nbff2, nbff3, nbff4 = st.columns(4)
        with nbff1:
            ninebox_extra['title_fontsize'] = st.number_input('Шрифт заголовка', min_value=6.0, value=13.0, step=1.0, key='ninebox_title_fontsize')
        with nbff2:
            ninebox_extra['axis_label_fontsize'] = st.number_input('Шрифт осей', min_value=6.0, value=11.0, step=1.0, key='ninebox_axis_label_fontsize')
        with nbff3:
            ninebox_extra['axis_tick_fontsize'] = st.number_input('Шрифт тиков', min_value=6.0, value=10.0, step=1.0, key='ninebox_axis_tick_fontsize')
        with nbff4:
            ninebox_extra['legend_fontsize'] = st.number_input('Шрифт легенды', min_value=6.0, value=9.0, step=1.0, key='ninebox_legend_fontsize')
        ninebox_extra['legend_position'] = st.selectbox('Размещение легенды', ['Вверху', 'Внизу', 'Отсутствует'], index=0, key='ninebox_legend_position')

        st.markdown('**Цвета и подписи 9 квадратов**')
        default_zone_cell_colors = {
            '0_2': '#E0D9CE', '1_2': '#30BAAD', '2_2': '#30BAAD',
            '0_1': '#FF474A', '1_1': '#E0D9CE', '2_1': '#30BAAD',
            '0_0': '#FF474A', '1_0': '#FF474A', '2_0': '#E0D9CE',
        }
        default_zone_cell_labels = {
            '0_2': 'Низкий-высокий', '1_2': 'Средний-высокий', '2_2': 'Высокий-высокий',
            '0_1': 'Низкий-средний', '1_1': 'Средний-средний', '2_1': 'Высокий-средний',
            '0_0': 'Низкий-низкий', '1_0': 'Средний-низкий', '2_0': 'Высокий-низкий',
        }
        ninebox_extra['zone_cell_colors'] = {}
        ninebox_extra['zone_cell_labels'] = {}
        zone_rows = [('0_2', '1_2', '2_2'), ('0_1', '1_1', '2_1'), ('0_0', '1_0', '2_0')]
        for row in zone_rows:
            zc1, zc2, zc3 = st.columns(3)
            for cell_key, zcol in zip(row, [zc1, zc2, zc3]):
                with zcol:
                    st.caption(f'Квадрат {cell_key}')
                    ninebox_extra['zone_cell_colors'][cell_key] = st.color_picker(
                        f'Цвет {cell_key}', value=default_zone_cell_colors[cell_key], key=f'ninebox_zone_cell_color_{cell_key}',
                    )
                    ninebox_extra['zone_cell_labels'][cell_key] = st.text_input(
                        f'Подпись {cell_key}', value=default_zone_cell_labels[cell_key], key=f'ninebox_zone_cell_label_{cell_key}',
                    )

    if st.button('Построить', width='stretch', key='plot_build_ninebox'):
        if not ninebox_x_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {ninebox_x_label}.'
        elif not ninebox_y_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {ninebox_y_label}.'
        elif ninebox_facet_col is not None and not ninebox_facet_values:
            clear_plot_state()
            st.session_state.plot_error = f'Выбери хотя бы одно значение для: {ninebox_facet_label}.'
        else:
            try:
                if float(ninebox_vmax) <= float(ninebox_vmin):
                    raise ValueError('Заполни корректные границы шкалы: максимум должен быть больше минимума.')
                if ninebox_cutpoint_method == 'fixed':
                    if not (float(ninebox_vmin) <= float(ninebox_x_t1) <= float(ninebox_x_t2) <= float(ninebox_vmax)):
                        raise ValueError('Для оси X отсечки должны удовлетворять vmin <= t1 <= t2 <= vmax.')
                    if not (float(ninebox_vmin) <= float(ninebox_y_t1) <= float(ninebox_y_t2) <= float(ninebox_vmax)):
                        raise ValueError('Для оси Y отсечки должны удовлетворять vmin <= t1 <= t2 <= vmax.')

                fig_width_inches = float(ninebox_extra['fig_width_cm']) / 2.54
                fig_height_inches = float(ninebox_extra['fig_height_cm']) / 2.54

                facet_candidates = None
                if ninebox_facet_col is not None:
                    facet_candidates = get_available_values_after_filters(
                        prepared_df,
                        target_col=ninebox_facet_col,
                        filters=ninebox_filters,
                        target_values=ninebox_facet_values,
                    )

                zip_bytes = None
                zip_name = None
                display_facet_values = facet_candidates
                if ninebox_facet_col is not None and facet_candidates:
                    valid_facet_values = []
                    pngs: dict[str, bytes] = {}
                    for facet_value in facet_candidates:
                        try:
                            one_fig, _, one_points_df, _, _ = plot_9box(
                                df=prepared_df,
                                point_mode=('employee' if ninebox_point_mode == 'Сотрудник' else ('competency' if ninebox_point_mode == 'Компетенция' else 'role')),
                                point_col=ninebox_point_col,
                                x_col=ninebox_x_col,
                                x_values=[str(x) for x in ninebox_x_values],
                                y_col=ninebox_y_col,
                                y_values=[str(x) for x in ninebox_y_values],
                                vmin=float(ninebox_vmin),
                                vmax=float(ninebox_vmax),
                                filters=ninebox_filters or None,
                                facet_col=ninebox_facet_col,
                                facet_values=[str(facet_value)],
                                cutpoint_method=ninebox_cutpoint_method,
                                cutpoint_scope=ninebox_cutpoint_scope,
                                percent_span=float(ninebox_percent_span),
                                cutpoints={'x': (float(ninebox_x_t1), float(ninebox_x_t2)), 'y': (float(ninebox_y_t1), float(ninebox_y_t2))} if ninebox_cutpoint_method == 'fixed' else None,
                                show_cell_counts=bool(ninebox_show_cell_counts),
                                label_cells=list(ninebox_label_cells),
                                label_fontsize=int(ninebox_extra['label_fontsize']),
                                point_size=int(float(ninebox_extra['point_size'])),
                                point_alpha=float(ninebox_extra['point_alpha']),
                                point_color=ninebox_extra['point_color'],
                                zone_cell_colors=ninebox_extra['zone_cell_colors'],
                                zone_cell_labels=ninebox_extra['zone_cell_labels'],
                                figsize=(fig_width_inches, fig_height_inches),
                                title=str(facet_value),
                                x_title=(ninebox_x_title.strip() or None),
                                legend_position={'Вверху': 'top', 'Внизу': 'bottom', 'Отсутствует': 'none'}[ninebox_extra['legend_position']],
                                y_title=(ninebox_y_title.strip() or None),
                                axis_label_fontsize=int(float(ninebox_extra['axis_label_fontsize'])),
                                axis_tick_fontsize=int(float(ninebox_extra['axis_tick_fontsize'])),
                                legend_fontsize=int(float(ninebox_extra['legend_fontsize'])),
                                title_fontsize=int(float(ninebox_extra['title_fontsize'])),
                                axis_tick_mode=str(ninebox_extra['axis_tick_mode']),
                                employee_name_format=str(ninebox_extra['employee_name_format']),
                            )
                        except Exception as e:
                            if 'Нет точек с данными одновременно по осям X и Y после фильтров.' in str(e):
                                continue
                            raise
                        if one_points_df is None or one_points_df.empty:
                            plt.close(one_fig)
                            continue
                        valid_facet_values.append(facet_value)
                        safe_name = str(facet_value).replace('/', '-').replace('\\', '-').strip() or 'facet'
                        pngs[f'{safe_name}.png'] = fig_to_png_bytes(one_fig)
                        plt.close(one_fig)
                    display_facet_values = valid_facet_values
                    if not valid_facet_values:
                        raise ValueError('Нет точек с данными одновременно по осям X и Y после фильтров.')
                    if len(valid_facet_values) > 5:
                        zip_bytes = build_png_zip(pngs)
                        zip_name = f'9box_facets_by_{str(ninebox_facet_label or "facet").strip().lower().replace(" ", "_")}.zip'
                        display_facet_values = valid_facet_values[:5]

                total_height_inches = fig_height_inches * max(len(display_facet_values), 1) if display_facet_values else fig_height_inches
                custom_title = (ninebox_title.strip() or None)
                fig, ax, points_df, cutpoints_used, cell_tables = plot_9box(
                    df=prepared_df,
                    point_mode=('employee' if ninebox_point_mode == 'Сотрудник' else ('competency' if ninebox_point_mode == 'Компетенция' else 'role')),
                    point_col=ninebox_point_col,
                    x_col=ninebox_x_col,
                    x_values=[str(x) for x in ninebox_x_values],
                    y_col=ninebox_y_col,
                    y_values=[str(x) for x in ninebox_y_values],
                    vmin=float(ninebox_vmin),
                    vmax=float(ninebox_vmax),
                    filters=ninebox_filters or None,
                    facet_col=ninebox_facet_col,
                    facet_values=[str(x) for x in display_facet_values] if display_facet_values is not None else None,
                    cutpoint_method=ninebox_cutpoint_method,
                    cutpoint_scope=ninebox_cutpoint_scope,
                    percent_span=float(ninebox_percent_span),
                    cutpoints={'x': (float(ninebox_x_t1), float(ninebox_x_t2)), 'y': (float(ninebox_y_t1), float(ninebox_y_t2))} if ninebox_cutpoint_method == 'fixed' else None,
                    show_cell_counts=bool(ninebox_show_cell_counts),
                    label_cells=list(ninebox_label_cells),
                    label_fontsize=int(ninebox_extra['label_fontsize']),
                    point_size=int(float(ninebox_extra['point_size'])),
                    point_alpha=float(ninebox_extra['point_alpha']),
                    point_color=ninebox_extra['point_color'],
                    zone_cell_colors=ninebox_extra['zone_cell_colors'],
                    zone_cell_labels=ninebox_extra['zone_cell_labels'],
                    figsize=(fig_width_inches, total_height_inches),
                    title=custom_title,
                    x_title=(ninebox_x_title.strip() or None),
                    legend_position={'Вверху': 'top', 'Внизу': 'bottom', 'Отсутствует': 'none'}[ninebox_extra['legend_position']],
                    y_title=(ninebox_y_title.strip() or None),
                    axis_label_fontsize=int(float(ninebox_extra['axis_label_fontsize'])),
                    axis_tick_fontsize=int(float(ninebox_extra['axis_tick_fontsize'])),
                    legend_fontsize=int(float(ninebox_extra['legend_fontsize'])),
                    title_fontsize=int(float(ninebox_extra['title_fontsize'])),
                    axis_tick_mode=str(ninebox_extra['axis_tick_mode']),
                    employee_name_format=str(ninebox_extra['employee_name_format']),
                )
                clear_plot_state()
                st.session_state.plot_done = True
                st.session_state.plot_fig_bytes = fig_to_png_bytes(fig)
                st.session_state.plot_stats_df = points_df
                st.session_state.plot_zip_bytes = zip_bytes
                st.session_state.plot_zip_name = zip_name
                plt.close(fig)
            except Exception as e:
                clear_plot_state()
                st.session_state.plot_error = f'Ошибка построения графика: {e}'


elif chart_name == 'Тепловая карта':
    heatmap_dimension_registry: dict[str, dict[str, Any]] = dict(hist_dimension_registry)
    if 'Сотрудник' in prepared_df.columns:
        heatmap_dimension_registry['Сотрудник'] = {
            'col': 'Сотрудник',
            'values': sorted_unique_values(prepared_df['Сотрудник']),
            'label': 'Сотрудник',
        }

    hm1, hm2 = st.columns(2)
    with hm1:
        heatmap_x_name = st.selectbox(
            'Характеристика для оси X',
            options=list(heatmap_dimension_registry.keys()),
            index=0,
            key='heatmap_x_name',
        )
        heatmap_x_meta = heatmap_dimension_registry[heatmap_x_name]
        heatmap_x_col = heatmap_x_meta['col']
        heatmap_x_label = heatmap_x_meta['label']
        heatmap_x_options = heatmap_x_meta['values']
        heatmap_x_values = st.multiselect(
            f'{heatmap_x_label}: порядок вывода',
            options=heatmap_x_options,
            default=heatmap_x_options[:min(10, len(heatmap_x_options))],
            key='heatmap_x_values',
        )
        heatmap_x_limit = st.number_input(
            'Показывать первых значений по оси X (0 = все)',
            min_value=0,
            value=10,
            step=1,
            key='heatmap_x_limit',
        )

        heatmap_y1_candidates = [name for name in heatmap_dimension_registry.keys() if heatmap_dimension_registry[name]['col'] != heatmap_x_col]
        heatmap_y1_name = st.selectbox(
            'Характеристика для оси Y',
            options=heatmap_y1_candidates,
            index=0,
            key='heatmap_y1_name',
        )
        heatmap_y1_meta = heatmap_dimension_registry[heatmap_y1_name]
        heatmap_y1_col = heatmap_y1_meta['col']
        heatmap_y1_values = st.multiselect(
            f'{heatmap_y1_meta["label"]}: порядок вывода',
            options=heatmap_y1_meta['values'],
            default=heatmap_y1_meta['values'][:min(10, len(heatmap_y1_meta['values']))],
            key='heatmap_y1_values',
        )
        heatmap_y1_limit = st.number_input(
            'Показывать первых значений по оси Y (0 = все)',
            min_value=0,
            value=10,
            step=1,
            key='heatmap_y1_limit',
        )

        heatmap_use_y2 = st.checkbox('Использовать второй уровень оси Y', value=False, key='heatmap_use_y2')
        heatmap_y_cols = [heatmap_y1_col]
        heatmap_y_values_map = {heatmap_y1_col: heatmap_y1_values}
        if heatmap_use_y2:
            heatmap_y2_candidates = [name for name in heatmap_dimension_registry.keys() if heatmap_dimension_registry[name]['col'] not in {heatmap_x_col, heatmap_y1_col}]
            heatmap_y2_name = st.selectbox(
                'Вторая характеристика для оси Y',
                options=heatmap_y2_candidates,
                index=0,
                key='heatmap_y2_name',
            )
            heatmap_y2_meta = heatmap_dimension_registry[heatmap_y2_name]
            heatmap_y2_col = heatmap_y2_meta['col']
            heatmap_y2_values = st.multiselect(
                f'{heatmap_y2_meta["label"]}: порядок вывода',
                options=heatmap_y2_meta['values'],
                default=heatmap_y2_meta['values'][:min(10, len(heatmap_y2_meta['values']))],
                key='heatmap_y2_values',
            )
            heatmap_y2_limit = st.number_input(
                'Показывать первых значений по второму уровню Y (0 = все)',
                min_value=0,
                value=10,
                step=1,
                key='heatmap_y2_limit',
            )
            heatmap_y_cols.append(heatmap_y2_col)
            heatmap_y_values_map[heatmap_y2_col] = heatmap_y2_values
        else:
            heatmap_y2_col = None
            heatmap_y2_limit = 10

        heatmap_facet_candidates = ['Без разбивки'] + [name for name in heatmap_dimension_registry.keys() if heatmap_dimension_registry[name]['col'] not in set([heatmap_x_col] + heatmap_y_cols)]
        heatmap_facet_name = st.selectbox(
            'Дополнительная разбивка на графики',
            options=heatmap_facet_candidates,
            index=0,
            key='heatmap_facet_name',
        )
        if heatmap_facet_name == 'Без разбивки':
            heatmap_facet_col = None
            heatmap_facet_label = None
            heatmap_facet_values = None
        else:
            heatmap_facet_meta = heatmap_dimension_registry[heatmap_facet_name]
            heatmap_facet_col = heatmap_facet_meta['col']
            heatmap_facet_label = heatmap_facet_meta['label']
            heatmap_facet_values = st.multiselect(
                f'{heatmap_facet_label}: порядок графиков',
                options=heatmap_facet_meta['values'],
                default=heatmap_facet_meta['values'],
                key='heatmap_facet_values',
            )

        heatmap_filters: dict[str, Any] = {}
        with st.expander('Фильтры', expanded=True):
            for dim_name, dim_meta in heatmap_dimension_registry.items():
                dim_col = dim_meta['col']
                if dim_col in set([heatmap_x_col] + heatmap_y_cols + ([heatmap_facet_col] if heatmap_facet_col else [])):
                    continue
                selected = st.multiselect(
                    dim_meta['label'],
                    options=dim_meta['values'],
                    default=dim_meta['values'],
                    key=f'heatmap_filter::{dim_col}',
                )
                if 0 < len(selected) < len(dim_meta['values']):
                    heatmap_filters[dim_col] = selected

        if heatmap_facet_col is not None:
            st.caption('Если по дополнительной разбивке получается больше 5 графиков, на экране показываются первые 5, а полный набор можно скачать ZIP-архивом.')

    with hm2:
        heatmap_stat_label = st.selectbox('Статистика', ['Среднее', 'Медиана'], index=0, key='heatmap_stat_label')
        heatmap_stat = {'Среднее': 'mean', 'Медиана': 'median'}[heatmap_stat_label]
        heatmap_min_employees = st.number_input('Минимальное количество сотрудников', min_value=1, value=1, step=1, key='heatmap_min_employees')
        heatmap_sort_categories = st.selectbox('Сортировка X', ['order', 'name', 'overall_stat_desc', 'overall_stat_asc'], index=0, key='heatmap_sort_categories')
        heatmap_sort_groups = st.selectbox('Сортировка Y', ['order', 'name', 'overall_stat_desc', 'overall_stat_asc'], index=0, key='heatmap_sort_groups')
        heatmap_cutoff_mode = st.selectbox('Режим отсечек', ['per_column', 'global'], index=0, key='heatmap_cutoff_mode')
        heatmap_percentile_span = st.number_input('Percentile span', min_value=0.0, max_value=50.0, value=25.0, step=1.0, key='heatmap_percentile_span')
        heatmap_use_manual_cutoffs = st.checkbox('Задать low/high вручную', value=False, key='heatmap_use_manual_cutoffs')
        hc1, hc2 = st.columns(2)
        with hc1:
            heatmap_low_cutoff = st.number_input('Low cutoff', value=0.0, step=0.1, disabled=not heatmap_use_manual_cutoffs, key='heatmap_low_cutoff')
        with hc2:
            heatmap_high_cutoff = st.number_input('High cutoff', value=0.0, step=0.1, disabled=not heatmap_use_manual_cutoffs, key='heatmap_high_cutoff')
        heatmap_show_values = st.checkbox('Показывать значения в ячейках', value=True, key='heatmap_show_values')
        heatmap_round_digits = st.selectbox('Округление до', [0, 1, 2, 3], index=2, key='heatmap_round_digits')
        heatmap_title = st.text_input('Заголовок', value='', key='heatmap_title')
        heatmap_reference_enabled = st.checkbox('Установить референс по оси X', value=False, key='heatmap_reference_enabled')
        if heatmap_reference_enabled:
            heatmap_reference_values = st.multiselect(
                f'Значения {heatmap_x_label} для референса',
                options=heatmap_x_values if heatmap_x_values else heatmap_x_options,
                default=heatmap_x_values if heatmap_x_values else heatmap_x_options[:min(3, len(heatmap_x_options))],
                key='heatmap_reference_values',
            )
            heatmap_reference_label = st.text_input(
                'Название референсной колонки',
                value='Референс',
                key='heatmap_reference_label',
            )
            heatmap_reference_delta = st.number_input(
                'Порог отклонения от референса',
                min_value=0.0,
                value=float((hist_default_vmax or 0.0) * 0.1),
                step=0.1,
                key='heatmap_reference_delta',
            )
            heatmap_reference_arrow_fontsize = st.number_input(
                'Размер стрелок референса',
                min_value=6.0,
                value=11.0,
                step=1.0,
                key='heatmap_reference_arrow_fontsize',
            )
        else:
            heatmap_reference_values = []
            heatmap_reference_label = 'Референс'
            heatmap_reference_delta = float((hist_default_vmax or 0.0) * 0.1)
            heatmap_reference_arrow_fontsize = 11.0
        if heatmap_x_col == 'Сотрудник' or 'Сотрудник' in heatmap_y_cols or heatmap_facet_col == 'Сотрудник':
            heatmap_employee_name_mode = st.selectbox(
                'Формат вывода сотрудников',
                ['Не менять', 'Первое слово', 'Второе слово', 'Третье слово'],
                index=0,
                key='heatmap_employee_name_mode',
            )
        else:
            heatmap_employee_name_mode = 'Не менять'

    heatmap_extra = {}
    with st.expander('Дополнительные параметры'):
        hs1, hs2 = st.columns(2)
        with hs1:
            st.markdown('**Размер картинки, см**')
            hf1, hf2 = st.columns(2)
            with hf1:
                heatmap_extra['fig_width_cm'] = st.number_input('Ширина', min_value=5.0, value=24.0, step=0.1, key='heatmap_fig_width_cm')
            with hf2:
                heatmap_extra['fig_height_cm'] = st.number_input('Высота одного графика', min_value=5.0, value=16.0, step=0.1, key='heatmap_fig_height_cm')
            heatmap_extra['cell_fill'] = st.number_input('Заполнение ячейки', min_value=0.1, max_value=1.0, value=0.5, step=0.05, key='heatmap_cell_fill')
            heatmap_extra['highlight_shape'] = st.selectbox('Фигура подсветки', ['Прямоугольник', 'Круг'], index=0, key='heatmap_highlight_shape')
            heatmap_extra['grid'] = st.checkbox('Сетка', value=True, key='heatmap_grid')
            heatmap_extra['grid_color'] = st.color_picker('Цвет сетки', value='#E0D9CE', key='heatmap_grid_color')
            heatmap_extra['draw_level_separators'] = st.checkbox('Разделители уровней', value=True, key='heatmap_draw_level_separators')
            heatmap_extra['separator_color'] = st.color_picker('Цвет разделителей', value='#495867', key='heatmap_separator_color')
        with hs2:
            heatmap_extra['low_color'] = st.color_picker('Цвет low', value='#FF474A', key='heatmap_low_color')
            heatmap_extra['mid_color'] = st.color_picker('Цвет mid', value='#606E7D', key='heatmap_mid_color')
            heatmap_extra['high_color'] = st.color_picker('Цвет high', value='#30BAAD', key='heatmap_high_color')
            hl1, hl2, hl3 = st.columns(3)
            with hl1:
                heatmap_extra['low_label'] = st.text_input('Подпись зоны low', value='low', key='heatmap_low_label')
            with hl2:
                heatmap_extra['mid_label'] = st.text_input('Подпись зоны mid', value='mid', key='heatmap_mid_label')
            with hl3:
                heatmap_extra['high_label'] = st.text_input('Подпись зоны high', value='high', key='heatmap_high_label')
            heatmap_extra['x_label_rotation'] = st.number_input('Поворот подписей X', min_value=0.0, max_value=180.0, value=0.0, step=5.0, key='heatmap_x_label_rotation')
            heatmap_extra['x_label_ha'] = st.selectbox('Выравнивание подписей X', ['left', 'center', 'right'], index=1, key='heatmap_x_label_ha')
            heatmap_extra['x_label_wrap_width'] = st.number_input('Ширина переноса X', min_value=5, value=15, step=1, key='heatmap_x_label_wrap_width')
            heatmap_extra['x_label_max_lines'] = st.number_input('Макс. строк X', min_value=1, value=3, step=1, key='heatmap_x_label_max_lines')
            heatmap_extra['y_label_joiner'] = st.text_input('Разделитель уровней Y', value=' • ', key='heatmap_y_label_joiner')
        st.markdown('**Размеры шрифтов**')
        hmf1, hmf2, hmf3, hmf4, hmf5 = st.columns(5)
        with hmf1:
            heatmap_extra['title_fontsize'] = st.number_input('Шрифт заголовка', min_value=6.0, value=13.0, step=1.0, key='heatmap_title_fontsize')
        with hmf2:
            heatmap_extra['x_tick_label_fontsize'] = st.number_input('Шрифт X', min_value=6.0, value=10.0, step=1.0, key='heatmap_x_tick_label_fontsize')
        with hmf3:
            heatmap_extra['y_tick_label_fontsize'] = st.number_input('Шрифт Y', min_value=6.0, value=10.0, step=1.0, key='heatmap_y_tick_label_fontsize')
        with hmf4:
            heatmap_extra['legend_fontsize'] = st.number_input('Шрифт легенды', min_value=6.0, value=9.0, step=1.0, key='heatmap_legend_fontsize')
        with hmf5:
            heatmap_extra['value_text_fontsize'] = st.number_input('Шрифт значений', min_value=6.0, value=8.0, step=1.0, key='heatmap_value_text_fontsize')

        st.markdown('**Ручные отсечки для каждой категории X**')
        st.caption('Работают в режиме per_column для выбранных значений оси X.')
        heatmap_x_values_limited = list(heatmap_x_values[:int(heatmap_x_limit)]) if int(heatmap_x_limit) > 0 else list(heatmap_x_values)
        heatmap_y1_values_limited = list(heatmap_y1_values)
        heatmap_y_values_map_limited = {heatmap_y1_col: heatmap_y1_values_limited}
        heatmap_y_top_n_map = {heatmap_y1_col: int(heatmap_y1_limit)}
        if heatmap_use_y2:
            heatmap_y2_values_limited = list(heatmap_y2_values)
            heatmap_y_values_map_limited[heatmap_y2_col] = heatmap_y2_values_limited
            heatmap_y_top_n_map[heatmap_y2_col] = int(heatmap_y2_limit)
        else:
            heatmap_y2_values_limited = []

        heatmap_cutoffs_by_category = {}
        if heatmap_cutoff_mode == 'per_column':
            for cat in heatmap_x_values_limited:
                use_cat_cutoff = st.checkbox(f'{cat}', value=False, key=f'heatmap_cutoff_enabled::{cat}')
                cc1, cc2 = st.columns(2)
                with cc1:
                    cat_low = st.number_input(f'Low cutoff: {cat}', value=0.0, step=0.1, disabled=not use_cat_cutoff, key=f'heatmap_cutoff_low::{cat}')
                with cc2:
                    cat_high = st.number_input(f'High cutoff: {cat}', value=0.0, step=0.1, disabled=not use_cat_cutoff, key=f'heatmap_cutoff_high::{cat}')
                if use_cat_cutoff:
                    heatmap_cutoffs_by_category[str(cat)] = (float(cat_low), float(cat_high))

    if st.button('Построить', width='stretch', key='plot_build_heatmap'):
        try:
            if not heatmap_x_values_limited:
                raise ValueError(f'Выбери хотя бы одно значение для: {heatmap_x_label}.')
            if heatmap_reference_enabled and not heatmap_reference_values:
                raise ValueError('Выбери хотя бы одно значение по оси X для расчета референса.')
            if not heatmap_y1_values_limited:
                raise ValueError(f'Выбери хотя бы одно значение для: {heatmap_y1_meta["label"]}.')
            if heatmap_use_y2 and not heatmap_y_values_map_limited.get(heatmap_y2_col):
                raise ValueError(f'Выбери хотя бы одно значение для: {heatmap_y2_meta["label"]}.')
            heatmap_cutoffs = None
            if heatmap_use_manual_cutoffs:
                if float(heatmap_high_cutoff) < float(heatmap_low_cutoff):
                    raise ValueError('High cutoff должен быть не меньше low cutoff.')
                heatmap_cutoffs = (float(heatmap_low_cutoff), float(heatmap_high_cutoff))
            for cat_name, (cat_low, cat_high) in heatmap_cutoffs_by_category.items():
                if float(cat_high) < float(cat_low):
                    raise ValueError(f'Для категории "{cat_name}" high cutoff должен быть не меньше low cutoff.')
            if heatmap_cutoff_mode == 'per_column' and heatmap_cutoffs_by_category:
                heatmap_cutoffs = heatmap_cutoffs_by_category

            facet_candidates = None
            if heatmap_facet_col is not None:
                facet_candidates = get_available_values_after_filters(
                    prepared_df,
                    target_col=heatmap_facet_col,
                    filters=heatmap_filters,
                    target_values=heatmap_facet_values,
                )

            fig_width_inches = float(heatmap_extra['fig_width_cm']) / 2.54
            fig_height_inches = float(heatmap_extra['fig_height_cm']) / 2.54
            zip_bytes = None
            zip_name = None
            display_facet_values = facet_candidates
            if heatmap_facet_col is not None and facet_candidates and len(facet_candidates) > 5:
                display_facet_values = facet_candidates[:5]
                fig_bytes_by_name = {}
                for facet_value in facet_candidates:
                    facet_fig, _, _, _, _ = plot_group_heatmap(
                        vis_df=prepared_df,
                        x_col=heatmap_x_col,
                        x_values=[str(x) for x in heatmap_x_values_limited],
                        y_cols=heatmap_y_cols,
                        y_values_map={k: [str(v) for v in vals] for k, vals in heatmap_y_values_map_limited.items()},
                        facet_col=heatmap_facet_col,
                        facet_values=[str(facet_value)],
                        facet_label=heatmap_facet_label,
                        filters=heatmap_filters or None,
                        stat=heatmap_stat,
                        min_employees=int(heatmap_min_employees),
                        sort_categories=heatmap_sort_categories,
                        sort_groups=heatmap_sort_groups,
                        cutoff_mode=heatmap_cutoff_mode,
                        cutoffs=heatmap_cutoffs,
                        percentile_span=float(heatmap_percentile_span),
                        figsize=(fig_width_inches, fig_height_inches),
                        cell_fill=float(heatmap_extra['cell_fill']),
                        show_values=bool(heatmap_show_values),
                        value_fmt='{:.' + str(int(heatmap_round_digits)) + 'f}',
                        grid=bool(heatmap_extra['grid']),
                        grid_color=heatmap_extra['grid_color'],
                        x_label_rotation=float(heatmap_extra['x_label_rotation']),
                        x_label_ha=heatmap_extra['x_label_ha'],
                        x_label_wrap_width=int(heatmap_extra['x_label_wrap_width']),
                        x_label_max_lines=int(heatmap_extra['x_label_max_lines']),
                        y_label_joiner=heatmap_extra['y_label_joiner'],
                        title=str(facet_value),
                        title_fontsize=int(heatmap_extra['title_fontsize']),
                        x_tick_label_fontsize=int(heatmap_extra['x_tick_label_fontsize']),
                        y_tick_label_fontsize=int(heatmap_extra['y_tick_label_fontsize']),
                        legend_fontsize=int(heatmap_extra['legend_fontsize']),
                        value_text_fontsize=int(heatmap_extra['value_text_fontsize']),
                        draw_level_separators=bool(heatmap_extra['draw_level_separators']),
                        separator_color=heatmap_extra['separator_color'],
                        low_color=heatmap_extra['low_color'],
                        mid_color=heatmap_extra['mid_color'],
                        high_color=heatmap_extra['high_color'],
                        employee_name_format={'Не менять': 'keep', 'Первое слово': 'first', 'Второе слово': 'second', 'Третье слово': 'third'}[heatmap_employee_name_mode],
                        highlight_shape='circle' if heatmap_extra['highlight_shape'] == 'Круг' else 'rectangle',
                        zone_labels=(heatmap_extra['low_label'], heatmap_extra['mid_label'], heatmap_extra['high_label']),
                        y_top_n_map=heatmap_y_top_n_map,
                        reference_enabled=bool(heatmap_reference_enabled),
                        reference_values=[str(x) for x in heatmap_reference_values],
                        reference_label=str(heatmap_reference_label).strip() or 'Референс',
                        reference_delta=float(heatmap_reference_delta),
                        reference_arrow_fontsize=float(heatmap_reference_arrow_fontsize),
                    )
                    safe_name = str(facet_value).replace('/', '-').replace('\\', '-').strip() or 'facet'
                    fig_bytes_by_name[f'{safe_name}.png'] = fig_to_png_bytes(facet_fig)
                    plt.close(facet_fig)
                zip_bytes = build_png_zip(fig_bytes_by_name)
                zip_name = f'heatmap_facets_by_{str(heatmap_facet_label or "facet").strip().lower().replace(" ", "_")}.zip'

            total_height_inches = fig_height_inches * max(len(display_facet_values), 1) if display_facet_values else fig_height_inches
            fig, ax, summary_df, mat, resolved_cutoffs = plot_group_heatmap(
                vis_df=prepared_df,
                x_col=heatmap_x_col,
                x_values=[str(x) for x in heatmap_x_values_limited],
                y_cols=heatmap_y_cols,
                y_values_map={k: [str(v) for v in vals] for k, vals in heatmap_y_values_map_limited.items()},
                facet_col=heatmap_facet_col,
                facet_values=[str(x) for x in display_facet_values] if display_facet_values is not None else None,
                facet_label=heatmap_facet_label,
                filters=heatmap_filters or None,
                stat=heatmap_stat,
                min_employees=int(heatmap_min_employees),
                sort_categories=heatmap_sort_categories,
                sort_groups=heatmap_sort_groups,
                cutoff_mode=heatmap_cutoff_mode,
                cutoffs=heatmap_cutoffs,
                percentile_span=float(heatmap_percentile_span),
                figsize=(fig_width_inches, total_height_inches),
                cell_fill=float(heatmap_extra['cell_fill']),
                show_values=bool(heatmap_show_values),
                value_fmt='{:.' + str(int(heatmap_round_digits)) + 'f}',
                grid=bool(heatmap_extra['grid']),
                grid_color=heatmap_extra['grid_color'],
                x_label_rotation=float(heatmap_extra['x_label_rotation']),
                x_label_ha=heatmap_extra['x_label_ha'],
                x_label_wrap_width=int(heatmap_extra['x_label_wrap_width']),
                x_label_max_lines=int(heatmap_extra['x_label_max_lines']),
                y_label_joiner=heatmap_extra['y_label_joiner'],
                title=heatmap_title.strip() or None,
                title_fontsize=int(heatmap_extra['title_fontsize']),
                x_tick_label_fontsize=int(heatmap_extra['x_tick_label_fontsize']),
                y_tick_label_fontsize=int(heatmap_extra['y_tick_label_fontsize']),
                legend_fontsize=int(heatmap_extra['legend_fontsize']),
                value_text_fontsize=int(heatmap_extra['value_text_fontsize']),
                draw_level_separators=bool(heatmap_extra['draw_level_separators']),
                separator_color=heatmap_extra['separator_color'],
                low_color=heatmap_extra['low_color'],
                mid_color=heatmap_extra['mid_color'],
                high_color=heatmap_extra['high_color'],
                employee_name_format={'Не менять': 'keep', 'Первое слово': 'first', 'Второе слово': 'second', 'Третье слово': 'third'}[heatmap_employee_name_mode],
                highlight_shape='circle' if heatmap_extra['highlight_shape'] == 'Круг' else 'rectangle',
                zone_labels=(heatmap_extra['low_label'], heatmap_extra['mid_label'], heatmap_extra['high_label']),
                y_top_n_map=heatmap_y_top_n_map,
                reference_enabled=bool(heatmap_reference_enabled),
                reference_values=[str(x) for x in heatmap_reference_values],
                reference_label=str(heatmap_reference_label).strip() or 'Референс',
                reference_delta=float(heatmap_reference_delta),
                reference_arrow_fontsize=float(heatmap_reference_arrow_fontsize),
            )
            st.session_state.plot_fig_bytes = fig_to_png_bytes(fig)
            st.session_state.plot_stats_df = summary_df
            st.session_state.plot_done = True
            st.session_state.plot_error = None
            st.session_state.plot_zip_bytes = zip_bytes
            st.session_state.plot_zip_name = zip_name
            plt.close(fig)
        except Exception as e:
            clear_plot_state()
            st.session_state.plot_error = f'Ошибка построения графика: {e}'

if st.session_state.plot_error:
    st.error(st.session_state.plot_error)

if st.session_state.plot_done and st.session_state.plot_fig_bytes is not None:
    st.success('График построен.')
    st.image(st.session_state.plot_fig_bytes, width='stretch')

    stats_df = st.session_state.plot_stats_df
    dcols = st.columns(3 if isinstance(stats_df, pd.DataFrame) else 1)
    with dcols[0]:
        st.download_button(
            'Скачать PNG',
            data=st.session_state.plot_fig_bytes,
            file_name='chart.png',
            mime='image/png',
            width='stretch',
        )
        if st.session_state.plot_zip_bytes is not None:
            st.download_button(
                'Скачать ZIP графиков',
                data=st.session_state.plot_zip_bytes,
                file_name=st.session_state.plot_zip_name or 'charts.zip',
                mime='application/zip',
                width='stretch',
            )

    if isinstance(stats_df, pd.DataFrame):
        st.markdown('**Таблица статистики**')
        st.dataframe(stats_df, width='stretch')
        with dcols[1]:
            st.download_button(
                'Скачать stats CSV',
                data=df_to_csv_bytes(stats_df),
                file_name='chart_stats.csv',
                mime='text/csv',
                width='stretch',
            )
        with dcols[2]:
            st.download_button(
                'Скачать stats XLSX',
                data=df_to_xlsx_bytes(stats_df),
                file_name='chart_stats.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                width='stretch',
            )
