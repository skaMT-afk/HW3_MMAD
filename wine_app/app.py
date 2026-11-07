import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title='Подборка вин', layout='wide')


@st.cache_resource
def load_model(path, model_version):
    return joblib.load(path)


@st.cache_data
def load_test(path):
    return pd.read_csv(path)


@st.cache_data
def test_proba(_model, features, model_version):
    # model_version меняет ключ кэша при обновлении файла модели.
    return _model.predict_proba(features)[:, 1]


model_path = APP_DIR / 'model.joblib'
model_version = model_path.stat().st_mtime_ns
model = load_model(model_path, model_version)
test = load_test(APP_DIR / 'test.csv')
config = json.loads((APP_DIR / 'config.json').read_text(encoding='utf-8'))
columns = list(model.feature_names_in_)
X_test = test[columns]
y_test = test['target']
threshold = st.sidebar.slider('Порог рекомендации', 0.0, 1.0,
                              value=config['threshold'], step=0.001,
                              format='%.6f')

st.title('Подборка вин')
st.write('Логистическая регрессия оценивает вероятность экспертной оценки не ниже 7.')
single, model_metrics, batch, coefficients = st.tabs([
    'Одно вино', 'Модельные метрики', 'Пакетное предсказание', 'Коэффициенты'
])

with single:
    with st.form('single_wine'):
        inputs = {}
        blocks = st.columns(3)
        for i, column in enumerate(columns):
            with blocks[i % len(blocks)]:
                if column == 'is_red':
                    inputs[column] = int(st.toggle(
                        'Красное вино (выкл. — белое)', value=bool(X_test[column].mode().iloc[0])
                    ))
                else:
                    values = X_test[column]
                    minimum, maximum = float(values.min()), float(values.max())
                    inputs[column] = st.number_input(
                        column, min_value=minimum, max_value=maximum,
                        value=float(values.median()), step=max((maximum-minimum)/100, 0.000001),
                        format='%.6f'
                    )
        submitted = st.form_submit_button('Оценить вино')
    if submitted:
        st.session_state['single_probability'] = float(
            model.predict_proba(pd.DataFrame([inputs], columns=columns))[0, 1]
        )
    if 'single_probability' in st.session_state:
        probability = st.session_state['single_probability']
        st.metric('Вероятность оценки ≥ 7', f'{probability:.1%}')
        if probability >= threshold:
            st.success('Включить в подборку')
        else:
            st.info('Не включать в подборку')

with model_metrics:
    scores = test_proba(model, X_test, model_version)
    predicted = scores >= threshold
    cm = metrics.confusion_matrix(y_test, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_test, predicted, zero_division=0)
    recall = metrics.recall_score(y_test, predicted, zero_division=0)
    f1 = metrics.f1_score(y_test, predicted, zero_division=0)
    for col, (label, score) in zip(st.columns(3), [
        ('Precision', precision), ('Recall', recall), ('F1', f1)
    ]):
        col.metric(label, f'{score:.3f}')
    fig, (cm_ax, roc_ax, pr_ax) = plt.subplots(1, 3, figsize=(13, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=[0, 1]).plot(ax=cm_ax, colorbar=False)
    cm_ax.set_title('Матрица ошибок')
    metrics.RocCurveDisplay.from_predictions(y_test, scores, ax=roc_ax, name='Модель')
    roc_ax.scatter(fp / (fp + tn), recall, c='tab:red', zorder=5, label='Текущий порог')
    roc_ax.set_title('ROC')
    metrics.PrecisionRecallDisplay.from_predictions(y_test, scores, ax=pr_ax, name='Модель')
    # При отсутствии рекомендаций precision для точки отображаем как 0, как и в метрике.
    pr_ax.scatter(recall, precision, c='tab:red', zorder=5, label='Текущий порог')
    pr_ax.axhline(y_test.mean(), ls='--', color='gray', label='Случайный уровень')
    pr_ax.set_title('Precision–recall')
    roc_ax.legend()
    pr_ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('При повышении порога в подборку попадает меньше вин или столько же. '
             'При снижении порога охват хороших вин растёт или остаётся прежним, '
             'но в подборку могут попасть и вина с низкой экспертной оценкой.')

with batch:
    st.caption('Загрузите CSV с 12 признаками, без target. Используется порог из боковой панели.')
    uploaded = st.file_uploader('Файл с характеристиками вин', type=['csv'])
    if uploaded is not None:
        try:
            data = pd.read_csv(uploaded)
            missing = [column for column in columns if column not in data]
            if missing:
                st.error('Не хватает столбцов: ' + ', '.join(missing))
            elif data.empty:
                st.error('Добавьте в CSV хотя бы одну строку с характеристиками.')
            else:
                features = data[columns].apply(pd.to_numeric, errors='coerce')
                if not np.isfinite(features.to_numpy(dtype=float)).all():
                    st.error('Все признаки должны быть конечными числами без пропусков.')
                elif not features['is_red'].isin([0, 1]).all():
                    st.error('В столбце is_red допустимы только 0 и 1.')
                else:
                    probabilities = model.predict_proba(features)[:, 1]
                    result = data.assign(probability=probabilities,
                                         recommended=(probabilities >= threshold).astype(int))
                    st.dataframe(result, hide_index=True)
                    st.download_button('Скачать результат', result.to_csv(index=False),
                                       file_name='wine_predictions.csv', mime='text/csv')
        except (ValueError, UnicodeError) as exc:
            st.error(f'Не удалось обработать CSV: {exc}')

with coefficients:
    weights = pd.Series(model.named_steps['logisticregression'].coef_[0], index=columns).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(weights.index, weights, color=['tab:blue' if w >= 0 else 'tab:red' for w in weights])
    ax.axvline(0, color='gray', linewidth=0.8)
    ax.set_xlabel('Коэффициент в логитах после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('За хорошее вино: ' + ', '.join(weights[weights > 0].nlargest(3).index) + '.')
    st.write('Против: ' + ', '.join(weights[weights < 0].nsmallest(3).index) + '.')
