import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

st.set_page_config(page_title="Оценка вина", page_icon="🍷", layout="wide")

APP_DIR = Path(__file__).parent

INPUT_FORMAT = {
    "fixed_acidity": (0.1, "%.1f"),
    "volatile_acidity": (0.01, "%.3f"),
    "citric_acid": (0.01, "%.2f"),
    "residual_sugar": (0.1, "%.2f"),
    "chlorides": (0.001, "%.3f"),
    "free_sulfur_dioxide": (1.0, "%.0f"),
    "total_sulfur_dioxide": (1.0, "%.0f"),
    "density": (0.0001, "%.5f"),
    "pH": (0.01, "%.2f"),
    "sulphates": (0.01, "%.2f"),
    "alcohol": (0.1, "%.1f"),
}


# Ответ к заданию 4.5 — в README.md
@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_config():
    return json.loads((APP_DIR / "config.json").read_text())


@st.cache_data
def load_test():
    df = pd.read_csv(APP_DIR / "test.csv")
    return df.drop(columns="target"), df["target"]


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def test_curves(y, proba):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    precision, recall, _ = metrics.precision_recall_curve(y, proba)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "roc_auc": metrics.roc_auc_score(y, proba),
        "pr_auc": metrics.average_precision_score(y, proba),
    }


def predict_batch(df, model, features, threshold):
    df = df.rename(columns=lambda c: str(c).strip())
    if df.empty:
        return None, "Файл пустой.", None

    missing = [f for f in features if f not in df.columns]
    if missing:
        return (
            None,
            "В файле не хватает столбцов: " + ", ".join(f"`{f}`" for f in missing),
            None,
        )

    X = df[features].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().any(axis=1)
    if bad.all():
        return (
            None,
            "Ни в одной строке нет корректных числовых значений всех признаков.",
            None,
        )

    warning = None
    if bad.any():
        warning = f"Пропущено строк с пустыми или нечисловыми значениями: {bad.sum()} (номера строк: {', '.join(str(i + 1) for i in np.flatnonzero(bad)[:10])}{'…' if bad.sum() > 10 else ''})."

    result = df.loc[~bad].drop(columns="target", errors="ignore").copy()
    result["proba_good"] = model.predict_proba(X.loc[~bad])[:, 1].round(4)
    result["is_good"] = (result["proba_good"] >= threshold).astype(int)
    return result, None, warning


model = load_model()
FEATURES = list(model.feature_names_in_)
X_test, y_test = load_test()
proba = test_proba(model, X_test)
default_threshold = round(float(load_config()["threshold"]), 3)

st.sidebar.header("Настройки")
threshold = st.sidebar.slider(
    'Порог "хорошего" вина',
    0.0,
    1.0,
    default_threshold,
    step=0.001,
    format="%.3f",
    help="Вино считается хорошим, если вероятность не ниже порога.",
)
st.sidebar.caption(
    f"Рекомендованный порог (максимум F1 на валидации): **{default_threshold:.3f}**"
)

st.title("🍷 Оценка качества вина")
st.caption("Логистическая регрессия предсказывает вероятность того, что вино хорошее.")

tab_one, tab_metrics, tab_batch, tab_coef = st.tabs(
    ["Одно вино", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with tab_one:
    st.subheader("Характеристики вина")
    stats = X_test.agg(["min", "max", "median"])
    values = {}

    is_red = st.toggle("Красное вино", value=bool(stats.loc["median", "is_red"]))
    cols = st.columns(3)
    for i, f in enumerate(f for f in FEATURES if f != "is_red"):
        step, fmt = INPUT_FORMAT.get(f, (0.01, "%.3f"))
        values[f] = cols[i % 3].number_input(
            f,
            min_value=float(stats.loc["min", f]),
            max_value=float(stats.loc["max", f]),
            value=float(stats.loc["median", f]),
            step=step,
            format=fmt,
        )
    values["is_red"] = int(is_red)

    p = model.predict_proba(pd.DataFrame([values])[FEATURES])[0, 1]
    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Вероятность, что вино хорошее", f"{p:.1%}")
    c2.metric("Порог", f"{threshold:.3f}")
    if p >= threshold:
        st.success("Решение модели: **хорошее вино** $-$ попадает в подборку.")
    else:
        st.warning("Решение модели: **обычное вино** $-$ в подборку не попадает.")

with tab_metrics:
    st.subheader("Качество на тестовой выборке")
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1", f"{f1:.3f}")
    c4.metric(
        "Вин в подборке", tp + fp, help=f"Всего вин в тестовой выборке: {len(pred)}"
    )

    curves = test_curves(y_test, proba)
    fpr_t = fp / (fp + tn)
    prec_t = precision if tp + fp else 1.0

    col_cm, col_text = st.columns([2, 3])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    metrics.ConfusionMatrixDisplay(
        np.array([[tn, fp], [fn, tp]]), display_labels=["обычное", "хорошее"]
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set(title="Матрица ошибок", xlabel="Предсказание", ylabel="Истина")
    fig.tight_layout()
    col_cm.pyplot(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(
        curves["fpr"], curves["tpr"], label=f"ROC (AUC = {curves['roc_auc']:.3f})"
    )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="случайная модель")
    axes[0].scatter(
        fpr_t, recall, color="red", zorder=3, s=60, label=f"порог {threshold:.3f}"
    )
    axes[0].set(title="ROC-кривая", xlabel="FPR", ylabel="TPR (recall)")
    axes[0].legend(loc="lower right")

    axes[1].plot(
        curves["recall"], curves["precision"], label=f"PR (AP = {curves['pr_auc']:.3f})"
    )
    axes[1].axhline(
        y_test.mean(), color="k", ls="--", lw=1, alpha=0.5, label="доля хороших вин"
    )
    axes[1].scatter(
        recall, prec_t, color="red", zorder=3, s=60, label=f"порог {threshold:.3f}"
    )
    axes[1].set(title="PR-кривая", xlabel="Recall", ylabel="Precision")
    axes[1].legend(loc="upper right")
    fig.tight_layout()

    col_text.info(
        "**Как порог влияет на подборку.**\n\n"
        "- **Порог выше** $\\rightarrow$ в подборку попадает меньше вин, но среди них больше действительно хороших "
        "(precision растёт). Часть хороших вин при этом теряется (recall падает).\n"
        "- **Порог ниже** $\\rightarrow$ подборка шире и почти не упускает хорошие вина (recall растёт), "
        "но в неё чаще попадают обычные вина (precision падает).\n\n"
        f"Сейчас из {tp + fp} вин подборки хороших $-$ {tp}, "
        f"а из {tp + fn} хороших вин тестовой выборки в подборку попали {tp}."
    )
    st.pyplot(fig)
    plt.close(fig)

with tab_batch:
    st.subheader("Оценка списка вин из CSV")
    st.write(
        f"Загрузите CSV-файл со столбцами: {', '.join(f'`{f}`' for f in FEATURES)}. "
        f"Используется порог из боковой панели: **{threshold:.3f}**."
    )
    st.download_button(
        "Скачать пример файла",
        X_test.head(20).to_csv(index=False).encode("utf-8"),
        file_name="wine_example.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("CSV-файл с признаками вин", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, sep=None, engine="python")
        except Exception as e:
            st.error(f"Не удалось прочитать файл как CSV: {e}")
        else:
            result, error, warning = predict_batch(df, model, FEATURES, threshold)
            if error:
                st.error(error)
            else:
                if warning:
                    st.warning(warning)
                st.write(f"Хороших вин: **{result['is_good'].sum()}** из {len(result)}")
                st.dataframe(result, width="stretch")
                st.download_button(
                    "Скачать результат",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="wine_predictions.csv",
                    mime="text/csv",
                )

with tab_coef:
    st.subheader("Коэффициенты логистической регрессии")
    st.caption(
        "Признаки стандартизованы, поэтому коэффициенты сравнимы между собой: "
        "это изменение логита при росте признака на одно стандартное отклонение."
    )
    coef = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coef.index, coef.values, color=np.where(coef > 0, "tab:green", "tab:red"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Коэффициент")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    pos = coef[coef > 0].sort_values(ascending=False).head(3)
    neg = coef[coef < 0].head(3)
    c1, c2 = st.columns(2)
    c1.markdown(
        "**За хорошее вино:**\n"
        + "\n".join(f"- `{k}` ({v:+.2f})" for k, v in pos.items())
    )
    c2.markdown(
        "**Против хорошего вина:**\n"
        + "\n".join(f"- `{k}` ({v:+.2f})" for k, v in neg.items())
    )
