"""Демонстрационная антифрод-система на данных Credit Card Fraud Detection 2026.
Запуск: streamlit run antifraud_app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="FraudGuard · Антифрод", page_icon="🛡️", layout="wide")
DATA_DEFAULT = Path(__file__).parent / "credit_card_fraud_detection_2026(1).csv"
MODEL_FILE = Path(__file__).parent / "fraudguard_model.joblib"
TARGET = "is_fraud"
# Эти признаки либо раскрывают ответ, либо являются идентификаторами/невалидными датами.
EXCLUDE = {"is_fraud", "fraud_type", "fraud_amount_lost", "transaction_id",
           "customer_id", "card_id", "merchant_id", "transaction_datetime",
           "transaction_day_of_week", "is_weekend"}


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Одинаковая подготовка для обучения, ручного ввода и пакетной проверки."""
    df = raw.copy()
    if "transaction_datetime" in df:
        dt = pd.to_datetime(df["transaction_datetime"], errors="coerce")
        df["weekday_corrected"] = dt.dt.dayofweek.astype("float64")
        df["weekend_corrected"] = dt.dt.dayofweek.isin([5, 6]).astype("float64")
        df.loc[dt.isna(), ["weekday_corrected", "weekend_corrected"]] = np.nan
        # Час берём из даты, если он доступен, а не из потенциально несогласованного поля.
        df["transaction_hour"] = dt.dt.hour.where(dt.notna(), pd.to_numeric(df.get("transaction_hour"), errors="coerce"))
    df = df.drop(columns=[c for c in EXCLUDE if c in df], errors="ignore")
    if "browser" in df:
        df["browser"] = df["browser"].fillna("Not applicable")
    return df


def load_csv(source) -> pd.DataFrame:
    df = pd.read_csv(source, low_memory=False)
    if TARGET not in df.columns:
        raise ValueError("В исходном CSV отсутствует столбец is_fraud")
    return df


def build_pipeline(x: pd.DataFrame) -> Pipeline:
    cat = x.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num = [c for c in x.columns if c not in cat]
    transformer = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num),
        ("cat", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=30)),
        ]), cat),
    ])
    model = RandomForestClassifier(n_estimators=140, max_depth=18,
                                   min_samples_leaf=4, class_weight="balanced_subsample",
                                   random_state=42, n_jobs=-1)
    return Pipeline([("features", transformer), ("classifier", model)])


def metrics_at_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}


@st.cache_resource(show_spinner="Обучаем модель на исторических транзакциях…")
def train_cached(csv_bytes: bytes):
    raw = load_csv(io.BytesIO(csv_bytes))
    raw = raw.dropna(subset=[TARGET]).copy()
    raw[TARGET] = pd.to_numeric(raw[TARGET], errors="coerce")
    raw = raw.dropna(subset=[TARGET])
    raw = raw[raw[TARGET].isin([0, 1])].copy()
    if len(raw) < 100 or raw[TARGET].nunique() != 2:
        raise ValueError("Недостаточно строк или присутствует только один класс")
    if "transaction_datetime" in raw:
        raw["__sort_date"] = pd.to_datetime(raw["transaction_datetime"], errors="coerce")
        raw = raw.sort_values("__sort_date", kind="stable").drop(columns="__sort_date")
    else:
        st.warning("Нет даты: разбиение выполнено в исходном порядке строк.")
    # Временной holdout: последние 20% не используются при обучении.
    cut = int(len(raw) * 0.8)
    train, test = raw.iloc[:cut].copy(), raw.iloc[cut:].copy()
    if train[TARGET].nunique() < 2 or test[TARGET].nunique() < 2:
        raise ValueError("Один из временных периодов содержит только один класс")
    x_train, x_test = prepare(train), prepare(test)
    features = x_train.columns.tolist()
    model = build_pipeline(x_train)
    model.fit(x_train, train[TARGET].astype(int))
    scores = model.predict_proba(x_test[features])[:, 1]
    return {"model": model, "features": features, "train": train, "test": test,
            "scores": scores, "ap": float(average_precision_score(test[TARGET], scores)),
            "auc": float(roc_auc_score(test[TARGET], scores)),
            "reference": train.iloc[0].to_dict()}


def predict_rows(bundle, rows: pd.DataFrame) -> pd.DataFrame:
    x = prepare(rows)
    # Обязательное условие: все обучающие поля должны присутствовать; лишние игнорируем.
    missing = [c for c in bundle["features"] if c not in x.columns]
    if missing:
        raise ValueError("Не хватает признаков: " + ", ".join(missing))
    scores = bundle["model"].predict_proba(x[bundle["features"]])[:, 1]
    result = rows.copy()
    result["fraud_probability"] = scores
    return result


st.title("🛡️ FraudGuard — интеллектуальный анализ транзакций")
st.caption("Учебный прототип: оценки модели не являются подтверждением мошенничества и не должны использоваться для реальных блокировок.")
with st.sidebar:
    st.header("⚙️ Настройки")
    threshold = st.slider("Порог подозрительности", 0.05, 0.95, 0.50, 0.05)
    st.caption("Низкий порог повышает чувствительность, но обычно увеличивает ложные тревоги.")
    upload = st.file_uploader("Исходный датасет (CSV)", type="csv")
    st.caption("Если файл лежит рядом с программой, он загрузится автоматически.")

if upload is not None:
    data_bytes = upload.getvalue()
elif DATA_DEFAULT.exists():
    data_bytes = DATA_DEFAULT.read_bytes()
else:
    st.info("Загрузите CSV слева или положите исходный файл рядом с antifraud_app.py.")
    st.stop()

try:
    bundle = train_cached(data_bytes)
except Exception as exc:
    st.error(f"Не удалось обучить модель: {exc}")
    st.stop()

train, test = bundle["train"], bundle["test"]
y_test = test[TARGET].astype(int).to_numpy()
scores = bundle["scores"]
m = metrics_at_threshold(y_test, scores, threshold)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Обзор", "🔎 Проверка операции", "📁 Проверка CSV", "🧪 Качество модели"])
with tab1:
    a, b, c, d = st.columns(4)
    a.metric("Обучающих операций", f"{len(train):,}".replace(",", " "))
    b.metric("Тестовых операций", f"{len(test):,}".replace(",", " "))
    c.metric("Мошенничество (в тесте)", f"{y_test.mean():.2%}")
    d.metric("PR-AUC (Average Precision)", f"{bundle['ap']:.3f}")
    st.subheader("Доли классов в исходном датасете")
    overview = pd.concat([train, test])[TARGET].value_counts().rename_axis("Класс").reset_index(name="Количество")
    overview["Класс"] = overview["Класс"].map({0: "Обычная", 1: "Мошенническая"})
    st.plotly_chart(px.bar(overview, x="Класс", y="Количество", text="Количество"), use_container_width=True)
    if "transaction_channel" in test:
        st.subheader("Доля мошенничества по каналам (тест)")
        grp = test.groupby("transaction_channel", dropna=False)[TARGET].agg(["mean", "size"]).reset_index()
        grp.columns = ["Канал", "Доля мошенничества", "Количество"]
        st.plotly_chart(px.bar(grp, x="Канал", y="Доля мошенничества", hover_data=["Количество"]), use_container_width=True)
    st.caption("Графики показывают статистику набора данных, а не действующую банковскую систему.")

with tab2:
    st.subheader("Проверка одной операции")
    st.write("Поля предзаполнены примером из обучающей части набора. Измените их и нажмите кнопку.")
    sample = bundle["reference"].copy()
    categories = sorted(train["merchant_category"].dropna().astype(str).unique()) if "merchant_category" in train else []
    channels = sorted(train["transaction_channel"].dropna().astype(str).unique()) if "transaction_channel" in train else []
    countries = sorted(train["merchant_country"].dropna().astype(str).unique()) if "merchant_country" in train else []
    with st.form("single_transaction"):
        left, right = st.columns(2)
        with left:
            sample["transaction_amount"] = st.number_input("Сумма транзакции", min_value=0.0, value=float(sample.get("transaction_amount", 100.0)), step=10.0)
            if categories:
                sample["merchant_category"] = st.selectbox("Категория магазина", categories, index=categories.index(str(sample["merchant_category"])))
            if channels:
                sample["transaction_channel"] = st.selectbox("Канал операции", channels, index=channels.index(str(sample["transaction_channel"])))
            if countries:
                sample["merchant_country"] = st.selectbox("Страна магазина", countries, index=countries.index(str(sample["merchant_country"])))
            sample["distance_from_home_km"] = st.number_input("Расстояние от дома (км)", min_value=0.0, value=float(sample.get("distance_from_home_km", 0.0)))
            sample["transactions_last_1h"] = st.number_input("Операций за последний час", min_value=0, value=int(sample.get("transactions_last_1h", 0)))
        with right:
            sample["amount_vs_avg_ratio"] = st.number_input("Сумма / средняя сумма клиента", min_value=0.0, value=float(sample.get("amount_vs_avg_ratio", 1.0)), step=0.1)
            sample["international_transaction"] = int(st.checkbox("Международная операция", value=bool(sample.get("international_transaction", 0))))
            sample["high_risk_country"] = int(st.checkbox("Страна повышенного риска", value=bool(sample.get("high_risk_country", 0))))
            sample["card_present"] = int(st.checkbox("Карта физически присутствует", value=bool(sample.get("card_present", 0))))
            sample["num_declined_last_30d"] = st.number_input("Отклонённых операций за 30 дней", min_value=0, value=int(sample.get("num_declined_last_30d", 0)))
            sample["credit_score"] = st.number_input("Кредитный рейтинг", min_value=0, value=int(sample.get("credit_score", 600)))
        submitted = st.form_submit_button("Оценить риск", type="primary")
    if submitted:
        try:
            prob = float(predict_rows(bundle, pd.DataFrame([sample]))["fraud_probability"].iloc[0])
            st.metric("Оценка вероятности мошенничества", f"{prob:.1%}")
            if prob >= threshold:
                st.warning("Операция требует дополнительной проверки (по выбранному порогу).")
            else:
                st.success("Риск ниже выбранного порога; это не гарантия безопасности.")
            st.progress(min(1.0, prob))
            st.caption("Остальные признаки взяты из одной примерной транзакции. Это демонстрация, а не независимая проверка введённых параметров.")
        except Exception as exc:
            st.error(str(exc))

with tab3:
    st.subheader("Пакетная проверка операций")
    st.write("Загрузите CSV с теми же входными полями, что в исходном наборе. Поля is_fraud, fraud_type и fraud_amount_lost для прогноза не нужны.")
    batch_file = st.file_uploader("CSV для проверки", type="csv", key="batch")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file, low_memory=False)
            if len(batch) > 100_000:
                st.error("Для демонстрации загрузите не более 100 000 строк за раз.")
            else:
                result = predict_rows(bundle, batch)
                result["requires_review"] = result["fraud_probability"] >= threshold
                st.metric("Подозрительных операций", int(result["requires_review"].sum()))
                st.dataframe(result[[c for c in ["transaction_id", "transaction_amount", "fraud_probability", "requires_review"] if c in result]], use_container_width=True)
                st.download_button("⬇️ Скачать результаты", result.to_csv(index=False).encode("utf-8-sig"), "fraud_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Ошибка проверки: {exc}")

with tab4:
    st.subheader("Качество на отложенных последних 20% транзакций")
    a, b, c, d = st.columns(4)
    a.metric("Precision", f"{m['precision']:.1%}")
    b.metric("Recall", f"{m['recall']:.1%}")
    c.metric("F1", f"{m['f1']:.3f}")
    d.metric("ROC-AUC", f"{bundle['auc']:.3f}")
    cm = pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                      index=["Факт: обычная", "Факт: мошенничество"],
                      columns=["Прогноз: обычная", "Прогноз: мошенничество"])
    st.plotly_chart(px.imshow(cm, text_auto=True, aspect="auto", title="Матрица ошибок"), use_container_width=True)
    st.caption("TP — найденные мошеннические; FN — пропущенные мошеннические; FP — ложные тревоги.")
    st.write(f"**PR-AUC / Average Precision:** {bundle['ap']:.4f} · **ROC-AUC:** {bundle['auc']:.4f}")
    st.info("Порог влияет на Precision/Recall/F1 и матрицу ошибок, но не на ROC-AUC и PR-AUC. Порог 0.5 не обязательно оптимален.")
    st.warning("Датасет может быть синтетическим. Данные с конца 2026 года не являются подтверждённой историей реальных транзакций. Прототип нельзя использовать для принятия банковских решений.")
