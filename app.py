import streamlit as st
import pandas as pd
import joblib
import random
import numpy as np
import geoip2.database

st.set_page_config(page_title="Cyber Attack Detection", layout="wide")
st.title("Cyber Attack Detection")

TARGET_COL = "Attack Type" 
CITY_DB_PATH = "GeoLite2-City.mmdb" 

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("models/model_LogReg.joblib"),
        "Decision Tree": joblib.load("models/model_RandomForest.joblib"),
        "Random Forest": joblib.load("models/model_DecisionTree.joblib")
    }

@st.cache_data
def load_reference_dataset():
    df = pd.read_csv("data/cybersecurity_attacks.csv")
    return df


def get_expected_feature_columns(df_ref: pd.DataFrame) -> list:
    expected_cols = [c for c in df_ref.columns if c != TARGET_COL]
    return expected_cols


def validate_columns_allow_target(df_input: pd.DataFrame, expected_feature_cols: list, target_col: str) -> tuple[bool, str]:
    input_cols = list(df_input.columns)

    # to handle if the target is missing or not 
    allowed_cols = set(expected_feature_cols) | {target_col}
    missing = [c for c in expected_feature_cols if c not in input_cols]
    extra = [c for c in input_cols if c not in allowed_cols]

    if missing or extra:
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing column(s): {missing}")
        if extra:
            msg_parts.append(f"Extra column(s): {extra}")
        return False, " | ".join(msg_parts)

    return True, "OK"

def make_random_rows(df_ref: pd.DataFrame, expected_cols: list, n: int) -> pd.DataFrame:
    rows = []
    for _ in range(n):
        row_dict = {}
        for col in expected_cols:
            row_dict[col] = df_ref[col].sample(1).iloc[0]
        rows.append(row_dict)
    return pd.DataFrame(rows)

def predict_table_for_rows(df_raw: pd.DataFrame, models: dict, class_labels: list, city_db_path: str) -> pd.DataFrame:
    df_feat = build_features(df_raw, city_db_path=city_db_path)
    out = pd.DataFrame(index=df_raw.index)
    for model_name, model in models.items():
        out[f"pred_{model_name}"] = model.predict(df_feat)

    out["pred_Random baseline"] = [random.choice(class_labels) for _ in range(len(df_raw))]

    return out.reset_index(drop=True)


def predict_with_all_models(df_input: pd.DataFrame, models: dict, class_labels: list) -> pd.DataFrame:
    results = []
    for model_name, model in models.items():
        df_feat = build_features(df_input, city_db_path="GeoLite2-City.mmdb")
        pred = model.predict(df_feat)[0]
        results.append({"Model": model_name, "Prediction": pred})

    # display prediction of each model + add a randomizer 
    results.append({"Model": "Random baseline", "Prediction": random.choice(class_labels)})

    return pd.DataFrame(results)





def build_features(df_raw: pd.DataFrame, city_db_path: str = "GeoLite2-City.mmdb") -> pd.DataFrame:
    df = df_raw.copy()
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    if "Attack Type" in df.columns:
        df = df.drop(columns=["Attack Type"])

    required_raw_cols = [
        "Timestamp",
        "Source IP Address",
        "Destination IP Address",
        "Source Port",
        "Destination Port",
        "Payload Data",
        "User Information",
        "Device Information",
        "Geo-location Data",
        "Proxy Information",
        "Malware Indicators",
        "Alerts/Warnings",
        "Firewall Logs",
        "IDS/IPS Alerts",
    ]
    missing_raw = [c for c in required_raw_cols if c not in df.columns]
    if missing_raw:
        raise ValueError(f"Missing required raw columns for feature engineering: {missing_raw}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["year"] = df["Timestamp"].dt.year
    df["month"] = df["Timestamp"].dt.month
    df["day"] = df["Timestamp"].dt.day
    df["hour"] = df["Timestamp"].dt.hour
    df["minute"] = df["Timestamp"].dt.minute
    df["second"] = df["Timestamp"].dt.second
    df["ts_dayofweek"] = df["Timestamp"].dt.dayofweek
    df["ts_is_weekend"] = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(int)
    src_country = []
    dst_country = []
    with geoip2.database.Reader(city_db_path) as reader:
        for ip in df["Source IP Address"].astype(str):
            try:
                r = reader.city(ip)
                src_country.append(r.country.iso_code)
            except Exception:
                src_country.append(None)

        for ip in df["Destination IP Address"].astype(str):
            try:
                r = reader.city(ip)
                dst_country.append(r.country.iso_code)
            except Exception:
                dst_country.append(None)

    df["src_country"] = src_country
    df["dst_country"] = dst_country
    df["same_country"] = (df["src_country"] == df["dst_country"]).astype(int)
    df["src_country"] = df["src_country"].fillna("UNK")
    df["dst_country"] = df["dst_country"].fillna("UNK")

    def port_class(p):
        try:
            p = int(p)
        except Exception:
            return "dynamic"  # fallback
        return "registered" if p <= 49151 else "dynamic"

    df["src_port_class"] = df["Source Port"].apply(port_class)
    df["dst_port_class"] = df["Destination Port"].apply(port_class)
    payload = df["Payload Data"].astype(str)
    df["payload_char_count"] = payload.str.len()
    df["payload_word_count"] = payload.str.split().str.len()
    df["payload_punct_count"] = payload.str.count(r"[^\w\s]")
    df["payload_punct_ratio"] = 0.0
    mask_nonzero = df["payload_char_count"] != 0
    df.loc[mask_nonzero, "payload_punct_ratio"] = (
        df.loc[mask_nonzero, "payload_punct_count"] / df.loc[mask_nonzero, "payload_char_count"]
    )
    user = df["User Information"].astype(str)
    df["user_first_name"] = user.str.split().str[0]
    df["user_last_name"] = user.str.split().str[-1]
    df["user_char_count"] = user.str.len()
    device = df["Device Information"].astype(str).str.lower()
    df["device_os"] = "other"
    df.loc[device.str.contains("windows", na=False), "device_os"] = "windows"
    df.loc[device.str.contains("mac", na=False), "device_os"] = "macos"
    df.loc[device.str.contains("linux", na=False), "device_os"] = "linux"
    df.loc[device.str.contains("android", na=False), "device_os"] = "android"
    df.loc[device.str.contains("ios|iphone|ipad", na=False), "device_os"] = "ios"
    df["device_type"] = "other"
    df.loc[device.str.contains("mobile|android|iphone", na=False), "device_type"] = "mobile"
    df.loc[device.str.contains("tablet|ipad", na=False), "device_type"] = "tablet"
    df.loc[device.str.contains("desktop|windows|mac|linux", na=False), "device_type"] = "desktop"
    df["device_browser"] = "other"
    df.loc[device.str.contains("chrome", na=False), "device_browser"] = "chrome"
    df.loc[device.str.contains("firefox", na=False), "device_browser"] = "firefox"
    df.loc[device.str.contains("safari", na=False), "device_browser"] = "safari"
    df.loc[device.str.contains("edge", na=False), "device_browser"] = "edge"
    df.loc[device.str.contains("opera", na=False), "device_browser"] = "opera"
    geo = df["Geo-location Data"].astype(str)
    geo_split = geo.str.split(",", expand=True)
    df["geo_city"] = geo_split[0].str.strip()
    df["geo_state"] = geo_split[1].str.strip() if geo_split.shape[1] > 1 else None
    df["is_proxy"] = df["Proxy Information"].notna().astype(int)
    bool_cols = ["Malware Indicators", "Alerts/Warnings", "Firewall Logs", "IDS/IPS Alerts"]
    for col in bool_cols:
        df[col] = df[col].notna().astype(int)
    columns_to_drop = [
        "Timestamp",
        "Source IP Address",
        "Destination IP Address",
        "Source Port",
        "Destination Port",
        "Payload Data",
        "User Information",
        "Device Information",
        "Geo-location Data",
        "Proxy Information",
    ]
    df = df.drop(columns=columns_to_drop)
    return df


models = load_models()
df_ref = load_reference_dataset()
expected_cols = get_expected_feature_columns(df_ref)
class_labels = sorted(df_ref[TARGET_COL].dropna().unique().tolist())

st.write("Select an input mode :")
mode = st.radio(
    "Input mode",
    ["Upload CSV", "Random row(s) from dataset"],
    index=0
)

st.divider()
if mode == "Upload CSV":
    st.subheader("Upload CSV")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        df_input = df_input.loc[:, ~df_input.columns.astype(str).str.match(r"^Unnamed")]
        df_input = df_input.dropna(axis=1, how="all")
        is_valid, msg = validate_columns_allow_target(df_input, expected_cols, TARGET_COL)
        if not is_valid:
            st.error("invalid csv format")
            st.write(msg)
            st.stop()

        if TARGET_COL in df_input.columns:
            st.info(f"'{TARGET_COL}' column detected : will be ignored for the prediction")
            df_input = df_input.drop(columns=[TARGET_COL])

        df_input = df_input.reset_index(drop=True)

        col_left, col_mid, col_right = st.columns([3, 1, 3], gap="large")

        with col_left:
            st.markdown("### Input (CSV)")
            st.dataframe(df_input, height=380)

        with col_mid:
            st.markdown("### ")
            st.markdown("### ")
            predict_clicked = st.button("Predict", use_container_width=True)

        with col_right:
            st.markdown("### Predictions")

            if predict_clicked:
                with st.spinner("Processing… (feature engineering + prediction)"):
                    preds = predict_table_for_rows(
                        df_raw=df_input,
                        models=models,
                        class_labels=class_labels,
                        city_db_path=CITY_DB_PATH
                    )
                preds = preds.reset_index(drop=True)
                st.dataframe(preds, height=380)

            
            else:
                st.info("Click Predict to generate predictions.")

else:
    st.subheader("Random row(s) from dataset")

    n_rows = st.radio(
        "How many random inputs?",
        options=[1, 5, 10],
        index=0,
        horizontal=True
    )

    # Générer une fois, puis prédire au clic
    if "df_random_current" not in st.session_state:
        st.session_state.df_random_current = None

    # Bouton génération (facultatif, mais pratique)
    if st.button("Generate random input(s)", use_container_width=True):
        st.session_state.df_random_current = make_random_rows(df_ref, expected_cols, n=int(n_rows)).reset_index(drop=True)

    df_raw = st.session_state.df_random_current
    if df_raw is None:
        st.info("Generate random input(s) first.")
        st.stop()

    col_left, col_mid, col_right = st.columns([3, 1, 3], gap="large")

    with col_left:
        st.markdown("### Input (Random)")
        st.dataframe(df_raw, height=380)

    with col_mid:
        st.markdown("### ")
        st.markdown("### ")
        predict_clicked = st.button("Predict", use_container_width=True)

    with col_right:
        st.markdown("### Predictions")
        if predict_clicked:
            with st.spinner("Processing… (feature engineering + prediction)"):
                preds = predict_table_for_rows(
                    df_raw=df_raw,
                    models=models,
                    class_labels=class_labels,
                    city_db_path=CITY_DB_PATH
                )
            preds = preds.reset_index(drop=True)
            st.dataframe(preds, height=380)
        else:
            st.info("Click Predict to generate predictions.")