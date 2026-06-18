from pathlib import Path
import logging
import traceback

import pandas as pd
import plotly.express as px
import streamlit as st

from main import generer_dataset_monte_carlo
from ml_utils import FEATURE_COLUMNS, load_model, predict_dataframe, predict_single, train_model


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "dataset_xgboost.csv"
MODEL_PATH = BASE_DIR / "models" / "rupture_model.pkl"
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "dashboard.log"


def _setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )


def _log_exception(label, exc):
    logging.exception("%s failed: %s", label, exc)


@st.cache_data(show_spinner=False)
def load_dataset(path: str):
    return pd.read_csv(path)


def _line_figure(df, x_col, y_cols, title, labels=None):
    available_cols = [col for col in y_cols if col in df.columns]
    if not available_cols:
        return px.line(title=title)

    plot_df = df[[x_col] + available_cols].melt(
        id_vars=x_col,
        value_vars=available_cols,
        var_name="serie",
        value_name="valeur",
    )
    fig = px.line(
        plot_df,
        x=x_col,
        y="valeur",
        color="serie",
        title=title,
        labels=labels,
    )
    return fig


def main():
    _setup_logging()

    try:
        st.set_page_config(page_title="Supply Chain Pharmaceutique", layout="wide")

        st.title("Simulation de la chaine pharmaceutique")
        st.caption("Generation des donnees, entrainement du modele et predictions")

        if "flash_kind" not in st.session_state:
            st.session_state.flash_kind = None
        if "flash_message" not in st.session_state:
            st.session_state.flash_message = None
        if "artifact" not in st.session_state:
            st.session_state.artifact = None
        if "training_report" not in st.session_state:
            st.session_state.training_report = None
        if "dataset_preview" not in st.session_state:
            st.session_state.dataset_preview = None

        def set_flash(kind, message):
            st.session_state.flash_kind = kind
            st.session_state.flash_message = message

        message = st.session_state.get("flash_message")
        if message:
            kind = st.session_state.get("flash_kind")
            if kind == "success":
                st.success(message)
            elif kind == "error":
                st.error(message)
            elif kind == "warning":
                st.warning(message)
            else:
                st.info(message)

        with st.sidebar:
            st.header("Parametres")
            n_simulations = st.number_input("Simulations Monte Carlo", min_value=10, max_value=5000, value=1000, step=10)
            n_periodes = st.number_input("Periodes par simulation", min_value=12, max_value=260, value=104, step=4)
            d0 = st.number_input("Demande moyenne", min_value=1.0, value=100.0, step=1.0)
            sigma = st.number_input("Ecart-type demande", min_value=0.0, value=15.0, step=1.0)
            st.divider()
            generate_clicked = st.button("Generer le dataset")
            train_clicked = st.button("Entrainer le modele")
            load_clicked = st.button("Charger le modele")

        def run_action(label, func):
            try:
                with st.spinner(f"{label} en cours..."):
                    progress = st.progress(0, text=f"{label}...")
                    progress.progress(20, text=f"{label}...")
                    result = func()
                    progress.progress(100, text=f"{label} termine")
                    progress.empty()
                return result
            except Exception as exc:  # pragma: no cover - runtime diagnostics
                _log_exception(label, exc)
                st.error(f"{label} a echoue.")
                st.code(traceback.format_exc())
                return None

        if generate_clicked:
            result = run_action(
                "Generation du dataset",
                lambda: generer_dataset_monte_carlo(
                    n_simulations=int(n_simulations),
                    n_periodes=int(n_periodes),
                    d0=float(d0),
                    sigma=float(sigma),
                    output_path=DATA_PATH,
                ),
            )
            if result is not None:
                dataset, output_path = result
                st.session_state.dataset_preview = dataset.head(1000)
                set_flash("success", f"Dataset genere et sauvegarde dans {output_path}")
                st.rerun()

        if train_clicked:
            if not DATA_PATH.exists():
                set_flash("error", "Le dataset est introuvable. Generer d'abord les donnees.")
                st.rerun()
            result = run_action(
                "Entrainement du modele",
                lambda: train_model(DATA_PATH, model_path=MODEL_PATH),
            )
            if result is not None:
                st.session_state.artifact = result
                st.session_state.training_report = result["metrics"]
                set_flash(
                    "success",
                    f"Modele entraine avec backend {result['backend']} et sauvegarde dans {MODEL_PATH}",
                )
                st.rerun()

        if load_clicked:
            if not MODEL_PATH.exists():
                set_flash("error", "Aucun modele sauvegarde n'a ete trouve.")
                st.rerun()
            result = run_action("Chargement du modele", lambda: load_model(MODEL_PATH))
            if result is not None:
                st.session_state.artifact = result
                st.session_state.training_report = result.get("metrics")
                set_flash("success", f"Modele charge depuis {MODEL_PATH}")
                st.rerun()

        if DATA_PATH.exists():
            df = load_dataset(str(DATA_PATH))
            st.subheader("Apercu du dataset")
            st.dataframe(df.head(20), width="stretch")
        else:
            df = None
            st.warning("Aucun dataset n'est encore genere.")

        if st.session_state.training_report:
            report = st.session_state.training_report
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{report['accuracy']:.3f}")
            c2.metric("Precision", f"{report['precision']:.3f}")
            c3.metric("Recall", f"{report['recall']:.3f}")
            c4.metric("F1", f"{report['f1']:.3f}")
            confusion_df = pd.DataFrame(
                {
                    "Prédit négatif": [
                        f"TN: {report.get('tn', report.get('TN'))}",
                        f"FN: {report.get('fn', report.get('FN'))}",
                    ],
                    "Prédit positif": [
                        f"FP: {report.get('fp', report.get('FP'))}",
                        f"TP: {report.get('tp', report.get('TP'))}",
                    ],
                },
                index=["Réel négatif", "Réel positif"],
            )
            st.subheader("Matrice de confusion")
            st.table(confusion_df)

        tab_simulation, tab_manuelle = st.tabs(["Predictions sur simulation", "Prediction manuelle"])

        with tab_simulation:
            if df is None:
                st.info("Generer d'abord le dataset pour visualiser les predictions.")
            elif st.session_state.artifact is None:
                st.info("Entrainer ou charger le modele pour afficher les predictions.")
            else:
                sim = st.selectbox("Simulation", sorted(df["simulation"].unique()))
                data = df[df["simulation"] == sim].copy()
                pred_data = predict_dataframe(data, st.session_state.artifact)

                col1, col2, col3 = st.columns(3)
                col1.metric("Ruptures grossiste", int(data["rupture_grossiste"].sum()))
                col2.metric("Ruptures pharmacie", int(data["rupture_pharmacie"].sum()))
                col3.metric("Probabilite moyenne", f"{pred_data['proba_rupture'].mean():.3f}")

                st.plotly_chart(
                    px.line(pred_data, x="periode", y="proba_rupture", title="Probabilite de rupture predite a 4 semaines"),
                    width="stretch",
                )
                st.plotly_chart(
                    _line_figure(pred_data, "periode", ["stock_grossiste", "stock_pharmacie"], "Stocks principaux"),
                    width="stretch",
                )
                st.plotly_chart(
                    _line_figure(
                        pred_data,
                        "periode",
                        ["demande_patient", "demande_prevue", "demande_lissee_fabricant"],
                        "Demande et signal de production",
                    ),
                    width="stretch",
                )
                st.dataframe(
                    pred_data[
                        [
                            "periode",
                            "proba_rupture",
                            "prediction_rupture",
                            "rupture_grossiste",
                            "stock_grossiste",
                            "stock_pharmacie",
                        ]
                    ].head(30),
                    width="stretch",
                )

        with tab_manuelle:
            if st.session_state.artifact is None:
                st.info("Entrainer ou charger le modele pour tester une prediction manuelle.")
            else:
                st.write("Base les valeurs sur une ligne du dataset ou saisis un scenario.")
                if df is not None:
                    base_row = df.iloc[0]
                else:
                    base_row = pd.Series({col: 0 for col in FEATURE_COLUMNS})

                with st.form("manual_prediction"):
                    inputs = {}
                    columns = st.columns(2)
                    for idx, col_name in enumerate(FEATURE_COLUMNS):
                        default_value = base_row.get(col_name, 0)
                        target_col = columns[idx % 2]
                        with target_col:
                            if isinstance(default_value, bool) or col_name in {"disruption_fabricant", "rupture_grossiste", "rupture_pharmacie"}:
                                inputs[col_name] = st.checkbox(col_name, value=bool(default_value))
                            else:
                                step = 1.0
                                if col_name.startswith("taux_"):
                                    step = 0.01
                                inputs[col_name] = st.number_input(col_name, value=float(default_value), step=step)

                    submitted = st.form_submit_button("Predire")

                if submitted:
                    prediction = predict_single(inputs, st.session_state.artifact)
                    c1, c2 = st.columns(2)
                    c1.metric("Probabilite rupture", f"{prediction['proba_rupture']:.3f}")
                    c2.metric("Classe predite", int(prediction["prediction_rupture"]))
                    st.json(
                        {
                            "backend": st.session_state.artifact["backend"],
                            "probabilite": prediction["proba_rupture"],
                            "classe": prediction["prediction_rupture"],
                        }
                    )

    except Exception as exc:  # pragma: no cover - runtime diagnostics
        _log_exception("dashboard", exc)
        st.error("Le dashboard a rencontre une erreur inattendue.")
        st.code(traceback.format_exc())
