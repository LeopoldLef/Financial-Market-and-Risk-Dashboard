import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# Configuration de la page Streamlit (mode large)
st.set_page_config(page_title="Dashboard Financier", page_icon="📈", layout="wide")

st.title("📊 Tableau de Bord Financier Interactif")
st.markdown("Analysez la performance, le risque et la corrélation de vos actifs boursiers en temps réel.")

# --- BARRE LATÉRALE (SIDEBAR) ---
st.sidebar.header("Paramètres de l'analyse")

# Saisie des tickers sous forme de texte séparé par des virgules
tickers_input = st.sidebar.text_input(
    "Actifs (séparés par des virgules)", "AAPL, MSFT, AMZN, TTE.PA, ^GSPC"
)
# Nettoyage de la saisie utilisateur (on enlève les espaces superflus)
tickers = [t.strip().upper() for t in tickers_input.split(",")]

# Sélection des dates de début et de fin
start_date = st.sidebar.date_input("Date de début", value=pd.to_datetime("2005-01-01"))
end_date = st.sidebar.date_input("Date de fin", value=pd.to_datetime("2026-01-01"))

# Bouton pour lancer les calculs
run_btn = st.sidebar.button("Lancer l'analyse")

# --- EXECUTION LORSAGE DU CLIC ---
if run_btn:
    with st.spinner("Téléchargement des données boursières en cours..."):
        # Téléchargement des prix de clôture ajustés
        data = yf.download(tickers, start=start_date, end=end_date)["Close"]

    if data.empty or len(data) == 0:
        st.error("⚠️ Aucune donnée trouvée. Vérifie l'orthographe de tes tickers.")
    else:
        # 1. Rendements journaliers
        daily_returns = data.pct_change(fill_method=None).dropna()

        # 2. Rendements cumulés
        cumulative_returns = (1 + daily_returns).cumprod() - 1

        # 3. Volatilité annualisée
        annualized_volatility = daily_returns.std() * np.sqrt(252)

        # 4. Maximum Drawdown
        rolling_max = (1 + daily_returns).cumprod().cummax()
        drawdown = (1 + daily_returns).cumprod() / rolling_max - 1
        max_drawdown = drawdown.min()

        # 5. Ratio de Sharpe (taux sans risque = 0)
        annualized_return = daily_returns.mean() * 252
        sharpe_ratio = annualized_return / annualized_volatility

        # 6. Matrice de corrélation
        correlation_matrix = daily_returns.corr()

        # --- AFFICHAGE SUR LE DASHBOARD ---
        
        # Graphique des rendements cumulés avec Plotly
        st.subheader("📈 Évolution des Rendements Cumulés")
        fig_cumul = px.line(
            cumulative_returns,
            title="Performance cumulée des actifs sur la période",
            labels={"value": "Rendement cumulé", "Date": "Date", "variable": "Actif"},
        )
        st.plotly_chart(fig_cumul, use_container_width=True)

        # Tableau synthétique des métriques
        st.subheader("📋 Tableau Synthétique des Performances et Risques")
        summary_df = pd.DataFrame(
            {
                "Rendement Annualisé": annualized_return,
                "Volatilité Annualisée": annualized_volatility,
                "Ratio de Sharpe": sharpe_ratio,
                "Max Drawdown": max_drawdown,
            }
        )
        # Affichage du tableau formaté joliment
        st.dataframe(summary_df.style.format("{:.2%}"), use_container_width=True)

        # Matrice de corrélation
        st.subheader("🔥 Matrice de Corrélation entre les Actifs")
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Carte de chaleur (Heatmap) de la corrélation",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("👈 Renseigne tes actifs et clique sur **Lancer l'analyse** dans la barre latérale pour afficher le tableau de bord.")