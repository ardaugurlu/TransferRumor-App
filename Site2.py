import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import base64
import plotly.express as px

# 1. Kod
@st.cache_data
def load_data():
    df = pd.read_excel("no_nans_data-2.xlsx", sheet_name="Sayfa1")
    df = df.drop_duplicates()
    df.columns = df.columns.str.upper()
    df[['CLUB', 'POSITION']] = df[['CLUB', 'POSITION']].applymap(lambda x: x.upper())
    return df

def ilgilenilebilecek_oyuncular(dataframe):
    takim = st.sidebar.selectbox("Team:", dataframe["CLUB"].unique())
    pozisyon = st.sidebar.selectbox("Position:", dataframe["POSITION"].unique())
    yas = st.sidebar.slider("Age Range:", min_value=16, max_value=40, key="yas_slider")
    deger = st.sidebar.slider("Value Range:", min_value=0, max_value=150000000,step = 100000, key="deger_slider")

    if st.sidebar.button("Get Predictions🔍"):
        if pozisyon == "ATTACK":
            attack = dataframe[dataframe['POSITION'] == 'ATTACK']
            X = attack.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1).values
        elif pozisyon == "MIDFIELD":
            midfield = dataframe[dataframe['POSITION'] == 'MIDFIELD']
            X = midfield.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1).values
        else:
            defence = dataframe[dataframe['POSITION'] == 'DEFENDER']
            X = defence.drop(["PLAYER", "CLUB", "NATION", "VALUE", "POSITION", "LEAGUE"], axis=1).values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(random_state=17)
        elbow = KElbowVisualizer(kmeans, k=(2, 20))
        elbow.fit(X_scaled)
        kmeans = KMeans(n_clusters=elbow.elbow_value_, random_state=17).fit(X_scaled)

        if pozisyon == "ATTACK":
            attack["CLUSTER"] = kmeans.labels_
            attack["CLUSTER"] = attack["CLUSTER"] + 1
            transfer_edilebilecekler = attack.loc[(attack["POSITION"] == pozisyon) & (attack["AGE"] <= yas) & (attack["VALUE"] <= deger) & (attack["CLUB"] != takim) & (attack["CLUSTER"] == round(attack.loc[attack["CLUB"] == takim, "CLUSTER"].mean()))]
        elif pozisyon == "MIDFIELD":
            midfield["CLUSTER"] = kmeans.labels_
            midfield["CLUSTER"] = midfield["CLUSTER"] + 1
            transfer_edilebilecekler = midfield.loc[(midfield["POSITION"] == pozisyon) & (midfield["AGE"] <= yas) & (midfield["VALUE"] <= deger) & (midfield["CLUB"] != takim) & (midfield["CLUSTER"] == round(midfield.loc[midfield["CLUB"] == takim, "CLUSTER"].mean()))]
        else:
            defence["CLUSTER"] = kmeans.labels_
            defence["CLUSTER"] = defence["CLUSTER"] + 1
            transfer_edilebilecekler = defence.loc[(defence["POSITION"] == pozisyon) & (defence["AGE"] <= yas) & (defence["VALUE"] <= deger) & (defence["CLUB"] != takim) & (defence["CLUSTER"] == round(defence.loc[defence["CLUB"] == takim, "CLUSTER"].mean()))]

        # Tahmin sonuçlarını ekrana yazdırın
        st.write(transfer_edilebilecekler[["PLAYER", "CLUB", "POSITION", "AGE", "VALUE"]])


# 2. Kod
def oyuncu_kazanc_beklentisi(dataframe):
    dataframe.columns = dataframe.columns.str.upper()
    dataframe[['PLAYER', 'CLUB', 'POSITION']] = dataframe[['PLAYER', 'CLUB', 'POSITION']].applymap(lambda x: x.upper())
    takim2 = st.sidebar.selectbox("Team: ", dataframe["CLUB"].unique()).upper()
    oyuncu_sonuc = dataframe[dataframe["CLUB"] == takim2][["PLAYER", "SALES_EXPECTATION","SEGMENT19_20","SEGMENT20_21","PERFORMANCE_SCORE_19/20","PERFORMANCE_SCORE_20_21","IMPROVEMENT SCORE"]]
    if st.sidebar.button("Get Predictions🔍"):
        st.write(oyuncu_sonuc)

# 3. Kod
def oyunculara_göre_aksiyon_tavsiyesi(dataframe):
    takim2 = st.sidebar.selectbox("Team: ", dataframe["Club"].unique())
    takim_df = dataframe.loc[(dataframe["Club"] == takim2), ["Player", "Club", "Age", "Position", "Recommend_For_Action"]]
    if st.sidebar.button("Get Predictions🔍"):
        st.write(takim_df)


# Ana uygulama
def main():
    new_title = '<p style="font-family:algerian; color:White; font-size: 55px;">TRANSFER RUMOR⚽️</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    selected_option = st.sidebar.radio("Choose Your Action:", ("Transfer Player Prediction", "Sales Expectation and Performance Analysis", "Recommendation for Action"))

    if selected_option == "Transfer Player Prediction":
        ilgilenilebilecek_oyuncular(load_data())
    elif selected_option == "Sales Expectation and Performance Analysis":
        df = pd.read_excel("no_nans_data-2.xlsx", sheet_name="Sayfa1")
        df2 = pd.read_excel("no_nans_data-2.xlsx", sheet_name="Sayfa2")
        df6 = pd.concat([df, df2], axis=1)
        oyuncu_kazanc_beklentisi(df6)
    else:
        df4 = pd.read_excel("Recommend_for_action.xlsx")
        df = pd.read_excel("no_nans_data-2.xlsx", sheet_name="Sayfa1")
        df2 = pd.read_excel("no_nans_data-2.xlsx", sheet_name="Sayfa2")
        df5 = pd.concat([df, df2], axis=1)
        df5 = df5.loc[:, ~df5.columns.duplicated()]
        oyunculara_göre_aksiyon_tavsiyesi(df5)


if __name__ == '__main__':
    main()

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://cdn.theatlantic.com/thumbor/U77N2yyvPFjHN6ayXkKBwKTivUM=/1x0:1999x1124/1600x900/media/img/2018/06/CULT_Dubois_soccer/original.jpg");
background-size: 110%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)