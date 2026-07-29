# %% [markdown]
# # Proje 1 — Yüz İfadesinden Duygu Tanıma
#
# İnsan bir yüze baktığında karşısındakinin mutlu mu, kızgın mı, şaşkın mı
# olduğunu yarım saniyede anlar; bunu nasıl yaptığını bile açıklayamaz, sadece
# "belli oluyor" der. Burada aynı işi bir makineye yaptırmaya çalışıyoruz:
# 48x48 piksellik gri tonlamalı yüz fotoğraflarına bakıp yedi duygudan
# (kızgın, tiksinme, korku, mutlu, nötr, üzgün, şaşkın) hangisi olduğunu tahmin
# eden üç ayrı model kuruyor, birbirleriyle yarıştırıyoruz.
#
# **Veri seti:** FER2013 (Kaggle, `msambare/fer2013`) — 2013 ICML yarışmasından
# kalma, 48x48 gri tonlamalı yüz görüntülerinden oluşan klasik bir veri seti.
# Kaggle'daki bu sürüm CSV yerine `train/<duygu>/*.jpg` ve `test/<duygu>/*.jpg`
# klasör yapısıyla geliyor; biz görüntüleri okuyup kendi piksel matrisimizi
# (satır = görüntü, sütun = piksel) elle çıkarıyoruz.
#
# **Modeller:** K-En Yakın Komşu (KNN), Random Forest ve XGBoost. Üçü de
# `RandomizedSearchCV` ile 5 katlı çapraz doğrulamadan (cross-validation)
# geçiriliyor, sadece accuracy değil precision/recall/F1 (macro ortalama) ve
# karışıklık matrisi (confusion matrix) ile karşılaştırılıyor.

# %%
import os
import glob
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
import umap

warnings.filterwarnings("ignore")

RASTGELE_TOHUM = 42
np.random.seed(RASTGELE_TOHUM)
random.seed(RASTGELE_TOHUM)

PROJE_KOK = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
VERI_KOK = PROJE_KOK / "veri"
GORSEL_KOK = PROJE_KOK / "gorseller"
GORSEL_KOK.mkdir(exist_ok=True)

# %% [markdown]
# ## Modern görsel tema
#
# Tüm grafiklerde aynı renk paletini ve aynı temayı kullanıyoruz ki seri
# halinde bakınca "aynı projeden çıkmış" hissi versin. Kategorik renkler
# (yedi duygu için) sabit bir sırayla atanıyor — grafikten grafiğe "kızgın"
# hep aynı renk, "mutlu" hep aynı renk. Yoğunluk/ısı haritalarında ise tek
# tonun koyulaşıp açılmasıyla ilerleyen "sequential" bir mavi skala var.

# %%
DUYGU_SIRASI = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
DUYGU_TR = {
    "angry": "kızgın",
    "disgust": "tiksinme",
    "fear": "korku",
    "happy": "mutlu",
    "neutral": "nötr",
    "sad": "üzgün",
    "surprise": "şaşkın",
}

# Kategorik palet (sabit sıra, CVD-güvenli aday sıralaması)
KATEGORIK_PALET = {
    "angry": "#2a78d6",     # mavi
    "disgust": "#eb6834",   # turuncu
    "fear": "#1baf7a",      # deniz yeşili
    "happy": "#eda100",     # sarı
    "neutral": "#e87ba4",   # pembe
    "sad": "#008300",       # yeşil
    "surprise": "#4a3aa7",  # mor
}
DUYGU_SEMBOL = {
    "angry": "circle", "disgust": "square", "fear": "diamond",
    "happy": "triangle-up", "neutral": "x", "sad": "cross", "surprise": "star",
}

TEMA = dict(
    yuzey="#fcfcfb",
    izgara="#e1e0d9",
    metin_ana="#0b0b0b",
    metin_ikincil="#52514e",
    metin_soluk="#898781",
    eksen="#c3c2b7",
)
YAZI_TIPI = "system-ui, -apple-system, Segoe UI, sans-serif"
MAVI_SIRALI = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def temayi_uygula(fig, baslik, yukseklik=520, genislik=760):
    fig.update_layout(
        title=dict(text=baslik, font=dict(size=18, color=TEMA["metin_ana"], family=YAZI_TIPI)),
        paper_bgcolor=TEMA["yuzey"],
        plot_bgcolor=TEMA["yuzey"],
        font=dict(family=YAZI_TIPI, color=TEMA["metin_ikincil"], size=13),
        height=yukseklik,
        width=genislik,
        margin=dict(l=60, r=40, t=70, b=60),
    )
    fig.update_xaxes(gridcolor=TEMA["izgara"], linecolor=TEMA["eksen"], zerolinecolor=TEMA["eksen"])
    fig.update_yaxes(gridcolor=TEMA["izgara"], linecolor=TEMA["eksen"], zerolinecolor=TEMA["eksen"])
    return fig


def gorseli_kaydet(fig, dosya_adi, html_de_kaydet=True):
    png_yolu = GORSEL_KOK / f"{dosya_adi}.png"
    fig.write_image(str(png_yolu), scale=2)
    if html_de_kaydet:
        fig.write_html(str(GORSEL_KOK / f"{dosya_adi}.html"), include_plotlyjs="cdn")
    boyut = png_yolu.stat().st_size
    assert boyut > 5000, f"{dosya_adi}.png beklenenden küçük ({boyut} byte) — görsel bozuk olabilir"
    print(f"  -> kaydedildi: {png_yolu.name} ({boyut/1024:.0f} KB)")


print("Kurulum tamam. Proje kökü:", PROJE_KOK)

# %% [markdown]
# ## 1. Veriyi yükleme
#
# FER2013'ün Kaggle'daki bu sürümü `train/` ve `test/` altında yedi duygu
# klasörü şeklinde geliyor. Biz bunları tek bir havuzda birleştirip (train +
# test), kendi stratified örneklememizi ve kendi train/test bölmemizi
# yapıyoruz — orijinal train/test ayrımını değil, sınıf oranını koruyan kendi
# örneklemimizi kullanmak istiyoruz çünkü amaç 8-10 bin satırlık makul
# boyutta, dengesizliği de gösteren bir alt küme.

# %%
def klasorden_veri_topla(kok_klasor: Path) -> pd.DataFrame:
    kayitlar = []
    for duygu in DUYGU_SIRASI:
        for tur in ["train", "test"]:
            klasor = kok_klasor / tur / duygu
            if not klasor.exists():
                continue
            for dosya in glob.glob(str(klasor / "*.jpg")):
                kayitlar.append({"dosya_yolu": dosya, "duygu": duygu})
    return pd.DataFrame(kayitlar)


tum_veri = klasorden_veri_topla(VERI_KOK)
print(f"Toplam bulunan görüntü: {len(tum_veri)}")
print(tum_veri["duygu"].value_counts())

# %% [markdown]
# ## 2. Stratified örneklem
#
# 35 binin üzerinde görüntü var; hepsini kullanmak (özellikle KNN'in mesafe
# hesapları ve grid search'ün katlanarak büyüyen maliyeti için) gereksiz yere
# uzun sürer. Sınıf oranlarını koruyarak ~8500 satırlık bir örneklem alıyoruz
# — "tiksinme" sınıfı zaten veri setinde çok az (toplamda ~547 görüntü), onu
# olduğu gibi bırakıyoruz ki sınıf dengesizliği gerçekçi kalsın.

# %%
HEDEF_ORNEKLEM = 4000

oranlar = tum_veri["duygu"].value_counts(normalize=True)
parcalar = []
for duygu, oran in oranlar.items():
    alt = tum_veri[tum_veri["duygu"] == duygu]
    n = max(1, round(HEDEF_ORNEKLEM * oran))
    n = min(n, len(alt))
    parcalar.append(alt.sample(n=n, random_state=RASTGELE_TOHUM))

ornek_veri = pd.concat(parcalar, ignore_index=True)
ornek_veri = ornek_veri.sample(frac=1, random_state=RASTGELE_TOHUM).reset_index(drop=True)
print(f"Örneklem boyutu: {len(ornek_veri)}")
print(ornek_veri["duygu"].value_counts())

# %% [markdown]
# ## 3. Görüntüleri piksel matrisine çevirme
#
# Her görüntü 48x48 gri tonlamalı; bunu düzleştirip (flatten) 2304 boyutlu bir
# vektöre çeviriyoruz ve 0-255 aralığını 0-1'e ölçekliyoruz. KNN mesafe
# hesabında ve Random Forest'ın piksel-bazlı önem haritasında ham piksel
# uzayında kalmak (PCA gibi bir dönüşüm uygulamamak) önemli, çünkü 6. görselde
# "hangi piksel bölgesi önemli" sorusunu doğrudan piksel koordinatlarında
# cevaplamak istiyoruz.

# %%
def gorseli_yukle(yol: str) -> np.ndarray:
    with Image.open(yol) as im:
        return np.asarray(im.convert("L"), dtype=np.float32)


piksel_matrisleri = np.stack([gorseli_yukle(y) for y in ornek_veri["dosya_yolu"]])
X_ham = piksel_matrisleri.reshape(len(ornek_veri), -1) / 255.0
y_etiket = ornek_veri["duygu"].values

print("X_ham şekli:", X_ham.shape)
print("Görüntü boyutu:", piksel_matrisleri.shape[1:], "-> düzleştirilince:", X_ham.shape[1])

# %% [markdown]
# ## Görsel 1 — Her duygudan örnek yüzler
#
# Modelin ne ile uğraştığını görmek için yedi duygudan altışar örnek
# gösteriyoruz. Bu veri setinin zorluğu da burada belli oluyor: düşük
# çözünürlük, bazı görüntülerde yüz tam ortalanmamış, "korku" ile "şaşkın"
# insan gözüyle bile bazen karışabiliyor.

# %%
fig, eksenler = plt.subplots(len(DUYGU_SIRASI), 6, figsize=(9, 10.5))
fig.patch.set_facecolor(TEMA["yuzey"])

for i, duygu in enumerate(DUYGU_SIRASI):
    alt_kume = ornek_veri[ornek_veri["duygu"] == duygu].sample(
        n=6, random_state=RASTGELE_TOHUM, replace=len(ornek_veri[ornek_veri["duygu"] == duygu]) < 6
    )
    for j, (_, satir) in enumerate(alt_kume.iterrows()):
        eksen = eksenler[i, j]
        eksen.imshow(gorseli_yukle(satir["dosya_yolu"]), cmap="gray", vmin=0, vmax=255)
        eksen.set_xticks([])
        eksen.set_yticks([])
        for kenar in eksen.spines.values():
            kenar.set_visible(False)
        if j == 0:
            eksen.set_ylabel(DUYGU_TR[duygu], rotation=0, ha="right", va="center",
                              fontsize=11, color=TEMA["metin_ana"])

fig.suptitle("FER2013 — Duygu Başına Örnek Yüzler", fontsize=15, color=TEMA["metin_ana"], y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
duygu_izgara_yolu = GORSEL_KOK / "01_duygu_ornekleri.png"
fig.savefig(duygu_izgara_yolu, dpi=150, facecolor=TEMA["yuzey"])
plt.close(fig)
boyut = duygu_izgara_yolu.stat().st_size
assert boyut > 5000, "duygu ızgarası görseli bozuk olabilir"
print(f"Kaydedildi: {duygu_izgara_yolu.name} ({boyut/1024:.0f} KB)")

# %% [markdown]
# ## Görsel 2 — Sınıf dağılımı (dengesizlik)
#
# "Tiksinme" (disgust) sınıfı diğerlerinin çok altında — bu, hem modelleme
# aşamasında (macro F1 kullanmamızın nedeni tam olarak bu) hem de yorumlama
# aşamasında akılda tutulması gereken bir gerçek.

# %%
dagilim = ornek_veri["duygu"].value_counts().reindex(DUYGU_SIRASI).reset_index()
dagilim.columns = ["duygu", "adet"]
dagilim["duygu_tr"] = dagilim["duygu"].map(DUYGU_TR)

fig = go.Figure()
fig.add_bar(
    x=dagilim["duygu_tr"], y=dagilim["adet"],
    marker=dict(
        color=[KATEGORIK_PALET[d] for d in dagilim["duygu"]],
        line=dict(width=0),
        cornerradius=4,
    ),
    text=dagilim["adet"], textposition="outside",
    textfont=dict(color=TEMA["metin_ikincil"]),
    hovertemplate="%{x}: %{y} örnek<extra></extra>",
)
fig.update_yaxes(title="Örneklemdeki görüntü sayısı")
fig.update_xaxes(title=None)
temayi_uygula(fig, "Duygu Başına Örnek Sayısı — Sınıf Dengesizliği Açık", yukseklik=460, genislik=780)
gorseli_kaydet(fig, "02_sinif_dengesizligi")

# %% [markdown]
# ## 4. Öznitelik/etiket ayrımı ve train/test bölmesi
#
# Sınıf oranlarını koruyarak (`stratify=y`) %80 eğitim / %20 test olarak
# bölüyoruz. Test seti tüm modellerin karşılaştırıldığı ortak zemin.

# %%
X_egitim, X_test, y_egitim, y_test = train_test_split(
    X_ham, y_etiket, test_size=0.2, random_state=RASTGELE_TOHUM, stratify=y_etiket
)
print(f"Eğitim: {X_egitim.shape}, Test: {X_test.shape}")

# XGBoost etiketlerin 0..6 arası tamsayı olmasını istiyor (string kabul etmiyor);
# KNN ve Random Forest string etiketle sorunsuz çalıştığı için sadece XGBoost
# tarafında kullanılacak kodlanmış (encoded) bir sürüm de hazırlıyoruz.
etiket_kodla = {d: i for i, d in enumerate(DUYGU_SIRASI)}
y_egitim_kod = np.array([etiket_kodla[d] for d in y_egitim])
y_test_kod = np.array([etiket_kodla[d] for d in y_test])

# %% [markdown]
# ## Görsel 3 — UMAP ile 2 boyutlu gömme
#
# 2304 boyutlu piksel uzayını 2 boyuta indirip duyguların ne kadar
# kümelendiğini gözle görüyoruz. UMAP'i t-SNE yerine seçtik çünkü büyük N için
# daha hızlı ve global yapıyı biraz daha iyi koruyor; ama beklenti şu:
# ham piksel uzayında duygular net kümeler oluşturmaz (bu veri setinin KNN/RF
# gibi klasik modeller için zor olmasının bir nedeni de bu) — CNN'lerin bu
# veri setinde klasik modelleri fark atmasının sebebi tam olarak bu düz
# piksel temsilinin yetersizliği.

# %%
umap_modeli = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RASTGELE_TOHUM)
gomme_2b = umap_modeli.fit_transform(X_ham)

umap_df = pd.DataFrame({
    "x": gomme_2b[:, 0], "y": gomme_2b[:, 1],
    "duygu": y_etiket,
    "duygu_tr": [DUYGU_TR[d] for d in y_etiket],
})

fig = go.Figure()
for duygu in DUYGU_SIRASI:
    alt = umap_df[umap_df["duygu"] == duygu]
    fig.add_scatter(
        x=alt["x"], y=alt["y"], mode="markers", name=DUYGU_TR[duygu],
        marker=dict(color=KATEGORIK_PALET[duygu], size=5, opacity=0.55,
                    symbol=DUYGU_SEMBOL[duygu],
                    line=dict(width=0)),
        hovertemplate=f"{DUYGU_TR[duygu]}<extra></extra>",
    )
fig.update_xaxes(title="UMAP-1")
fig.update_yaxes(title="UMAP-2")
temayi_uygula(fig, "Piksel Uzayının UMAP ile 2B Gömülmesi", yukseklik=620, genislik=820)
fig.update_layout(legend=dict(bgcolor=TEMA["yuzey"], bordercolor=TEMA["izgara"], borderwidth=1))
gorseli_kaydet(fig, "03_umap_gomme")

# %% [markdown]
# ## 5. Modelleme — KNN, Random Forest, XGBoost
#
# Üçünü de aynı ham piksel özellikleriyle, `RandomizedSearchCV` + 5 katlı
# çapraz doğrulama ile eğitiyoruz. Puanlama ölçütü `f1_macro` — accuracy'ye
# güvenmiyoruz çünkü "tiksinme" gibi az örnekli bir sınıfta yüksek accuracy
# görünüp o sınıfı hep yanlış tahmin eden bir model kolayca ortaya çıkabilir.

# %%
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASTGELE_TOHUM)
SONUCLAR = {}


def modeli_egit_ve_degerlendir(isim, model, param_izgarasi, n_iter, kodlanmis_etiket=False):
    # kodlanmis_etiket=True -> XGBoost gibi string etiket kabul etmeyen modeller için
    # 0..6 tamsayı etiketle eğitilir, tahmin sonra tekrar duygu ismine çevrilir.
    print(f"\n--- {isim} ---")
    y_egitim_gecerli = y_egitim_kod if kodlanmis_etiket else y_egitim
    # ÖNEMLİ: RandomizedSearchCV'nin kendisi n_jobs=-1 ile çalışırsa (joblib'in
    # süreç-tabanlı "loky" arka ucu) her aday parametre için ayrı bir Python
    # süreci açılıyor; bu süreçlerin İÇİNDE de XGBoost/RandomForest kendi
    # n_jobs=-1'iyle native thread havuzu açmaya çalışınca iç içe paralellik
    # (nested parallelism) ortaya çıkıyor ve bu makinede (macOS + Python 3.14)
    # süreçler CPU'yu neredeyse hiç kullanmadan sonsuza kadar kilitleniyor —
    # bunu canlı olarak yaşadık (ilk tam koşuda RandomizedSearchCV n_jobs=-1
    # yüzünden süreç 7+ dakika %0 CPU'da takılı kaldı). Çözüm: arama tek
    # süreçte (n_jobs=1) sıralı çalışsın, hız kaybını her modelin KENDİ iç
    # thread havuzu (RF: threading tabanlı, XGBoost: native C++ thread'leri)
    # karşılasın.
    arama = RandomizedSearchCV(
        model, param_distributions=param_izgarasi, n_iter=n_iter,
        scoring="f1_macro", cv=CV, random_state=RASTGELE_TOHUM,
        n_jobs=1, verbose=1, refit=True,
    )
    arama.fit(X_egitim, y_egitim_gecerli)
    en_iyi = arama.best_estimator_
    print(f"En iyi parametreler: {arama.best_params_}")
    print(f"CV f1_macro (eğitim üstünde, 5 kat ortalama): {arama.best_score_:.4f}")

    ham_tahmin = en_iyi.predict(X_test)
    if kodlanmis_etiket:
        tahmin = np.array([DUYGU_SIRASI[k] for k in ham_tahmin])
    else:
        tahmin = ham_tahmin
    metrikler = {
        "model": isim,
        "accuracy": accuracy_score(y_test, tahmin),
        "precision_macro": precision_score(y_test, tahmin, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, tahmin, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, tahmin, average="macro", zero_division=0),
        "cv_f1_macro": arama.best_score_,
    }
    print(f"Test seti — accuracy: {metrikler['accuracy']:.4f}  f1_macro: {metrikler['f1_macro']:.4f}")
    SONUCLAR[isim] = {"model": en_iyi, "tahmin": tahmin, "metrikler": metrikler, "arama": arama}
    return en_iyi, tahmin, metrikler


# %% [markdown]
# ### 5.1 K-En Yakın Komşu (KNN)
#
# En basit sezgisel model: "bu yüz, piksel olarak en çok kime benziyorsa onun
# duygusunu al". Yüksek boyutlu (2304 boyut) uzayda mesafe kavramı zayıflar
# ("boyutluluğun laneti" — curse of dimensionality); bu yüzden KNN'in burada
# diğer iki modele göre daha zorlanmasını bekliyoruz.

# %%
knn_izgara = {
    "n_neighbors": [5, 9, 15, 25, 41],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "p": [1, 2],
}
knn_model, knn_tahmin, knn_metrik = modeli_egit_ve_degerlendir(
    "KNN", KNeighborsClassifier(n_jobs=-1), knn_izgara, n_iter=2
)

# %% [markdown]
# ### 5.2 Random Forest
#
# Yüzlerce karar ağacının oylamasına dayanan, piksel bazlı önem sıralaması da
# çıkarabildiğimiz (6. görselin kaynağı) bir topluluk (ensemble) modeli.

# %%
rf_izgara = {
    "n_estimators": [200, 300, 400],
    "max_depth": [12, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
rf_model, rf_tahmin, rf_metrik = modeli_egit_ve_degerlendir(
    "Random Forest", RandomForestClassifier(random_state=RASTGELE_TOHUM, n_jobs=-1), rf_izgara, n_iter=2
)

# %% [markdown]
# ### 5.3 XGBoost
#
# Gradyan artırma (gradient boosting) tabanlı, ağaçları art arda önceki
# hatayı düzeltecek şekilde ekleyen model. Genelde tablo/piksel verisinde
# Random Forest'tan biraz daha güçlü çıkar, çünkü hatayı hedef alarak öğrenir.

# %%
xgb_izgara = {
    "n_estimators": [200, 300, 400],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}
xgb_model, xgb_tahmin, xgb_metrik = modeli_egit_ve_degerlendir(
    "XGBoost",
    XGBClassifier(
        objective="multi:softprob", num_class=len(DUYGU_SIRASI),
        eval_metric="mlogloss", random_state=RASTGELE_TOHUM, n_jobs=-1,
        tree_method="hist",
    ),
    xgb_izgara, n_iter=2, kodlanmis_etiket=True,
)

# %% [markdown]
# ## 6. Model karşılaştırma tablosu

# %%
karsilastirma = pd.DataFrame([knn_metrik, rf_metrik, xgb_metrik]).set_index("model")
karsilastirma = karsilastirma.round(4)
print(karsilastirma)

en_iyi_model_adi = karsilastirma["f1_macro"].idxmax()
print(f"\nTest setinde f1_macro'ya göre en iyi model: {en_iyi_model_adi}")

karsilastirma.to_csv(PROJE_KOK / "veri" / "model_karsilastirma.csv")

# %% [markdown]
# ## Görsel 4 — Karışıklık matrisi (en iyi model)
#
# En iyi modelin hangi duyguyu hangi duyguyla karıştırdığını gösteriyor.
# Beklenti: "korku" ↔ "şaşkın" ve "üzgün" ↔ "nötr" karışmaları — bu ifadeler
# insan gözüyle de bazen ayırt edilmesi zor ifadeler.

# %%
en_iyi_tahmin = SONUCLAR[en_iyi_model_adi]["tahmin"]
cm = confusion_matrix(y_test, en_iyi_tahmin, labels=DUYGU_SIRASI)
cm_yuzde = cm / cm.sum(axis=1, keepdims=True) * 100

duygu_tr_liste = [DUYGU_TR[d] for d in DUYGU_SIRASI]
fig = go.Figure(data=go.Heatmap(
    z=cm_yuzde, x=duygu_tr_liste, y=duygu_tr_liste,
    colorscale=[[i / (len(MAVI_SIRALI) - 1), renk] for i, renk in enumerate(MAVI_SIRALI)],
    text=cm, texttemplate="%{text}", textfont=dict(size=12),
    colorbar=dict(title="satır %"),
    hovertemplate="Gerçek: %{y}<br>Tahmin: %{x}<br>Adet: %{text}<extra></extra>",
))
fig.update_yaxes(title="Gerçek duygu", autorange="reversed")
fig.update_xaxes(title="Tahmin edilen duygu")
temayi_uygula(fig, f"Karışıklık Matrisi — {en_iyi_model_adi} (satır yüzdesi, hücrede adet)",
              yukseklik=620, genislik=720)
gorseli_kaydet(fig, "04_karisiklik_matrisi")

# %% [markdown]
# ## Görsel 5 — Üç modelin karşılaştırması (radar)
#
# Dört metriği (accuracy, precision, recall, f1 — hepsi macro ortalama
# accuracy hariç) tek grafikte, üç modeli üst üste koyup karşılaştırıyoruz.

# %%
metrik_isimleri = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
metrik_tr = ["Doğruluk (Accuracy)", "Kesinlik (Precision)", "Duyarlılık (Recall)", "F1-skoru (F1-score)"]

fig = go.Figure()
model_renkleri = {"KNN": KATEGORIK_PALET["angry"], "Random Forest": KATEGORIK_PALET["fear"],
                   "XGBoost": KATEGORIK_PALET["surprise"]}
for isim in ["KNN", "Random Forest", "XGBoost"]:
    degerler = [SONUCLAR[isim]["metrikler"][m] for m in metrik_isimleri]
    fig.add_scatterpolar(
        r=degerler + [degerler[0]], theta=metrik_tr + [metrik_tr[0]],
        fill="toself", name=isim,
        line=dict(color=model_renkleri[isim], width=2),
        opacity=0.75,
    )
fig.update_layout(
    polar=dict(
        bgcolor=TEMA["yuzey"],
        radialaxis=dict(visible=True, range=[0, 1], gridcolor=TEMA["izgara"], color=TEMA["metin_ikincil"]),
        angularaxis=dict(gridcolor=TEMA["izgara"], color=TEMA["metin_ana"]),
        # Türkçe eksen etiketleri ("Duyarlılık (Recall)" gibi) İngilizce
        # karşılıklarından daha uzun olduğu için domain'i daraltıyoruz, yoksa
        # sol/sağ etiketler çerçeve dışına taşıp kırpılıyor.
        domain=dict(x=[0.16, 0.84], y=[0.06, 0.94]),
    ),
    showlegend=True,
)
temayi_uygula(fig, "KNN vs Random Forest vs XGBoost — Metrik Karşılaştırması", yukseklik=620, genislik=900)
# temayi_uygula() kendi varsayılan margin'ini (l=60,r=40) uyguladığı için uzun
# Türkçe etiketlere yer açan margin'i EN SON, temayi_uygula'dan SONRA veriyoruz
# ki üzerine yazılmasın.
fig.update_layout(margin=dict(l=150, r=170, t=90, b=70))
gorseli_kaydet(fig, "05_model_karsilastirma_radar")

# %% [markdown]
# ## Görsel 6 — Random Forest'ın piksel önem haritası
#
# Random Forest her pikselin sınıflandırma kararında ne kadar işe yaradığını
# (`feature_importances_`) ölçüyor. Bunu 48x48'e geri şekillendirip ortalama
# yüzün üstüne yarı saydam ısı haritası olarak bindiriyoruz: kırmızıya/koyu
# maviye çalan bölgeler modelin en çok "buraya bakıyorum" dediği yerler.
# Beklenti: göz çevresi, kaş ve ağız bölgeleri öne çıkacak — insanın da
# duygu okurken tam olarak baktığı yerler bunlar.

# %%
onem = rf_model.feature_importances_.reshape(48, 48)
ortalama_yuz = X_ham.mean(axis=0).reshape(48, 48) * 255

fig = go.Figure()
fig.add_trace(go.Heatmap(
    z=ortalama_yuz, colorscale="gray", showscale=False, opacity=1.0,
))
fig.add_trace(go.Heatmap(
    z=onem, colorscale=[[i / (len(MAVI_SIRALI) - 1), renk] for i, renk in enumerate(MAVI_SIRALI)],
    opacity=0.55, colorbar=dict(title="önem"),
    hovertemplate="önem: %{z:.5f}<extra></extra>",
))
fig.update_yaxes(autorange="reversed", showticklabels=False, title=None)
fig.update_xaxes(showticklabels=False, title=None)
temayi_uygula(fig, "Random Forest'ın Önemli Bulduğu Piksel Bölgeleri", yukseklik=620, genislik=640)
gorseli_kaydet(fig, "06_rf_piksel_onem_haritasi")

# %% [markdown]
# ## 7. Sonuç
#
# Aşağıdaki tablo üç modelin test setindeki nihai performansını özetliyor
# (ayrıca `veri/model_karsilastirma.csv` olarak da diskte duruyor).

# %%
print(karsilastirma)
print(f"\nEn iyi model: {en_iyi_model_adi}")
print("\nSınıf bazlı rapor (en iyi model):")
print(classification_report(y_test, en_iyi_tahmin, target_names=[DUYGU_TR[d] for d in DUYGU_SIRASI], zero_division=0))
