# Yüz İfadesinden Duygu Tanıma

İnsan bir yüze baktığında karşısındakinin mutlu mu, kızgın mı, şaşkın mı olduğunu yarım saniyede anlar, üstelik bunu nasıl yaptığını kendisi de açıklayamaz. Burada aynı işi klasik makine öğrenmesiyle yapmaya çalıştık: 48x48 piksellik gri tonlamalı yüz fotoğraflarına bakıp yedi duygudan (kızgın, tiksinme, korku, mutlu, nötr, üzgün, şaşkın) hangisi olduğunu tahmin eden üç model kurduk.

## Veri seti

FER2013 (Kaggle, `msambare/fer2013`) kullanıldı. 2013 ICML yarışmasından kalma, 48x48 gri tonlamalı yüz görüntülerinden oluşan klasik bir veri seti. Bu sürüm CSV değil, `train/<duygu>/*.jpg` klasör yapısıyla geliyor; görüntüleri okuyup piksel matrisine kendimiz çevirdik.

Eğitim klasöründe 28.709 görüntü var. Hepsini kullanmak KNN'in mesafe hesaplarını ve grid search'ü gereksiz yere uzatacağı için, sınıf oranlarını koruyarak 4.000'lik bir örneklem aldık (3.200 eğitim, 800 test). "Tiksinme" sınıfı veri setinde zaten çok az (toplamda 547 görüntü, örneklemde 61), bunu yapay olarak büyütmedik ki gerçek dengesizlik korunsun.

## Modeller

KNN, Random Forest ve üçüncü model olarak XGBoost eklendi. Üçü de `RandomizedSearchCV` (n_iter=15-20) ile 5 katlı çapraz doğrulamadan geçirildi. Skorlama accuracy değil `f1_macro`, çünkü yedi sınıf dengesiz ve "hep mutlu de" diyen tembel bir model bile makul bir accuracy'ye ulaşabilir.

## Sonuçlar (test seti, 800 görüntü)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | CV F1 (macro) |
|---|---|---|---|---|---|
| KNN | 0.281 | 0.231 | 0.202 | 0.185 | 0.190 |
| Random Forest | 0.343 | 0.284 | 0.269 | 0.257 | 0.256 |
| **XGBoost** | **0.360** | 0.290 | 0.286 | **0.280** | 0.283 |

Dürüst olmak gerekirse bu sonuçlar mütevazı. En iyi model (XGBoost) sınıf başına ortalama %28 F1 alıyor, "tiksinme" sınıfında ise 12 test örneğinin hiçbirini doğru yakalayamadı (precision/recall 0.00): hem örnek sayısı az hem de bu duygu diğerleriyle piksel düzeyinde çok karışıyor. "Mutlu" sınıfında iş değişiyor, F1 0.54'e çıkıyor, çünkü gülümseme diğer ifadelere göre piksel deseninde daha ayırt edici.

Bunun nedeni veri kalitesi değil, yöntem seçimi: görüntüyü 2304 tek boyutlu sayıya düzleştirdiğimizde komşuluk, kenar, doku gibi uzamsal bilgiyi tamamen kaybediyoruz. KNN/Random Forest/XGBoost bu düzleştirilmiş vektörler üzerinde çalışıyor, oysa bir CNN aynı görüntüleri 2 boyutlu yapısını koruyarak işler ve bu veri setinde çok daha iyi sonuç verir (literatürde CNN'lerle %65-70 accuracy raporlanmış). Bu proje kapsamında istenen algoritmalar klasik ML olduğu için tavan burada; sonucu süslemek yerine nedenini açıklamayı tercih ettik.

## Görsellerden ikisi

`gorseller/03_umap_gomme.png` 4000 görüntüyü UMAP ile 2 boyuta indirip duyguya göre renklendiriyor: net kümeler yok, bulutlar birbirine geçmiş durumda. Modelin neden zorlandığını görsel olarak da doğruluyor.

`gorseller/06_rf_piksel_onem_haritasi.png` Random Forest'ın hangi piksellere en çok baktığını ortalama yüzün üstüne ısı haritası olarak bindiriyor: göz çevresi ve ağız bölgesi öne çıkıyor. Bu en azından insanın da duygu okurken baktığı yerlerle örtüşüyor.

## Dosyalar

- `proje.py`: çalıştırılabilir ana script (jupytext light format)
- `proje.ipynb`: `jupyter nbconvert --execute` ile baştan sona çalıştırılmış, çıktıları diskte duran defter
- `gorseller/`: 6 görsel (duygu örnekleri ızgarası, sınıf dengesizliği, UMAP gömme, karışıklık matrisi, model karşılaştırma radarı, RF piksel önem haritası). Plotly üretimleri hem `.png` hem `.html`

## Kullanılan kütüphaneler

- [scikit-learn](https://scikit-learn.org/stable/), KNN, Random Forest, `RandomizedSearchCV`, metrikler
- [XGBoost](https://xgboost.readthedocs.io/), gradyan artırımlı ağaç modeli
- [UMAP](https://umap-learn.readthedocs.io/), 2 boyutlu gömme görselleştirmesi
- [Pillow](https://pillow.readthedocs.io/), görüntü okuma
- [pandas](https://pandas.pydata.org/docs/) / [numpy](https://numpy.org/doc/), veri işleme
- [Plotly](https://plotly.com/python/), tüm görselleştirmeler
- [Kaleido](https://github.com/plotly/Kaleido), Plotly grafiklerini PNG'ye aktarma
- [Jupytext](https://jupytext.readthedocs.io/) / [nbconvert](https://nbconvert.readthedocs.io/), script/notebook dönüşümü ve çalıştırma
- [Kaggle CLI](https://github.com/Kaggle/kaggle-api), veri setini indirmek için

## Notlar

- Örneklem boyutu (4000) bilinçli bir hız/kalite dengesi: bu makinede birkaç proje paralel çalıştığı için tüm 28.709 görüntüyle grid search saatlerce sürerdi. Daha büyük örneklemle sonuçlar bir miktar iyileşir ama yöntem tavanı (düzleştirilmiş piksel) değişmez.
- `RandomizedSearchCV`'nin kendisi `n_jobs=1` (tek süreçte sıralı) çalıştırıldı, ama her modelin kendi kurucusunda (`KNeighborsClassifier`, `RandomForestClassifier`, `XGBClassifier`) `n_jobs=-1` var. Sebep: arama katmanı da modelin kendisi de aynı anda `n_jobs=-1` isteseydi iç içe paralellik (nested parallelism) oluşup joblib'in süreç havuzları birbirini kilitliyordu; bu proje geliştirilirken bu deadlock canlı olarak yaşandı ve arama katmanı sıralı, model içi paralellik açık bırakılarak çözüldü.
