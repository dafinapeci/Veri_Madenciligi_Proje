# Teknoloji Çalışanlarında Ruh Sağlığı Tedavisi Arama Tahmini

Bu proje, teknoloji sektöründe çalışan bireylerin ruh sağlığı için profesyonel tedavi arayıp aramayacağını makine öğrenmesi yöntemleriyle tahmin etmeyi amaçlamaktadır. Projede OSMI Mental Health in Tech Survey 2014 ve 2016 veri setleri kullanılmıştır.

Proje; veri ön işleme, keşifsel veri analizi, model eğitimi, model karşılaştırması, yanlılık denetimi, özellik önemi analizi ve Streamlit tabanlı web uygulaması aşamalarından oluşmaktadır.

## Projenin Amacı

Bu projenin temel amacı, denetimli makine öğrenmesi yöntemleri kullanarak `treatment` hedef değişkenini tahmin etmektir.

Hedef değişken şu şekilde kodlanmıştır:

- `0` = Çalışan tedavi aramadı
- `1` = Çalışan tedavi aradı

Proje aynı zamanda tedavi arama davranışını etkileyen değişkenleri incelemeyi amaçlamaktadır. Bu değişkenlerden bazıları şunlardır:

- ruh sağlığının iş hayatına müdahale etme sıklığı
- ailede ruh sağlığı hastalığı geçmişi
- işverenin ruh sağlığı desteği sunması
- bakım seçeneklerinden haberdar olma
- anonimlik durumu
- cinsiyet dağılımı
- iş yeri destek mekanizmaları

## Veri Seti

Projede aşağıdaki veri setleri kullanılmıştır:

- OSMI Mental Health in Tech Survey 2014
- OSMI Mental Health in Tech Survey 2016

Veri setleri `Datasets/` klasörü içine yerleştirilmelidir.

Örnek proje klasör yapısı:

```text
Project/
│
├── Datasets/
│   ├── survey_2014.csv
│   └── survey_2016.csv
│
├── Models/
│   ├── best_model.joblib
│   ├── feature_columns.joblib
│   └── ...
│
├── Dataset Visuals/
│   └── oluşturulan grafikler
│
├── Saved Responses/
│   └── responses.csv
│
├── main.py
├── app.py
├── fairness_audit.py
├── mental_health_datasets_visualizations.py
├── requirements.txt
└── README.md
```

Veri seti dosya adları indirdiğiniz kaynağa göre farklı olabilir. Gerekirse Python dosyalarının içindeki veri yolu bilgilerini kendi dosya adlarınıza göre güncelleyin.

## Projenin Temel Özellikleri

- 2014 ve 2016 OSMI anket verilerini birleştirir.
- `Gender`, `Age` ve `treatment` gibi sütunlardaki hatalı veya tutarsız değerleri temizler.
- Eksik değerleri düzenler.
- Kategorik değişkenleri One-Hot Encoding yöntemiyle sayısal forma dönüştürür.
- Üç farklı makine öğrenmesi modeli eğitir:
  - Logistic Regression
  - Random Forest
  - XGBoost
- GridSearchCV ile hiperparametre optimizasyonu yapar.
- Modelleri aşağıdaki metriklerle değerlendirir:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Özellik önemi analizi yapar.
- Cinsiyet gruplarına göre fairness audit uygular.
- Kullanıcı etkileşimi için Streamlit web uygulaması sunar.

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit
- PyCharm

## Kurulum

Öncelikle projeyi bilgisayarınıza klonlayın:

```bash
git clone https://github.com/kullanici-adiniz/depo-adiniz.git
cd depo-adiniz
```

Sanal ortam oluşturun:

```bash
python -m venv venv
```

Sanal ortamı aktif edin.

Windows için:

```bash
venv\Scripts\activate
```

macOS/Linux için:

```bash
source venv/bin/activate
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

Eğer henüz `requirements.txt` dosyanız yoksa, aşağıdaki içerikle oluşturabilirsiniz:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
joblib
streamlit
```

## Proje Nasıl Çalıştırılır?

### 1. Veri Setini Hazırlama

OSMI 2014 ve 2016 veri setlerini indirin ve `Datasets/` klasörü içine yerleştirin.

`main.py` dosyasındaki veri yollarının kendi dosya adlarınızla uyumlu olduğundan emin olun.

### 2. Modelleri Eğitme

Ana eğitim dosyasını çalıştırın:

```bash
python main.py
```

Bu dosya veri ön işleme, model eğitimi, hiperparametre denemeleri, model değerlendirme ve model kaydetme işlemlerini gerçekleştirir.

Dosya çalıştırıldıktan sonra eğitilmiş model dosyaları `Models/` klasörü içine kaydedilir.

### 3. Görselleştirmeleri Oluşturma

Keşifsel veri analizi grafiklerini oluşturmak için aşağıdaki dosyayı çalıştırın:

```bash
python mental_health_datasets_visualizations.py
```

Bu dosya veri seti ile ilgili grafikleri oluşturur ve görsel çıktı klasörüne kaydeder.

### 4. Fairness Audit Çalıştırma

Cinsiyet gruplarına göre model performansını incelemek için aşağıdaki dosyayı çalıştırın:

```bash
python fairness_audit.py
```

Bu dosya modelin farklı demografik gruplar üzerinde nasıl performans gösterdiğini analiz eder.

### 5. Streamlit Uygulamasını Başlatma

Web uygulamasını çalıştırmak için:

```bash
streamlit run app.py
```

Uygulama genellikle aşağıdaki adreste tarayıcıda açılır:

```text
http://localhost:8501
```

Kullanıcılar uygulama üzerinden anket sorularını yanıtlayabilir ve eğitilmiş modelden tahmin sonucu alabilir.

## Model Çıktısı

Streamlit uygulaması tahmin sonucunu olasılık temelli olarak verir. Sonuçlar şu şekilde gösterilebilir:

- Düşük olasılık
- Orta olasılık
- Yüksek olasılık

Bu sonuç tıbbi bir tanı değildir. Yalnızca anket verilerindeki örüntülere dayalı bir makine öğrenmesi tahminidir.

## Önemli Not

Bu proje yalnızca eğitim ve akademik amaçlarla hazırlanmıştır. Tıbbi tavsiye, teşhis veya tedavi sağlamaz. Ruh sağlığı konusunda destek ihtiyacı olan kişilerin uzman bir ruh sağlığı profesyoneline başvurması gerekir.

## Proje Sonuçları

Final değerlendirmesinde Random Forest modeli F1-skoruna göre en başarılı model olmuştur. Öne çıkan değişkenler şunlardır:

- `work_interfere`
- `family_history`
- `care_options`
- `benefits`
- `Age`

Sonuçlar, iş yeri koşullarının ve kişisel/ailevi ruh sağlığı geçmişinin tedavi arama davranışını tahmin etmede önemli faktörler olduğunu göstermektedir.

## Gelecek Geliştirmeler

Bu proje ileride aşağıdaki yollarla geliştirilebilir:

- Daha güncel OSMI anket verileri projeye eklenebilir.
- SHAP yöntemiyle model açıklanabilirliği artırılabilir.
- Streamlit uygulaması çevrim içi olarak yayınlanabilir.
- SVM veya Neural Network gibi farklı modeller test edilebilir.
- Daha dengeli ve güncel veri toplanarak model yeniden eğitilebilir.

## Hazırlayan

Hazırlayan: Dafina Peci  
Ders: BM-375 Veri Madenciliğinin Prensipleri  
Ders Eğitmeni: Prof. Dr. Oktay YILDIZ
