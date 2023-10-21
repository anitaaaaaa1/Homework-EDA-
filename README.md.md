# Team Predictive Pioneers

# STAGE 1

## Langkah-langkah melakukan EDA

## 1. Descriptive Statistics

### Import Dataset

Download Dataset [here](https://www.kaggle.com/datasets/adammaus/predicting-churn-for-bank-customers/)

Source code :

```code
import pandas as pd
```

### Load the dataset
```code
df = pd.read_csv('Churn_Modelling.csv')
```

### Display the first few rows of the dataset to understand its structure
```code
df.head()
```

Berdasarkan tampilan awal dari dataset tersebut, kita memiliki informasi tentang beberapa pelanggan dari sebuah bank dan apakah mereka telah keluar (churn) dari bank tersebut. Beberapa fitur yang ada antara lain:

**RowNumber** : Nomor baris

**CustomerId** : ID pelanggan

**Surname** : Nama belakang pelanggan

**CreditScore** : Skor kredit pelanggan

**Geography** : Negara asal pelanggan

**Gender** : Jenis kelamin pelanggan

**Age** : Usia pelanggan

**Tenure** : Durasi pelanggan telah menjadi anggota bank

**Balance** : Saldo akun pelanggan

**NumOfProducts** : Jumlah produk yang dimiliki pelanggan di bank

**HasCrCard** : Apakah pelanggan memiliki kartu kredit

**IsActiveMember** : Apakah pelanggan aktif

**EstimatedSalary** : Gaji yang diperkirakan

**Exited** : Apakah pelanggan telah keluar dari bank (1 berarti ya, 0 berarti tidak)

Dari fitur-fitur di atas, RowNumber, CustomerId, dan Surename sepertinya tidak relevan untuk analisis korelasi karena mereka adalah identifikasi unik pelanggan dan tidak memberikan informasi substansial tentang perilaku pelanggan.









### 2. Univariate Analysis







### 3. Multivariate Analysis
***Pertama,*** kita akan melihat korelasi antara masing-masing fitur dengan label (Exited).

```code
import matplotlib.pyplot as plt
import seaborn as sns
```

#Drop non-relevant columns for the correlation analysis
```code
df_corr = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
```

#Compute the correlation with the 'Exited' column
```code
correlation_with_label = df_corr.corr()['Exited'].sort_values(ascending=False)
```

#Plot the correlation with the 'Exited' column
```code
plt.figure(figsize=(10, 6))
sns.barplot(x=correlation_with_label.values, y=correlation_with_label.index)
plt.title('Correlation with Exited label')
plt.xlabel('Correlation Coefficient')
plt.show()
```

**correlation_with_label**

**Hasil nya:**

```code
RESULTE
Exited             1.000000
Age                0.285323
Balance            0.118533
EstimatedSalary    0.012097
HasCrCard         -0.007138
Tenure            -0.014001
CreditScore       -0.027094
NumOfProducts     -0.047820
IsActiveMember    -0.156128
Name: Exited, dtype: float64
```

**Grafik nya:**

Berdasarkan grafik dan tabel korelasi di atas:

![correlation_with_exited_label](C:\Users\LENOVO\Downloads\correlation_with_exited_label.jpg)

**Korelasi antara masing-masing fitur dengan label (Exited):**

- Age memiliki korelasi positif yang paling kuat dengan Exited, yang berarti pelanggan yang lebih tua cenderung memiliki kemungkinan lebih tinggi untuk keluar.
- Balance juga memiliki korelasi positif dengan Exited, meskipun tidak sekuat Age.
- IsActiveMember memiliki korelasi negatif terkuat dengan Exited, menunjukkan bahwa anggota yang aktif memiliki kemungkinan lebih rendah untuk keluar.
- Fitur-fitur lain seperti EstimatedSalary, HasCrCard, Tenure, CreditScore, dan NumOfProducts memiliki korelasi yang relatif lemah dengan Exited.

**Rekomendasi fitur yang harus dipertahankan berdasarkan relevansinya dengan Exited:**

- Age, Balance, dan IsActiveMember tampaknya merupakan fitur yang paling relevan berdasarkan korelasinya dengan Exited dan harus dipertahankan.
- Meskipun fitur lain memiliki korelasi yang lebih rendah, mereka mungkin tetap berguna tergantung pada model yang kita gunakan dan pertimbangan lain seperti interpretasi bisnis.

**A. Kemudian, kita akan melihat korelasi antar fitur.**

#Compute the correlation matrix
```code
correlation_matrix = df_corr.corr()
```

#Plot the correlation heatmap
```code
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()
```

Grafik nya:

Berdasarkan heatmap korelasi antar-fitur:

Observasi:
- Tidak ada dua fitur yang memiliki korelasi yang sangat kuat satu sama lain.
- Sebagian besar fitur memiliki korelasi yang lemah atau sedang dengan fitur lainnya.
- NumOfProducts memiliki korelasi negatif dengan Balance. Meskipun korelasinya tidak terlalu kuat, ini dapat menunjukkan bahwa pelanggan dengan lebih banyak produk cenderung memiliki saldo yang lebih rendah. Ini mungkin karena beberapa produk bank tidak memerlukan saldo yang tinggi atau pelanggan dengan banyak produk memanfaatkan layanan lain yang tidak memerlukan saldo yang besar.

Rekomendasi:
- Tidak ada fitur yang memiliki korelasi yang sangat kuat sehingga tidak perlu ada tindakan khusus seperti menghapus salah satu dari dua fitur yang berkorelasi.
- Selanjutnya, kita mungkin ingin mempertimbangkan analisis lebih lanjut atau pemodelan untuk mengekstrak informasi lebih lanjut dari fitur-fitur ini.

Lalu kami juga menganalisa 2 categorical kolom terhadap label ‘Exited’ yaitu kolom ‘Geography’ dan ‘Gender’.

Source code : 

#memisahkan kolom ke dalam categorical
```code
df_cat = df[['Surname','Geography', 'Gender','Exited']]
sns.histplot(data=df_cat, x=df_cat['Geography'], hue=df_cat['Exited'])
plt.title('Geography vs Exited')
plt.ylabel('Sum of Exited')
```



**Insight :**
Berdasarkan analisa menggunakan kolom Geography, kami mendapatkan bahwa negara germany memiliki tingkat yang berpotensi churn terbanyak diantara kedua negara lainnya. Oleh karena itu, perlunya action yang harus dilakukan terhadap negara tersebut agar pelanggan dapat betah dengan product yang ditawarkan.

Source code : 
```code
sns.histplot(data=df_cat, x=df_cat['Gender'], hue=df_cat['Exited'])
plt.title('Gender vs Exited')
plt.ylabel('Sum of Exited')
```



**Insight :**
Berdasarkan analisa menggunakan kolom Gender, didapatkan bahwa pelanggan dengan jenis kelamin female lebih berpotensi untuk melakukan churn daripada laki-laki. Oleh karna itu, bank harus lebih memberikan banyak tindakan terhadap nasabah perempuan agar tetap menjadi nasabah pada bank tersebut.


### 4. Business Insight

Source code : 

Import dataframe ‘Bank Churn’ menjadi variable ‘data’

**#Membuat plot distribusi EstimatedSalary**
```code
import plotly.express as px
fig = px.histogram(data, x="EstimatedSalary", y='Exited', nbins=30, title="Distribusi EstimatedSalary terhadap tingkat churn")
fig.show()
```

![4_1](/Downloads/4_1.jpg)

**Hasil analisis :** EstimatedSalary atau jumlah pendapatan **tidak memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi yang hampir merata pada setiap jangkauan pendapatan atau EstimatedSalary

**#Membuat plot distribusi Usia terhadap tingkat churn**
```code
fig1 = px.histogram(data, x="Age", y='Exited', nbins=30, title="Distribusi Usia terhadap Churn")
fig1.show()
``` 

![4_2](C:\Users\LENOVO\Downloads\4_2.jpg)

**Hasil analisis :** Usia **memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi normal pada age atau usia, dengan nasabah yang melakukan exit atau churn umumnya memiliki jangkauan usia 40-49 tahun


**#Membuat plot distribusi Balance terhadap tingkat churn**
```code
fig2 = px.histogram(data, x="Balance", y='Exited', nbins=30, title="Distribusi Balance terhadap Churn")
fig2.show()
```

![4_3](C:\Users\LENOVO\Downloads\4_3.jpg)

**Hasil analisis :** Balance **memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi normal pada Balance, dengan nasabah yang melakukan exit atau churn umumnya memiliki jangkauan balance 0

**#Membuat plot distribusi ActiveMember terhadap tingkat churn**
```code
fig3 = px.histogram(data, x="IsActiveMember", y='Exited', title="Pengaruh Active member terhadap Churn")
fig3.show()
```

![4_4](C:\Users\LENOVO\Downloads\4_4.jpg)

**Hasil analisis :** Jumlah active member **memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari nasabah yang melakukan exit atau churn umumnya bukan member aktif

**#Membuat plot distribusi Tenure terhadap tingkat churn**
```code
fig4 = px.histogram(data, x="Tenure", y='Exited', nbins=30, title="Distribusi Tenure terhadap Churn")
fig4.show()
```

![4_5](C:\Users\LENOVO\Downloads\4_5.jpg)

**Hasil analisis :** Tenure atau masa nasabah menggunakan layanan bank **tidak memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi yang hampir merata pada setiap jangkauan tenure

**#Membuat plot distribusi Credit Score terhadap tingkat churn**
```code
fig5 = px.histogram(data, x="CreditScore ", y='Exited', title="Distribusi CreditScore terhadap Churn")
fig5.show()
```

![4_6](C:\Users\LENOVO\Downloads\4_6.jpg)

**Hasil analisis :** Credit score **memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi normal pada creditscore, dengan nasabah yang melakukan exit atau churn memiliki jangkauan credit score 580-699

**#Membuat plot distribusi kepemilikan CreditCard terhadap tingkat churn**
```code
fig6 = px.histogram(data, x="HasCrCard", y='Exited', title="Pengaruh HasCrCard terhadap Churn")
fig6.show()
```

![4_7](C:\Users\LENOVO\Downloads\4_7.jpg)

**Hasil analisis :** HasCrCard atau kepemilikan kartu kredit **memiliki pengaruh signifikan** terhadap churn, hal ini dapat dilihat dari kecenderungan nasabah yang memiliki kartu kredit untuk melakukan churn

**#Membuat plot distribusi Gender terhadap tingkat churn**
```code
fig7 = px.histogram(data, x="Gender", y='Exited', title="Pengaruh Gender terhadap Churn")
fig7.show()
```

![4_8](C:\Users\LENOVO\Downloads\4_8.jpg)

**Hasil analisis :** Gender atau jenis kelamin **memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi gender wanita atau Female yang memiliki jumlah exit lebih tinggi dibanding pria. Hal ini dapat dilihat bahwa wanita memiliki kecenderungan untuk churn lebih tinggi dibandingkan dengan pria.


**#Membuat plot distribusi Geography terhadap tingkat churn**
```code
fig8 = px.histogram(data, x="Geography", y='Exited', title="Distribusi Geography terhadap Churn")
fig8.show()
```

![4_9](C:\Users\LENOVO\Downloads\4_9.jpg)

**Hasil analisis :** Geography atau negara asal nasabah **tidak memiliki pengaruh** terhadap churn, hal ini dapat dilihat dari distribusi yang hampir tidak berbeda signifikan antar lokasi. Distribusi France & Germany berbeda signifikan dengan distribusi Spain

Berdasarkan dari berbagai figur plot, dari berbagai atribut, nasabah yang cenderung melakukan churn banyak berada di jangkauan sebagai berikut,

**Card Balance :** 0

**Age :** 40-49

**Member :** Not Active Member

**CreditCard :** has credit card

**Credit score :** 580-699

**Gender :** Female

**Geography :** France or Germany


**Rekomendasi bisnis :**  menawarkan program benefit loyalitas bagi nasabah dengan kartu kredit dan memiliki credit score 600-800, sehingga nasabah terpacu untuk mempertahankan loyalitas nasabah yang memiliki credit score pada range tersebut. Solusi lainnya yaitu menyebarkan survey kepuasan pelanggan, sehingga diperoleh permasalahan yang dapat menyebabkan ketidakpuasan pelanggan (yang menyebabkan card balance 0 dan tidak aktif)


