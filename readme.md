# Face Recognition Application
Aplikasi pengenalan wajah berbasis GUI yang mengimplementasikan metode Eigenface dan algoritma jarak Euclidean. 
Sistem ini mendeteksi identitas dengan cara mengekstraksi dan mempelajari karakteristik dari sekumpulan citra wajah di dalam dataset, kemudian mencocokkan pola wajah masukan dengan data yang tersedia untuk mengidentifikasi citra yang paling serupa.

## Persyaratan Sistem
Pastikan perangkat Anda telah terinstal Python (versi 3.8 atau yang lebih baru). 
Beberapa library yang wajib dipasang:
- streamlit
- opencv-python
- numpy
- pandas

## Instalasi
- Jalankan perintah berikut untuk menginstal dependensi:  
  `pip install streamlit opencv-python numpy pandas`
- Perintah untuk mengakses aplikasi:  
  `streamlit run src/facerecog.py`

## Cara Menjalankan
1. Pada sidebar, tentukan direktori atau folder dataset wajah Anda (misal: `./dataset`).
2. Klik tombol Browse files untuk mengunggah foto wajah baru yang ingin diidentifikasi.
3. Sistem secara otomatis akan memproses dataset (mengalkulasi matriks Eigenface) dan menganalisis kecocokan wajah terdekat.
4. Hasil akhir berupa identitas wajah yang cocok beserta nilai jarak Euclidean akan ditampilkan pada layar utama.

## Video Demo Project EigenFace
YouTube     : https://youtu.be/P2pYXHcZOPM

# Link Github Project
GitHub      : https://github.com/bintangriy/EigenFace 
