
import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps 


st.set_page_config(page_title="Deteksi Alzheimer.AI")

model = load_model('modelalzheimer.h5')


def prediksi_gambar(file_path):
    class_names = open("labelsalzheimer.txt", "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(file_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    hasil = {
        'label_kelas': class_name,
        'skor_kepercayaan': confidence_score    
    }

    return hasil




# Aplikasi Streamlit

# Navigasi
halaman_terpilih = st.selectbox("Pilih Halaman", ["Beranda", "Halaman Deteksi", "Visualisasi Model"], format_func=lambda x: x)

if halaman_terpilih == "Beranda":
    # Tampilkan halaman Beranda
    st.header("Selamat datang di Aplikasi Deteksi Alzheimer!", divider='rainbow')
    st.write(
        "Aplikasi ini memungkinkan Anda untuk mengunggah Hasil Foto MRI Anda "
        "dan mendapatkan hasil deteksi apakah Foto MRI Anda terdeteksi Mild Demented, Moderate Demented, Non Demented, atau Very Mild Demented."
    )
    st.write("Silakan pilih 'Halaman Deteksi' untuk memulai Deteksi hasil foto MRI.")

elif halaman_terpilih == "Halaman Deteksi":
    # Tampilkan halaman Deteksi
    st.title("Unggah Gambar")
    st.markdown("---")

    # Unggah gambar melalui Streamlit
    berkas_gambar = st.file_uploader("Silahkan Pilih gambar ", type=["jpg", "jpeg", "png"])

    if berkas_gambar:
        # Tampilkan gambar yang dipilih
        st.image(berkas_gambar, caption="Gambar yang Diunggah", use_column_width=True)

        # Lakukan prediksi saat tombol ditekan
        if st.button("Deteksi"):
            # Simpan berkas gambar yang diunggah ke lokasi sementara
            with open("temp_image.jpg", "wb") as f:
                f.write(berkas_gambar.getbuffer())

            # Lakukan prediksi pada berkas yang disimpan
            hasil_prediksi = prediksi_gambar("temp_image.jpg")

            # Tampilkan hasil prediksi
            st.write(f"Deteksi: {hasil_prediksi['label_kelas']}")
            st.write(f"Skor Kepercayaan: {hasil_prediksi['skor_kepercayaan']:.2%}")

else: 
    



# Aplikasi Streamlit
    st.title("Kinerja Model AI")
    st.markdown("---")

    def display_image_table(image_path1, title1, caption1, image_path2, title2, caption2):
        try:
            print(f"Trying to open image: {image_path1}")
            image1 = Image.open(image_path1)
            image2 = Image.open(image_path2) if image_path2 else None
        except FileNotFoundError:
            st.error(f"File not found: {image_path1}")
            return

        col1, col2 = st.columns(2)

        with col1:
            col1.markdown(f'<h2 style="text-align:center;">{title1}</h2>', unsafe_allow_html=True)
            col1.markdown(
                f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
            )
            col1.image(image1, use_column_width=True)
            col1.markdown(f'<p stye="text-align:left;">{caption1}</p>', unsafe_allow_html=True)
        
        with col2:
            if image2:
                col2.markdown(f'<h2 style="text-align:center;">{title2}</h2>', unsafe_allow_html=True)
            
                col2.markdown(
                    f'<div style="display: flex; justify-content: center;"></div>',
                unsafe_allow_html=True
                )
                col2.image(image2, use_column_width=True)
                col2.markdown(f'<p stye="text-align:left;">{caption2}</p>', unsafe_allow_html=True)

    images_info = [
        {'akurasi.png', 'title':'Accuracy Class', 'caption':'Data “Accuracy per class” menunjukkan bahwa model memiliki akurasi (0.87) untuk mengidentifikasi Alzheimer, Dan (0.93) untuk mengidentifikasi Non Demented atau Normal. Ini menunjukkan bahwa model efektif dalam membedakan foto hasil MRI amtara Alzheimer dan Non Demented atau Normal.'},
        {'epoch.png', 'title':'Accuracy Epoch', 'caption':'Grafik akurasi per epoch menunjukkan bahwa akurasi pada data pelatihan (garis biru) meningkat dengan cepat dan mencapai nilai tinggi mendekati 1 setelah sekitar 50 epoch, tetapi mengalami beberapa fluktuasi awal. Akurasi pada data pengujian (garis oranye) juga meningkat di awal, namun setelah sekitar 50 epoch stabil pada nilai sekitar 0.85 dan tidak mengalami peningkatan signifikan lebih lanjut. Ini mengindikasikan bahwa model mampu belajar dari data pelatihan, namun setelah 50 epoch, peningkatan akurasi pada data pelatihan tidak diikuti oleh peningkatan pada data pengujian, mengisyaratkan kemungkinan overfitting, di mana model tidak mampu generalisasi dengan baik ke data baru.'},
        {'confusionmatrix.png', 'title':'Confusion Matrix', 'caption':'Pada matriks kebingungan ini, terdapat dua kelas: "Alzheimer" dan "Normal". Elemen diagonal (419 dan 448) mewakili jumlah prediksi yang benar untuk setiap kelas. Elemen di luar diagonal (61 dan 32) mewakili jumlah prediksi yang salah atau kesalahan klasifikasi. Dari matriks tersebut, kita dapat melihat bahwa model berhasil memprediksi 419 instance sebagai "Alzheimer" dan 448 instance sebagai "Normal". Namun, model juga salah mengklasifikasi 61 instance sebagai "Alzheimer" padahal sebenarnya "Normal", dan 32 instance sebagai "Normal" padahal sebenarnya "Alzheimer". Informasi ini membantu menilai akurasi, presisi, recall, dan metrik performa lainnya dari model.'},
        {'lossepoch.png', 'title':'Loss Epoch', 'caption':'Grafik tersebut menunjukkan bahwa model mengalami overfitting setelah sekitar 50 epoch, di mana loss pada data pelatihan terus menurun sementara test loss meningkat tajam dan tetap tinggi. Ini mengindikasikan bahwa model terlalu menyesuaikan data pelatihan sehingga tidak mampu generalisasi dengan baik ke data baru. Untuk mengatasi masalah ini, dapat digunakan teknik seperti early stopping, regularisasi, atau penambahan data pelatihan untuk meningkatkan kemampuan generalisasi model.'},
    ]
    for i in range(0, len(images_info), 2):
        if i + 1 < len(images_info):
            display_image_table(
                images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'],
                images_info[i + 1]['path'], images_info[i + 1]['title'], images_info[i + 1]['caption']
            )
        else:
            display_image_table(images_info[i]['path'], images_info[i]['title'], images_info[i]['caption'], '', '', '')
