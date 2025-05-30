import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def squash(vectors, axis=-1):
    """Fungsi aktivasi squash untuk Capsule Network."""
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm)
    return scale * vectors / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())

@register_keras_serializable()
def capsule_length(z):
    return tf.norm(z, axis=-1)

@register_keras_serializable()
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, num_routing=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules
        self.num_routing = num_routing

    def build(self, input_shape):
        self.input_num_capsules = input_shape[1]
        self.input_dim_capsules = input_shape[2]
        self.W = self.add_weight(
            shape=[self.input_num_capsules, self.num_capsules, self.input_dim_capsules, self.dim_capsules],
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), -1)
        W_tiled = tf.tile(tf.expand_dims(self.W, 0), [tf.shape(inputs)[0], 1, 1, 1, 1])
        u_hat = tf.matmul(W_tiled, inputs_expand)
        u_hat = tf.squeeze(u_hat, axis=-1)

        b = tf.zeros_like(u_hat[:, :, :, 0])

        for i in range(self.num_routing):
            c = tf.nn.softmax(b, axis=2)
            outputs = squash(tf.reduce_sum(tf.expand_dims(c, -1) * u_hat, axis=1))

            if i < self.num_routing - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(outputs, 1), axis=-1)

        return outputs

# --- Muat Model ---
model = tf.keras.models.load_model(
    "CapsNet100-16.h5",
    compile=False,
    custom_objects={
        "squash": squash,
        "CapsuleLayer": CapsuleLayer,
        "capsule_length": capsule_length
    }
)

# Label kelas 
class_labels = ['Segar', 'Tidak Segar']  

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # shape: (1, 128, 128, 3)
    return image

# UI Streamlit
st.set_page_config(page_title="Klasifikasi Kesegaran Cumi", layout="centered")

# Sidebar - Petunjuk Penggunaan
with st.sidebar:
    st.header("📌 Petunjuk Penggunaan")
    st.markdown("""
    1. Klik tombol **Browse files** untuk memilih gambar dari perangkat Anda.
    2. Pastikan gambar hanya menunjukkan bagian tubuh dari cumi-cumi.
    3. Setelah gambar muncul, tekan tombol **Identifikasi**.
    4. Hasil klasifikasi akan muncul di bawah gambar dengan  label **Segar** atau **Tidak Segar** beserta tingkat kepercayaannya.
    """)

st.title("🦑 Identifikasi Kesegaran Cumi")
st.markdown("Unggah citra cumi-cumi dan tekan tombol **Identifikasi** untuk mengetahui kesegarannya.")
st.markdown(
    "<div style='text-align: center;'>"
    "<img src='https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExdG40dDMxdWN6emZkZGVicXZ3aTFmZXFweW1xbGEzdHpiamFyNmR0ZiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/vXAq5Lv9YdmtG/giphy.gif', width='250'>"
    "</div>",
    unsafe_allow_html=True
)



uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diunggah", width=300)

    if st.button("Identifikasi"):
        with st.spinner("Memproses..."):
            img_preprocessed = preprocess_image(image)
            prediction = model.predict(img_preprocessed)

            
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]

            label = class_labels[predicted_class]
            st.success(f"Hasil Prediksi: **{label}** ({confidence * 100:.2f}%)")
