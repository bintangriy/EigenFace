import streamlit as st
import numpy as np
from PIL import Image
import cv2
import time
import base64
from io import BytesIO

from utils import load_images_from_folder, preprocess_face
from eigenface import EigenFaceRecognizer

THRESHOLD      = 4000
NUM_COMPONENTS = 30
DATASET_PATH   = "./dataset"

st.set_page_config(page_title="Face Recognition", page_icon="◎", layout="wide", initial_sidebar_state="expanded")

st.markdown(, unsafe_allow_html=True)


@st.cache_resource
def load_dataset(path, n):
    imgs, lbls, lbl_names, color_imgs = load_images_from_folder(path)
    rec = EigenFaceRecognizer(imgs, num_components=n)
    return imgs, lbls, lbl_names, color_imgs, rec

images, labels, label_names, color_images, recognizer = load_dataset(DATASET_PATH, NUM_COMPONENTS)

if "img" not in st.session_state:
    st.session_state.img = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0



with st.sidebar:
    st.markdown('<p class="label">Dataset</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="mono">📁 {DATASET_PATH}</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    col_a.markdown(f'<p class="stat">{len(labels)}</p><p class="mono">Images</p>', unsafe_allow_html=True)
    col_b.markdown(f'<p class="stat">{len(label_names)}</p><p class="mono">Identities</p>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="label">Upload Image</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "img",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded:
        st.session_state.img = uploaded

    if st.session_state.img:
        col_name, col_btn = st.columns([3, 1])
        col_name.caption(f"✓ {st.session_state.img.name}")
        if col_btn.button("✕", type="secondary", key="remove_btn"):
            st.session_state.img = None
            st.session_state.uploader_key += 1
            st.rerun()



result  = None
exec_ms = None
image   = None

if st.session_state.img is not None:
    image   = Image.open(st.session_state.img)
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    t0      = time.time()
    face    = preprocess_face(img_bgr)

    if face is not None:
        idx, dist = recognizer.recognize(face)
        exec_ms   = (time.time() - t0) * 1000
        name      = label_names[labels[idx]] if dist < THRESHOLD else None
        result    = {"name": name, "distance": dist, "matched": color_images[idx]}



def to_base64(img) -> str:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def show_image(img):
    if img is not None:
        b64 = to_base64(img)
        st.markdown(
            f'<div class="img-wrap"><img src="data:image/png;base64,{b64}"/></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown('<div class="img-wrap"><span class="img-ph">🖼</span></div>', unsafe_allow_html=True)

def meta(label, value, style="mono"):
    st.markdown(f'<p class="label" style="margin-top:10px">{label}</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="{style}">{value}</p>', unsafe_allow_html=True)



st.markdown("# Face Recognition")
st.divider()

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown('<p class="label">Test Image</p>', unsafe_allow_html=True)
    show_image(image)
    meta("Execution Time", f"{exec_ms:.2f} ms" if exec_ms else "—")
    meta("Distance Score", f"{result['distance']:.2f}" if result else "—")

with col_right:
    st.markdown('<p class="label">Closest Match</p>', unsafe_allow_html=True)
    show_image(result["matched"] if result else None)

    if result is None:
        meta("Result", "—")
    elif result["name"]:
        meta("Result", f"✓ {result['name']}", style="ok")
    else:
        meta("Result", "Unknown", style="fail")