import streamlit as st

st.title("Camera Feed")
img = st.camera_input("Take a picture")
if img:
    st.image(img)