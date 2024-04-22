import streamlit as st
from PIL import Image
import numpy as np
import model_loader as ml
import pandas as pd
import time
import cv2
import osascript
# leaf blight grae
# bacterial spot bell pepper

def camera_feed():
    img = st.camera_input("Take a Picture 📷")
    # cmd = 'do shell script "screencapture ~/Desktop/screenshot'+str(int(time.time()))+'.png"'
    
    
    if img:
        filename = 'savedImage.jpg'
        cv2.imwrite(filename, img) 
        return filename

st.set_page_config(
    page_title="Plant Disease Predictor",
    page_icon="favicon.png"
)
st.markdown("# Plant Disease Predictor", unsafe_allow_html=True)
st.markdown("---")

plant_type = st.sidebar.selectbox(
    label="Choose plant type",
    options=ml.get_plants()
)

# plant_image = st.sidebar.file_uploader(
#     label="Upload your image",
#     type=["png", "jpg", "jpeg"]
# )
plant_image = None 
if st.sidebar.button("Capture Image"):
    captured_image = st.camera_input("Take a Picture 📷")
    print(captured_image)
    if captured_image:
        plant_image = captured_image
    

else:
    plant_image = st.sidebar.file_uploader(
        label="Upload your image",
        type=["png", "jpg", "jpeg"]
    )



def process_and_display_image(image): 
    img = Image.open(image)
    img.thumbnail((800, 500))  # Keep the aspect ratio intact
    return img


placeholder = st.empty()

if plant_image and "image" in plant_image.type:
    st.image(
        process_and_display_image(image=plant_image),
        channels="RGB"
    )


def read_image(data) -> np.ndarray:
    input_image = Image.open(data)
    resized_image = input_image.resize((256, 256))
    final_image = resized_image.convert("RGB")
    return np.array(final_image)


def predict_disease(plant, image) -> pd.DataFrame:
    model = ml.load_model(plant=plant_type)
    plant_diseases = ml.get_disease(plant=plant)
    img = read_image(image)
    img_batch = np.expand_dims(img, 0)

    prediction = np.array([x * 100 for x in model.predict(img_batch)[0]])
    max_disease = max(prediction)
    disease_index = np.where(prediction == max_disease)
    df = pd.DataFrame(
        data=prediction.reshape(1, -1), columns=plant_diseases, index=["Confidence %"]
    )
    return df
def stream_data(plant_disease_solution):
    for word in plant_disease_solution.split(" "):
        yield word + " "
        time.sleep(0.02)


def sol(query):
        plant_disease_solution = ml.get_solution(query)
    # st.write(plant_disease_solution[0])
    # st.write(plant_disease_solution[1])

        st.write_stream(stream_data(plant_disease_solution[0]))
        ml.read_solution(plant_disease_solution[0], "en")
        st.write_stream(stream_data(plant_disease_solution[1]))
        ml.read_solution(plant_disease_solution[1], "hi")




    # plant_disease_solution_hi =  ml.read_top_snippet(plant_disease_solution)
    # return [plant_disease_solution, plant_disease_solution_hi ]

placeholder = st.empty()

if st.button(label="Run Prediction") and plant_image:
    placeholder.write("Please wait...")
    data = predict_disease(plant=plant_type, image=plant_image)
    placeholder.empty()
    #st.dataframe(data=data)
    newList = data.values.tolist()
    high = max(newList[0])
    disease_index = newList[0].index(high)
    plant_disease_ls = ml.get_disease(plant = plant_type)
    plant_disease = plant_disease_ls[disease_index]

    st.title("Disease "+ plant_disease)


    sol(plant_disease)









