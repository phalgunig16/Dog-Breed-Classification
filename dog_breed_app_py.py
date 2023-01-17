import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import altair as alt 
from utils import load_and_prep, get_classes

@st.cache(suppress_st_warning=True)

def predicting(image, model):
  image = load_and_prep(image) # uplaod and load image
  image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
  preds = model.predict(image)
  pred_label = class_names[tf.argmax(preds[0])]
  pred_conf = tf.reduce_max(preds[0])
  top_5_i = sorted((preds.argsort())[0][-5:][::-1])
  values = preds[0][top_5_i] * 100
  labels = []

  for  x in range(5):
    labels.append(class_names[top_5_i[x]])

  df = pd.DataFrame({"Top 5 Predictions": labels,
                     "F1 Scores": values,
                     'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953'
                     ]})

  return pred_label, pred_conf, df

  st.set_page_config(page_title="Dog Breed Prediction")

#### SideBar ####

st.sidebar.title("Dog Breed Deep Learning")
st.sidebar.write("""
Dog Vision is an end-to-end **CNN Image Classification Model** which identifies the dog breed in your image. 
  
It can identify over 120 different dog breeds
  
It is based upom a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.
  
**Accuracy :** **`85%`**
**Model :** **`TransferLearning`**
**Dataset :** **`Dog breed - Kaggle`**
""")

#### Main Body ####

st.title("Dog Breed Classification")
st.header("Identify the breed of dog!")

st.write("To know more about this app, visit [**GitHub**](https://github.com/phalgunig16/Dog-Breed-Classification)")
file = st.file_uploader(label="Upload an image of a dog.",type=["jpg","jpeg","png"])

model = tf.keras.models.load_model("./models/EfficientNetB1.hdf5")

st.sidebar.markdown("created by **Phalguni G**")

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
  image = file.read()
  st.image(image, use_column_width=True)
  pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_label} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))