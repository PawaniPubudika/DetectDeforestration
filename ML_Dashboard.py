import streamlit as st
from patchify import patchify
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import load_model
import base64


def callback():
  st.session_state.Button = True


@st.cache_data(persist=True)
def getImageAsBase64(file):
  with open(file, "rb") as f:
    data = f.read()
  return base64.b64encode(data).decode()


def display():
  for i in range(len(st.session_state.image)):
      col1,col2 = st.columns(2)
      col1.write("""<h1 style='text-align: center; font-size: 10mm; color: black; background-color: rgba(255, 255, 255, 0.2); padding: 8px; border-radius: 15px 50px;'>Original Image</h1>""", unsafe_allow_html=True)
      col1.image(st.session_state.image[i])
      col2.write("""<h1 style='text-align: center; font-size: 10mm; color: black; background-color: rgba(255, 255, 255, 0.2); padding: 8px; border-radius: 15px 50px;'>Mask Image</h1>""", unsafe_allow_html=True)
      col2.image(st.session_state.mask_image[i])

      if st.session_state.predict[i] == 'Forest':
        st.write(f"""<p style = 'background-color: rgba(0, 255, 0, 0.9); color: black; font-size: 6mm; padding: 15px; border-radius: 15px 50px;'>
                      <b>Predicted Label : Forest Area</b></p>""",unsafe_allow_html=True)
      else:
        st.write(f"""<p style = 'background-color: rgba(255, 0, 0, 0.9); color: black; font-size: 6mm; padding: 15px; border-radius: 15px 50px;'>
                      <b>Predicted Label : Deforested Area</b></p>""",unsafe_allow_html=True)


file_path = '/content/drive/MyDrive/4th_Year/DSC4173/Project/'
st.set_page_config(page_title="Detect Deforestration",page_icon="🌴",layout="wide",initial_sidebar_state="expanded")
img = getImageAsBase64(file_path + "forest.jpg")
st.markdown(f"""
  <style>
    [data-testid="stAppViewContainer"]{{
      background-image: url("data: image/png;base64,{img}");
      background-size: cover;
    }}
    [data-testid="stHeader"]{{
      background-color: rgba(0,0,0,0);
    }}
    [data-testid="stToolbar"]{{
      right: 0rem;
    }}
    .uploadedFileName css-1uixxvy e1b2p2ww6{{
      font-size: 20px;
    }}
  </style>""",unsafe_allow_html=True)
st.write("""<h1 style='text-align: center; font-size: 13mm; color: rgba(255,0,0,0.9);'>Detect Deforestration</h1></br>
<p style='text-align: center; font-size: 6mm; color: rgba(255,0,0,0.9);'>Faculty of Science</br>University of Peradeniya</p></b>""", unsafe_allow_html=True)


if 'unet' not in st.session_state:
  st.session_state.unet = load_model(file_path + 'Unet_epoch_50')
if 'cnn' not in st.session_state:
  st.session_state.cnn = load_model(file_path + 'ResNet_epoch_100')
if 'image' not in st.session_state:
  st.session_state.image = []
if 'mask_image' not in st.session_state:
  st.session_state.mask_image =[]
if 'predict' not in st.session_state:
  st.session_state.predict = []
if 'num_images_prev' not in st.session_state:
  st.session_state.num_images_prev = 0


images = st.file_uploader("Drag and drop image or images : ", accept_multiple_files=True, type = ['png','jpg'])
num_images_now = len(images)

if 'Button' not in st.session_state:
      st.session_state.Button = False

if images != []:
  if num_images_now != st.session_state.num_images_prev:
    submit = st.button("Submit")

    if submit:
      st.session_state.mask_image =[]
      st.session_state.image = []
      st.session_state.predict = []
      time = 0

      for image in images:
        image = Image.open(image)
        image = np.array(image)
        patches = patchify(image, (256,256,3), step = 256)
        time = time + patches.shape[0]*patches.shape[1]
      
      st.write(f"""<p style = 'background-color: rgba(0, 0, 255, 0.5); color: white; font-size: 6mm; padding: 20px; border-radius: 15px 50px;'>
                      <b>Please wait {time//60}m {time % 60}s... Model is on process...</b></p>""",unsafe_allow_html=True)

      for image in images:
        image = Image.open(image)
        image = np.array(image)

        image_height = (image.shape[0]//256)*256
        image_width = (image.shape[1]//256)*256

        image = Image.fromarray(image)
        image = image.crop((0, 0, image_width, image_height))

        image = np.array(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        st.session_state.image.append(image)
        patches = patchify(image, (256,256,3), step = 256)

        rows = patches.shape[0]
        cols = patches.shape[1]

        all_masks = []
        for row in range(rows):
          for col in range(cols):
            img_input = np.expand_dims((patches[row][col][0])/255, 0)
            predicted_mask = st.session_state.unet.predict(img_input)
            all_masks.append(predicted_mask)


        labels = []

        for img_input in all_masks:
          predict_label = st.session_state.cnn.predict(img_input)[0][0]
          labels.append(predict_label)
        labels = np.array(labels)
        label = sum(labels)/len(labels)

        if label >= 0.5:
            st.session_state.predict.append('Deforest')
        else:
          st.session_state.predict.append('Forest')


        full_mask_image = np.zeros((256*rows,256*cols,3))

        image_count = 0
        for row in range(rows):
          for col in range(cols):
            for channel in range(3):
              full_mask_image[row*256:(row+1)*256, col*256:(col+1)*256, channel] = all_masks[image_count][0][:,:,channel]*255
            image_count += 1

        full_mask_image = full_mask_image.astype('int')
        st.session_state.mask_image.append(full_mask_image)
      # st.session_state.Button = False
      st.session_state.num_images_prev = num_images_now
      st.experimental_rerun()
  else:
    display()
