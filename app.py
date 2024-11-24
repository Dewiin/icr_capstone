import streamlit as st
import google.generativeai as genai
import numpy as np
from ultralytics import YOLO
from onnx_model import ImageToWordModel
from mltu.configs import BaseModelConfigs
from PIL import Image
import time
import cv2

st.set_page_config(
  page_title='Welcome! ✍🏼',
  page_icon='👋'
)

# Initialize all models
line_detection_model = YOLO('models/detect_lines.pt')

word_detection_model = YOLO('models/detect_words.pt')

configs = BaseModelConfigs.load('models/configs.yaml')
word_prediction_model = ImageToWordModel(model_path='models/classify_word.onnx', char_list=configs.vocab)

GEMINI_API_KEY = st.secrets['GEMINI_API_KEY']
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')


# Gemini model
def gemini_response(words):
  prompt = f'''You will be given an input with 1 or more word with potential typos, misspellings, and bad autocorrection: {words}.
  Return ONLY the input after fixing any potential errors. If there are no errors, just return the input without fixing anything.'''

  response = gemini_model.generate_content(prompt)
  return response.text


# Image padding
def prepare_image(uploaded_file):
  image = Image.open(uploaded_file)
  image = np.array(image)

  if len(image.shape) == 2:
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

  h,w = image.shape[:2]
  new_size = max(h,w) + 200

  padded_image = np.full((new_size, new_size, 3), (255,255,255), dtype=np.uint8)
  x_offset = (new_size-w)//2
  y_offset = (new_size-h)//2

  padded_image[y_offset:y_offset+h, x_offset:x_offset+w] = image
  return padded_image


# Line detection
def detect_lines(padded_image):
  results = line_detection_model.predict(padded_image, conf=0.1, iou=0.3)
  coords = []
  output_lines = []

  for result in results:
    for box in result.boxes:
      x1, y1, x2, y2 = map(int, box.xyxy[0])
      coords.append([x1,y1,x2,y2])
  
  coords = sorted(coords, key=lambda x:x[1])

  for coord in coords:
    x1,y1,x2,y2 = coord
    output_lines.append(padded_image[y1:y2, x1:x2])

  return output_lines


# Word detection 
def detect_words(lines):  
  words = []

  for line in lines:
    coords = []
    results = word_detection_model.predict(line, conf=0.1, iou=0.1)
    for result in results:
      for box in result.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        if (x2-x1 < 7) | (y2-y1 < 7):
          continue
        coords.append([x1,y1,x2,y2])
    
    coords = sorted(coords, key=lambda x:x[0])
    for coord in coords:
      x1,y1,x2,y2 = coord
      words.append(line[y1:y2, x1:x2])
  
  return words


# Word prediction model
def predict_words(words):
  output = ''
  for word in words:
    prediction = word_prediction_model.predict(word)

    output += (prediction + ' ')

  result = gemini_response(output)
  return result


# stream text
def generate_stream(input):
  for word in input.split():
    yield word + ' '
    time.sleep(0.05)


# main
def main():
  st.markdown('<h1 style="text-align:center;">Handwriting Recognition ✍🏼</h1>', unsafe_allow_html=True)
  st.markdown('<h6 style="text-align:center; margin-bottom: 2rem;">Upload an image of handwritten text.</h6>', unsafe_allow_html=True)

  uploaded_file = st.file_uploader('', type=['png', 'webp', 'jpg', 'jpeg'])

  if uploaded_file:
    with st.spinner('Analyzing...'):
      col1, col2 = st.columns(2)
      with col1:
        st.image(uploaded_file)

      with col2:
        image = prepare_image(uploaded_file)
        lines = detect_lines(image)
        if not lines:
          image = Image.open(uploaded_file)
          image = np.array(image)
          if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
          elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
          result = word_prediction_model(image)
          st.markdown(result)
        else:
          words = detect_words(lines)
          result = predict_words(words)
          st.write_stream(generate_stream(result))
      

if __name__ == '__main__':
  main()
