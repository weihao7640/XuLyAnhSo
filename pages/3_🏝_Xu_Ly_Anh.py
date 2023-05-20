import base64
from io import BytesIO
import streamlit as st
import cv2
import numpy as np
from io import BytesIO

from ThuatToanXuLyAnh import Algorithm as al
from PIL import Image

#al = Algorithm()
menu_dict = {
    "NotThing" : [al.foo, "RGB"],
    #Chuong 3
    "Negative": [al.Negative, "L"],  
    "Logarit": [al.Logarit, "L"], 
    "Power": [al.Power, "L"], 
    "Histogram": [al.Histogram, "L"],
    "Hist Equal": [al.HistEqual, "L"],
    "Hist Equal Color" : [al.HistEqualColor, "RGB"],
    "Local Hist": [al.LocalHist, "L"],
    "Hist Stat": [al.HistStat, "L"],
    "My Box Filter": [al.MyBoxFilter, "L"],
    "Box Filter": [al.BoxFilter, "RGB"],
    "Threshold": [al.Threshold, "RGB"],
    "Median Filter": [al.MedianFilter, "L"],
    "Sharpen": [al.Sharpen, "RGB"],
    "Gradient": [al.Gradient, "RGB"],
    #Chuong 4
    "Spectrum": [al.Spectrum, "L"],
    "Frequency Filter": [al.FrequencyFilter, "L"],
    "Remove Moire": [al.RemoveMoire, "L"],
    #Chuong 5
    "Create Motion Noise": [al.CreateMotionNoise, "L"],
    "Denoise Motion": [al.DenoiseMotion, "L"],
    #Chuong 9
    "Boundary": [al.Boundary, "RGB"],
    "Hole Fill": [al.HoleFill, "L"],
    "My Connected Components": [al.MyConnectedComponent, "L"],
    "Connected Components": [al.ConnectedComponent, "L"],
}

def Choose_Algorithm(algorithm, image):
    image = np.array(image.convert(menu_dict[algorithm][1]))
    new_img = menu_dict[algorithm][0](image)
    new_img = st.image(new_img)
    return new_img

def main():
    st.title('Xử Lý Ảnh')
    sidebar = st.sidebar
    selection = sidebar.selectbox("Chọn thuật toán xử lý ảnh", menu_dict.keys(), key='c3')

    image = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg', 'webp'])            

    if image is None:
        image = 'vango.jpg'
    image = Image.open(image)
    new_img = Choose_Algorithm(selection, image)

if __name__ == '__main__':
    main()
    