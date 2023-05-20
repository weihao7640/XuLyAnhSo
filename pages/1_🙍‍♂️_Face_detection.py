import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from urllib.error import URLError
from PIL import Image, ImageEnhance

import cv2 as cv


scale = 1.0
score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000

## [initialize_FaceDetectorYN]
detector = cv.FaceDetectorYN.create(
    'pages/model/face_detection_yunet_2022mar.onnx',
    "",
    (320, 320),
    score_threshold,
    nms_threshold,
    top_k
)

tm = cv.TickMeter()

if 'stop' not in st.session_state:
    st.session_state.stop = False

def visualize(input, faces, fps, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            #print('Face {}, top-left coordinates: ({:.0f}, {:.0f}), box width: {:.0f}, box height {:.0f}, score: {:.2f}'.format(idx, face[0], face[1], face[2], face[3], face[-1]))

            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def camera_reg(FRAME_WINDOW):
    deviceId = 0
    cap = cv.VideoCapture(deviceId)
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)*scale)
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)*scale)
    detector.setInputSize([frameWidth, frameHeight])

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()    
        if not hasFrame:
            print('No frames grabbed!')
            break

        frame = cv.resize(frame, (frameWidth, frameHeight))

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()

        # Draw results on the input image
        visualize(frame, faces, tm.getFPS())

        # Visualize results
        
        FRAME_WINDOW.image(frame, channels='BGR')
        #cv.imshow('Live', frame)
    cv.destroyAllWindows()

def picture_reg(picture):
    picture = Image.open(picture)
    img1 = np.array(picture.convert('RGB'))

    img1Width = int(img1.shape[1]*scale)
    img1Height = int(img1.shape[0]*scale)

    img1 = cv.resize(img1, (img1Width, img1Height))
    tm.start()

    ## [inference]
    # Set input size before inference
    detector.setInputSize((img1Width, img1Height))

    faces1 = detector.detect(img1)
    ## [inference]

    tm.stop()
    if faces1[1] is None:
        st.warning('Không tìm thấy khuôn mặt trong hình')

    # Draw results on the input image
    visualize(img1, faces1, tm.getFPS())

    # Visualize results in a new window
    st.image(img1)

def main():
    st.markdown("# Phát hiện khuôn mặt")
    st.sidebar.header("Phát hiện khuôn mặt")

    picture = st.file_uploader("Chọn file để upload", type=['jpg', 'png', 'jpeg', 'webp'])
    container = st.empty()

    if picture:
        picture_reg(picture)
    
    FRAME_WINDOW = st.image([])
    
    if st.session_state.stop:
        camera = container.button("Bật camera")
        if camera:
            container.empty()
            camera = container.button("Tắt camera")
            st.session_state.stop = False
            camera_reg(FRAME_WINDOW)
    else:
        container.empty()
        camera = container.button("Bật camera")
        FRAME_WINDOW = st.image([])
        st.session_state.stop = True


if __name__ == '__main__':
    main()