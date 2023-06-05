import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time
import pandas as pd
from moviepy.editor import VideoFileClip
from base64 import b64encode
from pathlib import Path
from streamlit_player import st_player, _SUPPORTED_EVENTS

st.title('rPPG Visualizer')
avi_path = r'dataset\subject3\vid.avi'
mp4_path = r'dataset\subject3\vid.mp4'

try:
    video_file = open(mp4_path, 'rb')
except Exception as e:
    clip = VideoFileClip(avi_path)
    clip.write_videofile(mp4_path)
finally:
    video_file = open(mp4_path, 'rb')

def local_video(path, mime="video/mp4"):
    data = b64encode(Path(mp4_path).read_bytes()).decode()
    return [{"type": mime, "src": f"data:{mime};base64,{data}"}]


options = {
                "events": ["onPlay"] ,
                "progress_interval": 1000,
                "playing": True,
                "controls": st.checkbox("Controls", True),
                "playback_rate" : 1
        }

event = st_player(local_video(mp4_path), **options)

#st_player("http://localhost:8501/dataset/subject3/vid.mp4",,progress_interval=1000)

video_bytes = video_file.read()
#st.video(video_bytes )

df = pd.read_json(r'vizdata\f_vizdata.json')
x = df.shape[0]

progress_bar = st.progress(0)
status_text = st.empty()
#[gt_hr_peak, gt_hr_fft, pred_hr_fft,pred_hr_peak,signal_hr_peak, signal_hr_fft]

st.header("Predition")
st.text("PreditionPeak:" + str(df['HR'][2]) + " PreditionFFT:" + str(df['HR'][3]))
chart1 = st.line_chart([df['Prediction'][0]],width=10)

st.header("GroundTruth")
st.text("TruePeak:" + str(df['HR'][0]) + " TrueFFT:" + str(df['HR'][1]))
chart2 = st.line_chart([df['Label'][0]], width=10)

st.header("Signal")
st.text("SignalPeak:" + str(df['HR'][4]) + " SignalFFT:" + str(df['HR'][5]))
chart3 = st.line_chart([df['Signal'][0]], width=10)

if(event[0] == "onPlay"):
    for i in range(x):
        # Update progress bar.
        progress_bar.progress(i / x)

        new_rows_1 = [df['Prediction'][i]]
        new_rows_2 = [df['Label'][i]]
        new_rows_3 = [df['Signal'][i]]

        # Update status text.
        status_text.text(
            'Prediction data is: %s' % new_rows_1)
        status_text.text(
            'Label data is: %s' % new_rows_2)
        status_text.text(
            'Signal data is: %s' % new_rows_3)

        # Append data to the chart.
        chart1.add_rows(new_rows_1)
        chart2.add_rows(new_rows_2)
        chart3.add_rows(new_rows_3)

        # Pretend we're doing some computation that takes time.
        time.sleep(0.1)

status_text.text('Done!')


#st.balloons()