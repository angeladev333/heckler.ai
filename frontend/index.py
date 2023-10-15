from taipy.gui import Gui, notify, navigate
from webcam import Webcam
import cv2

import PIL.Image
import io

import logging
import uuid
from pathlib import Path
from demo.faces import detect_faces, recognize_face, train_face_recognizer

logging.basicConfig(level=logging.DEBUG)

training_data_folder = Path("images")

show_capture_dialog = False
capture_image = False
show_add_captured_images_dialog = False

labeled_faces = []  # Contains rect with label (for UI component)

captured_image = None
captured_label = ""


# Definition of the page
page = """
<|toggle|theme|>

<container|container|part|

# **PresentAR**{: .color-primary} - the modernized presentation assistant

Welcome to PresentAR! We use [Taipy](https://taipy.io/) with a [custom GUI component](https://docs.taipy.io/en/latest/manuals/gui/extension/) to capture video from your webcam and do realtime presentation feedback 

<br/>

<|layout|columns=1 1|
<|
### Are you ready to improve your presentation skills?
|>

<|
## Features included:

- Crossed Arms
- Disengaged
- Smiling
|>
|>

<|card|card p-half|part|
## **Webcam**{: .color-primary} component

<|text-center|part|
<|webcam.Webcam|classname=face_detector|id=my_face_detector|sampling_rate=100|>

|card>
|container>
|>

"""



page2_md = """
## Rules and Evaluations

We evaluate your presentation skills based on the following

1. Your posture (slouching)

2. Your body language (crossed arms, still gestures, positive gestures like using hand movements)

3. Your facial expressions (smiling vs stoic)
"""

pages = {
    "/": "<|toggle|theme|>\n<center>\n<|navbar|>\n</center>",
    "app": page,
    "rules": page2_md,
}


# def on_button_action(state):
#     notify(state, 'info', f'The text is: {state.text}')
#     state.text = "Button Pressed"


# def on_change(state, var_name, var_value):
#     if var_name == "text" and var_value == "Reset":
#         state.text = ""
#         return


if __name__ == "__main__":
    # # Create dir where the pictures will be stored
    # if not training_data_folder.exists():
    #     training_data_folder.mkdir()

    # train_face_recognizer(training_data_folder)

    gui = Gui(pages=pages)
    gui.add_library(Webcam())
    gui.run()
