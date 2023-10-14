from taipy.gui import Gui

# Definition of the page
page1_md = """

# **PresentAR**{: .color-primary} - the modernized presentation assistant

Welcome to PresentAR! We use [Taipy](https://taipy.io/) with a [custom GUI component](https://docs.taipy.io/en/latest/manuals/gui/extension/) to capture video from your webcam and do realtime presentation feedback 

<|layout|columns=1 1|
<|
## Are you ready to improve your presentation skills?
|>

<|
## Features included:

- Crossed Arms
- Disengaged
- Smiling
|>
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
    "app": page1_md,
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
    gui.run()
