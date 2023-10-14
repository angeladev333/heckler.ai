from taipy.gui import Gui, notify, navigate

# Gui(page="# Getting Started\n\nHello, world!").run()  # use_reloader=True

text = "Original Text"

# Definition of the page
page1_md = """

# PresentAR - the modernized presentation assistant

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

My text: <|{text}|>

<|{text}|input|>

<|Try it out!|button|on_action=on_button_action|>
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


def on_button_action(state):
    notify(state, 'info', f'The text is: {state.text}')
    state.text = "Button Pressed"


def on_change(state, var_name, var_value):
    if var_name == "text" and var_value == "Reset":
        state.text = ""
        return


Gui(pages=pages).run()
