from taipy.gui import Gui, notify

# Gui(page="# Getting Started\n\nHello, world!").run()  # use_reloader=True

text = "Original Text"

# Definition of the page
page = """
<|toggle|theme|>

# PresentAR - the modernized presentation assistant

<|layout|columns=1 1|
<|
Are you ready to improve your presentation skills?
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


def on_button_action(state):
    notify(state, 'info', f'The text is: {state.text}')
    state.text = "Button Pressed"


def on_change(state, var_name, var_value):
    if var_name == "text" and var_value == "Reset":
        state.text = ""
        return


Gui(page).run()
