# hackthevalley

## Setup

with anaconda:

```
conda create -n ergonomy
conda env list
conda activate ergonomy
pip install mediapipe opencv-python

```

# Backend - Pose detection

The pose detection was trained during this hackathon on MediaPipe landmark connections

to run, run `face.py`

## Our Process
1. Capture Landmarks and Export to CSV
2. Train our custom model using SciKit Learn
    - Read in collected data and process it
    - Train Machine Learning Classification Models
    - Evaluate and serialize model with best performance
3. Make detections with trained model and give feedback!

## Features included:

- Crossed Arms
- Disengaged
- Smiling

# Installation

This demo requires Python 3.9 or 3.10. Python 3.11 is currently not supported by Taipy.

To install the dependencies:
```
pip install -r requirements.txt
```

## Building the Webcam component

```
cd frontend/webcam/webui
npm i
```

- Find the location of taipy-gui with the `find_taipy_gui_dir.py` script and run:

```
 npm i <path to taipy-gui>/webapp
```

- Change `webpack.config.js` with taipy-gui path and run:

```
npm run build
```


Finally, to run the demo:
```
python main.py
```