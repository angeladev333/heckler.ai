# 📸 Heckler.ai
## Hack the Valley First Place 🏆

## ✨ Inspiration
In the wake of a COVID-altered reality, where the lack of resources continuously ripples throughout communities, Heckler.AI emerges as a beacon for students, young professionals, those without access to mentorship and individuals seeking to harness vital presentation skills. Our project is a transformative journey, empowering aspiring leaders to conquer their fears. In a world, where connections faltered, Heckler.AI fills the void, offering a lifeline for growth and social resurgence. We're not just cultivating leaders; we're fostering a community of resilient individuals, breaking barriers, and shaping a future where every voice matters. Join us in the journey of empowerment, where opportunities lost become the stepping stones for a brighter tomorrow. Heckler.AI: Where anxiety transforms into confidence, and leaders emerge from every corner.

## ✨ What it does
Heckler.AI is your personalized guide to mastering the art of impactful communication. Our advanced system meticulously tracks hand and arm movements, decodes facial cues from smiles to disinterest, and assesses your posture. This real-time analysis provides actionable feedback, helping you refine your presentation skills. Whether you're a student, young professional, or seeking mentorship, Heckler.AI is the transformative tool that not only empowers you to convey confidence but also builds a community of resilient communicators. Join us in this journey where your every move becomes a step towards becoming a compelling and confident leader. Heckler.AI: Precision in every gesture, resonance in every presentation.

## ✨ How we built it
We used OpenCV and the MediaPipe framework to detect movements and train machine learning models. The resulting model could identify key postures that affect your presentation skills and give accurate feedback to fast-track your soft skills development. To detect pivotal postures in the presentation, we used several Classification Models, such as Logistic Regression, Ridge Classifier, Random Forest Classifier, and Gradient Boosting Classifier. Afterwards, we select the best performing model, and use the predictions personalized to our project's needs.

## ✨ Challenges we ran into
Implementing webcam into taipy was difficult since there was no pre-built library. Our solution was to use a custom GUI component, which is an extension mechanism that allows developers to add their own visual elements into Taipy's GUI. In the beginning we also wanted to use a websocket to provide real-time feedback, but we deemed it too difficult to build with our limited timeline. Incorporating a custom webcam into Taipy was full of challenges, and each developer's platform was different and required different setups. In hindsight, we could've also used Docker to containerize the images for a more efficient process.

Furthermore, we had difficulties deploying Taipy to our custom domain name, "hecklerai.tech". Since Taipy is built of Flask, we tried different methods, such as deploying on Vercel, using Heroku, and Gunicorn, but our attempts were in vain. A potential solution would be deploying on a virtual machine through cloud hosting platforms like AWS, which could be implemented in the future.

## ✨ Accomplishments that we're proud of
We are proud of ourselves for making so much progress in such little time with regards to a project that we thought was too ambitious. We were able to clear most of the key milestones of the project.

## ✨ What we learned
We learnt how to utilize mediapipe and use it as the core technology for our project. We learnt about how to manipulate the different points of the body to accurately use these quantifiers to assess the posture and other key metrics of presenters, such as facial expression and hand gestures. We also took the time to learn more about taipy and use it to power our front end, building a beautiful user friendly interface that displays a video feed through the user's webcam.

## ✨ What's next for Heckler.ai
At Heckler.AI, we're committed to continuous improvement. Next on our agenda is refining the model's training, ensuring unparalleled accuracy in analyzing hand and arm movements, facial cues, and posture. We're integrating a full-scale application that not only addresses your presentation issues live but goes a step further. Post-video analysis will be a game-changer, offering in-depth insights into precise moments that need improvement. Our vision is to provide users with a comprehensive tool, empowering them not just in the moment but guiding their growth beyond, creating a seamless journey from identifying issues to mastering the art of compelling communication. Stay tuned for a Heckler.AI that not only meets your needs but anticipates and addresses them before you even realize. Elevate your communication with Heckler.AI: Your growth, our commitment.



## ---------------------------------------------------------------------------------
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
