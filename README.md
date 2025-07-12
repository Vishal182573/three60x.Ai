# three60x.Ai
# download the player_ball_detector.pt fine tuned yolo model from :- https://drive.google.com/file/d/1eDYM7HhvwXmbE39I5lBfyoHPkP0nglgN/view?usp=sharing and paste this model inside the models directory 
# setup the virtual environment before running the code by following commands :-
# 1) open terminal
# 2) enter this command :- python -m venv venv
# 3) activate venv using this command :- venv/Scripts/activate
# 4) install requirements :- pip install -r requirements.txt
# 5) run the pipeline file using this command :- python app.py (this will initiate the process of players mapping to id from both videos)

initial version by using pre trained model resnet50 for feature extraction for player appearance along with yolo model to detect players and ball in each frame