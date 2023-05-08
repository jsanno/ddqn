# DDQN-based optimal targeted therapy
## Introduction
Code used in paper called "DDQN-based optimal targeted therapy with reversible inhibitors to combat the Warburg effect"

## Launch
This project requires Python 3.10. To run this project, create a virtualenv (recomended) and then install the requirements as:


$ pip install -r requirements.txt


To show the results obtained in the paper using the pretrained weights:

$ python plot_results.py

To train your own weights:

$ python train_ddqn.py

Note that training your own weights may take several hours, depending on the configuration of your computer and the number of threads that you set in the script train_ddqn.py (the code uses ten thread by default).
