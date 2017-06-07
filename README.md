# pacman_ai
This project shows Pacman AI agent, implemented using deep reinforcement algorithm.

To run a program:

50 episodes of training, 60 episodes total (standard agent):
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic

500 episodes of training, 510 episodes total (advanced agent):
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumClassic
