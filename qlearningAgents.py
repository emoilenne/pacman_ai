# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qvals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if (state, action) not in self.qvals:
          self.qvals[(state, action)] = 0.0
        return self.qvals[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
          return 0.0
        tmp = util.Counter()
        for action in legalActions:
          tmp[action] = self.getQValue(state, action)
        return tmp[tmp.argMax()]

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        legalActions = self.getLegalActions(state)
        if len(legalActions) == 0:
          return None
        tmp = util.Counter()
        for action in legalActions:
          tmp[action] = self.getQValue(state, action)
        actions = tmp.argMax(allMax=True)
        return random.choice(actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) != 0:
          if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
          else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        self.qvals[(state,action)] =  ((1-self.alpha) * self.getQValue(state,action)) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        feats = self.featExtractor.getFeatures(state, action)

        qval = 0
        for f in feats:
            qval += feats[f] * self.getWeights()[f]
        return qval

    # def calculateQValsPercentage(self, state):
    #     qvals = util.Counter()
    #     qvalsPerc = util.Counter()
    #
    #     # get qvals for a state
    #     legalActions = self.getLegalActions(state)
    #     for action in legalActions:
    #       qvals[action] = self.getQValue(state, action)
    #       if qvals[action] < 0: qvals[action] = 0.0
    #
    #     # compute sum and determine values
    #
    #     qvalsSum = sum(qvals.values())
    #     if not qvalsSum: qvalsSum = 1.0
    #     for action in qvals:
    #         qvalsPerc[action] = qvals[action] / qvalsSum * 100
    #     return qvalsPerc

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        feats = self.featExtractor.getFeatures(state, action)
        # ------------ LOG -------------

####### TODO
#
#       optimization with getQValue
#       display training
#
#
#


        # qvalsPerc = self.calculateQValsPercentage(state)
        qvals = util.Counter()
        legalActions = self.getLegalActions(state)
        for act in legalActions:
          qvals[act] = self.getQValue(state, act)

        import os
        os.system('clear')
        print "--- Update [%d] ---" % self.episodesSoFar

        green = '\x1b[1;32;40m'
        red = '\x1b[1;31;40m'
        yellow = '\x1b[1;33;40m'
        nocolor = '\x1b[0m'

        color = lambda x: green if x > 0 else yellow if x == 0 else red

        directions = {'West': 'Left', 'East': 'Right', 'North': 'Up', 'South': 'Down', 'Stop': 'Stop'}
        dirs = ['West', 'North', 'East', 'South', 'Stop']
        for act in dirs:
            actStr = directions[act]
            if actStr == directions[action]:
                actStr = green + actStr + nocolor
            print '%s: %s%f%s' % (actStr, color(qvals[act]), qvals[act], nocolor) #, qvalsPerc[act])
        for stateW in self.getWeights().sortedKeys():
            print "State ", stateW, " has weight %s%f%s" % (color(self.getWeights()[stateW]), self.getWeights()[stateW], nocolor)
        for name in feats.sortedKeysByName():
            if name == "bias": continue
            print "%s: %s%f%s" % (name, color(feats[name]), feats[name] * 100, nocolor)

        # ------------ LOG -------------

        for f in feats:
          self.weights[f] = self.weights[f] + self.alpha * feats[f]*((reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action))

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            for state in self.getWeights().sortedKeys():
                print "State ", state, " has weight ", self.getWeights()[state]

            pass
