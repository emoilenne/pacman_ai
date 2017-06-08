# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
from util import manhattanDistance as md
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return 0.0

def closestCapsule(pos, capsules, walls):
    """
    closestCapsule -- find closest capsule (max distance 5)
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a capsule at this location then exit
        if (pos_x, pos_y) in capsules or dist == 5:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no capsule found
    return 0.0

def smallestFoodPath(pos, food, walls):
    """
    smallestFoodPath -- determine how many food left on this path (up to 5)
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        expanded.add((pos_x, pos_y))
        # spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        countWithFood = 0
        for nbr_x, nbr_y in nbrs:
            if (nbr_x, nbr_y) not in expanded and food[nbr_x][nbr_y]:
                fringe.append((nbr_x, nbr_y, dist+1))
                countWithFood += 1
        if countWithFood == 0 and food[pos_x][pos_y] or dist == 5: # dist > 0 ?
            return dist
    # no food found
    return 5

def closestGhost(pos, ghosts, walls):
    """
    closestGhost -- find minimum distance to ghosts
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we found ghost then exit
        for g in ghosts:
            g_x, g_y = g.getPosition()
            if (int(g_x), int(g_y)) == (pos_x, pos_y):
                return (g, dist)
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no ghost found
    return (None, dist)

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        capsules = state.getCapsules()
        walls = state.getWalls()
        ghosts = state.getGhostStates()
        scaredGhosts = [g for g in ghosts if g.scaredTimer > 0]
        notScaredGhosts = [g for g in ghosts if g.scaredTimer == 0]



        features = util.Counter()

        # features["bias"] = 0.5

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum(g.getPosition() in Actions.getLegalNeighbors((next_x, next_y), walls) for g in notScaredGhosts)

        # eat scared ghost ( continue in "closest scared ghost")
        features["eats-scared-ghost"] = 0.0

        # closest scared ghost
        features["closest-scared-ghost"] = 0.0
        if scaredGhosts and not features["#-of-ghosts-1-step-away"]:
            closest, dist = closestGhost((next_x, next_y), scaredGhosts, walls)
            if not features["#-of-ghosts-1-step-away"] and dist < 1.0:
                features["eats-scared-ghost"] = 0.5
            if (closest.scaredTimer / 2.0 > dist) and closest.scaredTimer >= 2:
                features["closest-scared-ghost"] = (closest.scaredTimer / 2.0 - dist) / 50.0 # TODO fix
        shouldChase = features["closest-scared-ghost"] > 0.0


        # if there is no danger of ghosts then add the food feature
        features["eats-food"] = 0.0
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 0.1

        # eat small path (< 5) of food before if there is a choice
        features["eat-small-path-food"] = 0.0
        if features["eats-food"]:
            features["eat-small-path-food"] = (5.0 - smallestFoodPath((next_x, next_y), food, walls)) / 50.0


        dist = closestFood((next_x, next_y), food, walls)
        features["closest-food"] = 0.0
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # capsules
        features["capsule-nearby"] = 0.0
        if not features["#-of-ghosts-1-step-away"] and capsules:
            features["capsule-nearby"] = (5 - closestCapsule((next_x, next_y), capsules, walls)) / 10.0

        # stopped
        # features["stopped"] = 1.0 if action == 'Stop' else 0.0

        # decrement all other features if pacman is chasing ghost
        if shouldChase:
            features["eats-food"] /= 10.0
            features["closest-food"] /= 10.0
            features["capsule-nearby"] /= 10.0
            features["eat-small-path-food"] /= 10.0

        features.divideAll(10.0)
        return features
