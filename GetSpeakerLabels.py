import sys
import numpy as np
from scipy import spatial


def NormalizeFeatureVectors(features):
    # "features" is a 2-D array with the i'th row containing embedded speakers vector for the i'th superframe.
    #  superframe 0  c0, c1, c2, ...., c259
    #  superframe 1  c0, c1, c2, ...., c259
    #  superframe 2  c0, c1, c2, ...., c259
    #      .
    #      .
    #      .
    #  superframe(N-1) c0, c1, c2, ...., c259
    #
    # Normalize features so that the max sum over a class is 1.0.
    # Get the sum over 0 to N-1 for each class ci.
    # Get the max sum.
    # Normalize all coefs in features array by this value.
    sumOverRows = np.sum(features, axis=0)
    maxSum = sumOverRows.max()
    features = features / maxSum
    return features


def CalculateSpeakerLabels(features, newSpeakerThreshold, sameSpeakerThreshold):

    """
    Using the feature vectors calculated (one feature vector per superframe), detect speaker changes
    and assign a speaker label (1, 2, ...) to each superframe.
    """

    # R cosine similarity to detect speaker changes.
    # Record the change as a sequence of speaker labels:
    #     1 for speaker 1, 2 for speaker 2, ...
    speakerLabels = [1]  # speaker labels sequence over the whole recording.
    speakerCount = 1  # total number of speakers.
    currentSpeaker = 1
    speakerFeatureDict = {}
    speakerFeatureDict[currentSpeaker] = features[0]  # Dictionary of (speaker label, last feature vector)

    featureCount = features.shape[0]  # Number of features.
    for i in range(1, featureCount):
        # Get cosine distance between last feature (i-1) and feature i.
        dist = spatial.distance.cosine(features[i - 1], features[i])
        if dist < newSpeakerThreshold:
            # no speaker change
            speakerLabels.append(currentSpeaker)
            speakerFeatureDict[currentSpeaker] = features[i]
        else:
            # Is this a new speaker or a speaker already seen?
            # Check distance between  features[i] and each previously seen speaker.
            minDist = sys.float_info.max
            minSpeakerLabel = -1
            for kv in speakerFeatureDict.items():
                dist = spatial.distance.cosine(features[i], kv[1])
                if dist < minDist:
                    minDist = dist
                    minSpeakerLabel = kv[0]

            if minSpeakerLabel > 0 and minDist < sameSpeakerThreshold:
                # This is a speaker change to a previously seen speaker
                currentSpeaker = minSpeakerLabel
            else:
                # We have a new speaker:
                speakerCount += 1
                currentSpeaker = speakerCount

            speakerLabels.append(currentSpeaker)
            speakerFeatureDict[currentSpeaker] = features[i]

    return speakerLabels