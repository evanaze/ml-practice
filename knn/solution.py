import numpy as np
from numpy.core.fromnumeric import sort
import pandas as pd
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


def predict(train, test, k=5, metric = "euclidean", outcome = "OUTCOME"):
    # Drop the outcome column from observations
    X_train, y_train = train.drop(outcome, axis=1), train[outcome]
    X_test, y_test = test.drop(outcome, axis=1), test[outcome]
    # Calculate the distance for each test item
    if metric == "euclidean":
        distances = euclidean_distances(X_train, X_test)
    elif metric == "manhattan":
        distances = manhattan_distances(X_train, X_test) 
    else:
        print("Invalid distance metric.")
        exit(1)
    # Iterate through each obsersvation in the test data
    # to select the k most similar individuals.
    result = []
    for obs in range(len(distances[0])):
        # Get the array of distances for that individual
        obs_distances = pd.Series(distances[:, obs])
        # If there are any zero distance individuals, or answer is only those.
        if 0 in obs_distances:
            top_individuals = obs_distances[obs_distances == 0]
        # Get the indexes of the smallest distance neighbors
        sorted_distances = obs_distances.sort_values(ascending=True)
        # Get the top k individuals, while allowing for ties
        i, obs_distance = 0, 0
        while True:
            # Record the previous distance
            prev_distance = obs_distance
            obs_distance = sorted_distances.iloc[i]
            # Check if we are ending the count on a tie
            if obs_distance == prev_distance or i < k:
                i += 1
            else:
                break
        top_individuals = sorted_distances.iloc[:i]
        # Compute the predicted outcome
        inv_distances = 1/top_individuals
        result.append(np.dot(inv_distances, y_train[top_individuals.index])/np.sum(inv_distances))
    return result