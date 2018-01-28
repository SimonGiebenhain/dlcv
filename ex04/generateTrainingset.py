import numpy as np

# This function uses the provided data to generate two [?, 28, 28, 1] matrices, one with the cyrillic images and one
# with the latin images. Each image pair (cyrillic + latin) is randomized such that the labels/classes are equal.
# The sizeMultiplier parameter determines how large the returned dataset is (with how many latin images each cyrillic
# image gets paired). A sizeMultiplier of 4 means that each cyrillic image is copied 4 times into the new cyr-matrix
# with 4 random latin images of the same label/class being copied at the same indexes in the latin-matrix. Both
# matrices will roughly have 4x the size than the original data (4*~5000=20000)
# The function includes standardization (zero mean and unit standarddeviation) of the images
def genererateTrainingset(sizeMultiplier):
    # Load dataset
    X_cyr = np.load('data/X_cyr.npy') #shape = (28, 28, 5798)
    labels_cyr = np.load('data/labels_cyr.npy') #shape = (5798,)
    X_lat = np.load('data/X_lat.npy') #shape = (28, 28, 6015)
    labels_lat = np.load('data/labels_lat.npy')

    # REORDER DIMENSIONS OF DATA
    X_cyr = np.transpose(X_cyr, (2,0,1))
    X_lat = np.transpose(X_lat, (2,0,1))

    # STANDARDIZE DATA
    for i in range(0, len(X_cyr)-1):
        X_cyr[i] = (X_cyr[i] - np.mean(X_cyr[i]) / np.std(X_cyr[i]))
    for i in range(0, len(X_lat)-1):
        X_lat[i] = (X_lat[i] - np.mean(X_lat[i]) / np.std(X_lat[i]))

    # MATCH TRAINING DATA (AND EXTEND TRAINING SET)
    nr_classes = len(np.unique(labels_cyr)) # nr. of classes is equal in both label arrays
    stepsCyr = np.insert(np.where(labels_cyr[:-1] != labels_cyr[1:])[0], 0, 0)
    stepsLat = np.insert(np.where(labels_lat[:-1] != labels_lat[1:])[0], 0, 0)
    amountCyr = np.insert(np.diff(stepsCyr), 0, stepsCyr[1])
    amountLat = np.insert(np.diff(stepsLat), 0, stepsLat[1])
    sum = 0
    for i in range(0, nr_classes-1):
        sum += min(amountCyr[i], amountLat[i])
    cyrData = np.empty((sum*sizeMultiplier, 28, 28, 1)) # create dataset with 4x the training data size
    latData = np.empty((sum*sizeMultiplier, 28, 28, 1))
    curr_pos = 0
    for _ in range(sizeMultiplier):
        for i in range(0, nr_classes-1):
            # check which side has less images
            if amountCyr[i] < amountLat[i]:
                permuted_indexes = np.random.permutation(amountLat[i])
                for j in range(0, amountCyr[i]-1):
                    cyrData[curr_pos,:,:,0] = X_cyr[stepsCyr[i]+j]
                    latData[curr_pos,:,:,0] = X_lat[permuted_indexes[j]+stepsLat[i]]
                    curr_pos += 1
            else:
                permuted_indexes = np.random.permutation(amountCyr[i])
                for j in range(0, amountLat[i]-1):
                    cyrData[curr_pos,:,:,0] = X_cyr[permuted_indexes[j]+stepsCyr[i]]
                    latData[curr_pos,:,:,0] = X_lat[stepsLat[i]+j]
                    curr_pos += 1

    return cyrData, latData