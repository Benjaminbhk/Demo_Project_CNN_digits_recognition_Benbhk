# function to maximize image contrast.
# Thus, the performance of the model will be
# greater with the use of the hand drawing app (see the final app)

def fun(e):
    if e > 50:
        return 255
    return 0

# function to normalize the pixels

def img_normalizer(X):
    return X / 255 - 0.5
