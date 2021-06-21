import numpy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return ds


def image2vector(image):
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return v


def normalize_rows(x):
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    x = x/x_norm
    return x


def softmax(x):
    x_exp = np.exp(x)
    X_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp/x_sum
    return s


def L1(yhat, y):
    loss = np.sum(abs(y-yhat))
    return loss


def L2(yhat, y):
    loss = np.sum(np.power((y-yhat),2))
    return loss


