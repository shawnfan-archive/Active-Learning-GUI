import numpy as np
import keras.backend as K

def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

# Function to evaluate specific tissue (class) dice index
class tissue_dice(object):
    def __init__(self, class_id, tissue_name):
        self.class_id = class_id
        self.__name__ = tissue_name
        self.smoothing_factor = 1e-6

    # returns calculated tissue dice when called
    def __call__(self, y_true, y_pred):
        return self.tissue_dice(y_true, y_pred)

    # calculates tissue dice in Keras
    def tissue_dice(self, y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # calculates dice from true and predicted labels
        y_true_f = K.cast(K.equal(class_id_true, self.class_id), 'float32')
        y_pred_f = K.cast(K.equal(class_id_preds, self.class_id), 'float32')
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + self.smoothing_factor) / (K.sum(y_true_f) + K.sum(y_pred_f) + self.smoothing_factor)