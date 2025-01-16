import numpy as np
from numpy import seterr
from typing import Optional
import traceback
import sys

seterr(all='raise')

def npsafe_divide(numerator, denominator, name=None):
    try:
        return np.where(
        np.greater(denominator, 0),
        np.divide(numerator, denominator),
        np.zeros_like(numerator))
    except Exception as e:
        print(e)
        traceback.print_exc(file=sys.stdout)



def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v