import numpy as np


class NILFunction():
    
    def age_NIL(x, b0, b1, b2):
        age, NIL = x
        return b0 + b1 * age + b2 * NIL

    def age_NIL_kurtosis(x, b0, b1, b2, b3):
        age, NIL, kurtosis = x
        return b0 + b1 * age + b2 * NIL + b3 * kurtosis 