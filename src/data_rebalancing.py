import imblearn
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
import random
from collections import Counter


def naieve_oversample(x, y, seed=None):
    if seed is None:
        seed = random.randint()
    ros = RandomOverSampler(random_state=seed)
    x_resampled, y_resampled = ros.fit_resample(x, y)
    return x_resampled, y_resampled


def naieve_undersample(x, y, seed=None):
    if seed is None:
        seed = random.randint()
    rus = RandomUnderSampler(random_state=seed)
    x_resampled, y_resampled = rus.fit_resample(x, y)
    return x_resampled, y_resampled

def base_smote_oversample(x, y, seed=None):
    if seed is None:
        seed = random.randint()
    bsmote = SMOTE(random_state=seed)
    x_resampled, y_resampled = bsmote.fit_resample(x, y)
    return x_resampled, y_resampled


def balance_samples(x, y, seed=None, type="naieve_oversample"):
    if seed is None:
        seed = random.randint()
    if type is None:
        return x, y
    sample_types = {"naieve_oversample": RandomOverSampler,
                    "naieve_undersample": RandomUnderSampler,
                    "base_smote": SMOTE}
    if type not in sample_types.keys():
        raise ValueError("Invalid sampling type.")
    sample_class = sample_types[type]
    sampler = sample_class(random_state=seed)
    x_resampled, y_resampled = sampler.fit_resample(x, y)
    return x_resampled, y_resampled

