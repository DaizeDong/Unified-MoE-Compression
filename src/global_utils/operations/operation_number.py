from typing import Union


def normalize_value(value: Union[int, float], mean, std):
    if std != 0:
        return (value - mean) / std
