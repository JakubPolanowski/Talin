from src.talin import trig
from src.talin.trig import trigonometry
import random
import numpy as np
import pytest

slope = random.randrange(0, 999)


def test_linear_angle():
    assert trigonometry.linear_angle(slope) == np.degrees(np.arctan(slope))


def test_linear_angle_deg():
    assert trigonometry.linear_angle(
        slope, "deg") == np.degrees(np.arctan(slope))


def test_linear_angle_rad():
    assert trigonometry.linear_angle(slope, "rad") == np.arctan(slope)


def test_linear_angle_invalid_unit():
    with pytest.raises(ValueError) as excinfo:
        trigonometry.linear_angle(slope, "notAUnit")
