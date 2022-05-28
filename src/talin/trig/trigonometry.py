from math import radians
import numpy as np

__all__ = ["linear_angle"]


def linear_angle(slope, units="deg"):
    """Returns the angle of the line specified by the slope, often referred to as Linear Regression Angle

    Args:
        slope (Number): Slope of the line
        units (str, optional): "deg" or "rad" for either degrees or radians. Defaults to "deg".

    Raises:
        ValueError: Invalid unit specified

    Returns:
        Number: Angle of linear line in either degrees (default) or radians
    """

    if units != "deg" and units != "rad":
        raise ValueError(
            "Units must be either 'deg' (for degrees) or 'rad' (for radians).")

    radians = np.arctan(slope)

    if units == "rad":
        return radians
    else:
        return np.degrees(radians)
