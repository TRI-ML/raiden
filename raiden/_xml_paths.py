"""Cached robot XML path helpers."""

import functools

from i2rt.robots.utils import (
    ARM_YAM_XML_PATH,
    GRIPPER_LINEAR_4310_PATH,
    combine_arm_and_gripper_xml,
)


@functools.lru_cache(maxsize=1)
def get_yam_4310_linear_xml_path() -> str:
    """Return path to a combined YAM arm + linear-4310 gripper XML (written to /tmp/)."""
    return combine_arm_and_gripper_xml(ARM_YAM_XML_PATH, GRIPPER_LINEAR_4310_PATH)
