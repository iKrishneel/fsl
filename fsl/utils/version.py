#!/usr/bin/env python

from packaging import version


def major_version(version_string: str) -> int:
    parsed_version = version.parse(version_string)
    return int(parsed_version.release[0])


def minor_version(version_string: str) -> int:
    parsed_version = version.parse(version_string)
    return int(parsed_version.release[1])
