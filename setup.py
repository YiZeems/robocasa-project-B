"""Setuptools metadata for editable installation of project package."""

from setuptools import find_packages, setup


setup(
    name="robocasa-telecom",
    version="0.1.0",
    description="RoboCasa Telecom project package",
    packages=find_packages(include=["robocasa_telecom", "robocasa_telecom.*"]),
    include_package_data=True,
)
