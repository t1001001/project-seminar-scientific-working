from setuptools import setup, find_packages

setup(
    name="project_seminar_scientific_working",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "project=main:main"
        ]
    }
)