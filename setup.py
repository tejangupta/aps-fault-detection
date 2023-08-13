from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return a list of requirements
    """
    with open(file_path) as file_obj:
        requirements = [req.replace('\n', '') for req in file_obj.readlines()]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='sensor',
    version='0.0.1',
    author='Tejan',
    author_email='tejangupta8@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)
