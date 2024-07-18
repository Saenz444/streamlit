from setuptools import setup

setup(
    name='mi_proyecto',
    version='0.1',
    install_requires=[
        # Otras dependencias
    ],
    extras_require={
        'dev': [
            'pywin32==300; platform_system=="Windows"',
        ],
    },
)