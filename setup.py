from setuptools import setup, find_packages

setup(
    name='geostatsapp', # Nome do seu pacote (tudo minúsculo, sem espaços)
    version='0.1.0',
    author='Seu Nome / Sua Equipe',
    description='Biblioteca de Geoestatística Otimizada com Numba e Panel',
    packages=find_packages(), # Encontra automaticamente a pasta com o __init__.py
    install_requires=[
        'pandas',
        'numpy',
        'numba',
        'scipy',
        'plotly',
        'panel',
        'jupyter_bokeh' 
    ],
    python_requires='>=3.8',
)
