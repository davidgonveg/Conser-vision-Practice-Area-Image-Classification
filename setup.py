from setuptools import setup, find_packages

setup(
    name="tai-park-species-classification",
    version="0.1.0",
    description="Camera trap species classification for Taï National Park",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "albumentations>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "python-dotenv>=0.19.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "monitoring": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=scripts.train_model:main",
            "evaluate-model=scripts.evaluate_model:main",
            "generate-submission=scripts.generate_submission:main",
        ],
    },
)
