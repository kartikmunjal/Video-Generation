from setuptools import setup, find_packages

setup(
    name="video-generation-rlhf",
    version="0.1.0",
    description="Video generation fine-tuning with RLHF and preference alignment",
    author="Kartik Munjal",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "diffusers>=0.27.0",
        "transformers>=4.40.0",
        "peft>=0.10.0",
        "accelerate>=0.28.0",
        "imageio>=2.34.0",
        "opencv-python>=4.9.0",
        "einops>=0.7.0",
        "lpips>=0.1.4",
        "gradio>=4.20.0",
        "numpy>=1.26.0",
        "pandas>=2.1.0",
        "PyYAML>=6.0.1",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": ["jupyter", "matplotlib", "seaborn", "plotly", "wandb"],
    },
)
