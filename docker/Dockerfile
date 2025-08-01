# Dockerfile para Taï National Park Wildlife Classification
# Base: PyTorch con CUDA para computer vision

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Metadata
LABEL maintainer="Wildlife Classification Team"
LABEL description="Docker image for Taï National Park species classification"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /workspace

# Update system and install essential packages
RUN apt-get update && apt-get install -y \
    # System essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    htop \
    tree \
    vim \
    # Image processing libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install core Python packages for computer vision and ML
RUN pip install \
    # Core ML/DL
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    # Computer Vision
    opencv-python==4.8.1.78 \
    albumentations==1.3.1 \
    Pillow==10.0.1 \
    scikit-image==0.21.0 \
    # Data manipulation
    pandas==2.1.1 \
    numpy==1.25.2 \
    scipy==1.11.3 \
    # Visualization
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    plotly==5.17.0 \
    # ML utilities
    scikit-learn==1.3.0 \
    # Progress bars and utilities
    tqdm==4.66.1 \
    rich==13.6.0 \
    # Configuration
    pyyaml==6.0.1 \
    omegaconf==2.3.0 \
    # Experiment tracking
    wandb==0.15.12 \
    tensorboard==2.14.1 \
    # Development tools
    jupyter==1.0.0 \
    jupyterlab==4.0.6 \
    ipywidgets==8.1.1 \
    # Testing
    pytest==7.4.2 \
    pytest-cov==4.1.0 \
    # Code quality
    black==23.9.1 \
    isort==5.12.0 \
    flake8==6.1.0 \
    # Additional utilities
    click==8.1.7 \
    python-dotenv==1.0.0

# Install timm for state-of-the-art models
RUN pip install timm==0.9.7

# Install additional computer vision packages
RUN pip install \
    # Image augmentation
    imgaug==0.4.0 \
    # Model interpretability
    grad-cam==1.4.8 \
    # Fast image loading (fix version)
    PyTurboJPEG==1.7.1

# Create directories for the project
RUN mkdir -p /workspace/{data,src,scripts,configs,results,notebooks,tests}

# Set up Jupyter configuration
RUN jupyter lab --generate-config
RUN echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py
RUN echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

# Copy requirements.txt if it exists (will be overridden by mount)
COPY requirements.txt* ./

# Install additional requirements if file exists
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Set Python path
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Create non-root user for security (optional, comment out if you prefer root)
# RUN useradd -m -u 1000 developer && \
#     chown -R developer:developer /workspace
# USER developer

# Expose ports
EXPOSE 8888 6006 8097

# Default command
CMD ["bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" || exit 1