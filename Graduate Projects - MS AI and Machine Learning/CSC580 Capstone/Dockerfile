FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Optional: install additional Python packages your project needs
# Example:
# RUN pip install matplotlib pandas scikit-learn

WORKDIR /opt/project

COPY . /opt/project
