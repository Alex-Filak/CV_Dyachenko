FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    vim \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set working directory
WORKDIR /app

# Install Python packages (optional)
RUN pip3 install --upgrade pip
# RUN pip3 install jupyterlab numpy pandas torch  # Uncomment as needed

# Expose port for Jupyter or other services
EXPOSE 8888

# Default command
CMD ["/bin/bash"]
