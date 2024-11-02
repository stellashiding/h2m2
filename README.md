# Text Summarizer and Analyzer

This project is a text summarizer and analyzer application that consists of a Python-based backend for processing text using AI models (OpenAI, Hugging Face) and a frontend served using Docker with Nginx.

## Table of Contents

1. [Backend Setup](#backend-setup)
   - [Install Python](#install-python)
   - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
   - [Install Backend Dependencies](#install-backend-dependencies)
   - [Configure Environment Variables](#configure-environment-variables)
   - [Run Backend Server](#run-backend-server)
2. [Frontend Setup](#frontend-setup)
   - [Install Docker](#install-docker)
   - [Build and Run Frontend Docker Image](#build-and-run-frontend-docker-image)
3. [Access Frontend](#access-frontend)
4. [Docker Commands Reference](#docker-commands-reference)
   - [Build Docker Image](#build-docker-image)
   - [Run Docker Container](#run-docker-container)
   - [Stop Docker Container](#stop-docker-container)
   - [Remove Docker Container](#remove-docker-container)
   - [List Docker Containers](#list-docker-containers)
   - [View Docker Logs](#view-docker-logs)
   - [Remove Docker Images](#remove-docker-images)
5. [Troubleshooting](#troubleshooting)

---

## Backend Setup

The backend is built using Python 3.12.3 and requires a virtual environment to manage dependencies. Follow the instructions for your operating system.

### Install Python

Ensure Python 3.12.3 is installed on your machine.

- **macOS**:  
  Install using [Homebrew](https://brew.sh/):
  ```bash
  brew install python@3.12
  ```
  
- **Ubuntu**:  
  Install from source:
  ```bash
  sudo apt update
  sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev wget
  wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
  tar -xf Python-3.12.3.tgz
  cd Python-3.12.3
  ./configure --enable-optimizations
  make -j 4
  sudo make altinstall
  ```

- **Windows**:  
  Download and install from the [official Python website](https://www.python.org/downloads/release/python-3123/).

### Create and Activate Virtual Environment

1. **Create the virtual environment**:
   - **macOS & Ubuntu**:
     ```bash
     python3.12 -m venv venv
     ```
   - **Windows**:
     ```bash
     python -m venv venv
     ```

2. **Activate the virtual environment**:
   - **macOS & Ubuntu**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```bash
     .\venv\Scripts\activate
     ```

### Install Backend Dependencies

Install the required dependencies from the `backend/requirements.txt` file.

```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the `backend/` directory and add your OpenAI API key.

1. Create the `.env` file:
   ```bash
   touch backend/.env
   ```

2. Add the following content to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

### Run Backend Server

Run the backend server by executing the `run.py` file:

```bash
python backend/run.py
```

The backend server will be running at `http://127.0.0.1:5000/`.

---

## Frontend Setup

The frontend is served via Docker using Nginx. Follow the instructions below to set up Docker and run the frontend container.

### Install Docker

- **macOS**:  
  Install Docker Desktop from [Docker’s official website](https://www.docker.com/products/docker-desktop).
  
- **Ubuntu**:
  ```bash
  sudo apt update
  sudo apt install -y docker.io
  sudo systemctl start docker
  sudo systemctl enable docker
  ```

- **Windows**:  
  Install Docker Desktop from [Docker’s official website](https://www.docker.com/products/docker-desktop).

### Build and Run Frontend Docker Image

1. **Build the Docker image**:
   Navigate to the root project directory and build the Docker image using the `frontend/Dockerfile`:

   ```bash
   sudo docker build -t frontend-app ./frontend
   ```

2. **Run the Docker container**:
   Start the container, exposing it on port 8080.

   ```bash
   sudo docker run -d -p 8080:80 --name frontend-app-container frontend-app
   ```

---

## Access Frontend

Once the frontend container is running, you can access the frontend application at:

[http://localhost:8080/](http://localhost:8080/)

---

## Docker Commands Reference

### Build Docker Image

To build the Docker image for the frontend:

```bash
sudo docker build -t frontend-app ./frontend
```

### Run Docker Container

To run a Docker container for the frontend:

```bash
sudo docker run -d -p 8080:80 --name frontend-app-container frontend-app
```

### Stop Docker Container

To stop a running Docker container:

```bash
sudo docker stop frontend-app-container
```

### Remove Docker Container

To remove a stopped Docker container:

```bash
sudo docker rm frontend-app-container
```

### List Docker Containers

To list all Docker containers (including stopped ones):

```bash
sudo docker ps -a
```

To list only running Docker containers:

```bash
sudo docker ps
```

### View Docker Logs

To view the logs of a running Docker container:

```bash
sudo docker logs frontend-app-container
```

### Remove Docker Images

To remove a Docker image:

```bash
sudo docker rmi frontend-app
```

---

## Troubleshooting

### Common Issues and Solutions

1. **Backend Server Not Starting:**
   - Ensure that the virtual environment is activated.
   - Check for missing dependencies in `backend/requirements.txt`.
   
2. **Docker: Container Name Already in Use:**
   - If you see the error `Conflict. The container name "/frontend-app-container" is already in use`, remove or rename the existing container:
     ```bash
     sudo docker rm frontend-app-container
     ```

3. **Cannot Access Frontend at `http://localhost:8080/`:**
   - Ensure the frontend Docker container is running:
     ```bash
     sudo docker ps
     ```
   - Verify that the port mapping is correct (`8080:80`).

4. **CORS Issues Between Frontend and Backend:**
   - Ensure the backend is configured to handle CORS properly. Update `backend/app/__init__.py` with appropriate CORS settings.

---

## Additional Notes

- Ensure Docker is running before attempting to build or run the frontend container.
- If the backend or frontend fails to start, check the logs for more information.
- When running multiple Docker containers, consider using `docker-compose` for easier management.
