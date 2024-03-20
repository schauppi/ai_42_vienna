# Project Installation Guide

This guide will help you install and run the project which is managed using Poetry, a dependency management tool for Python.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python (Check the required version in the project)
- [Poetry](https://python-poetry.org/docs/#installation)

## Create .env File

1. **Create a .env File**

   - Create a file named `.env` in the project directory.
   - Add the following environment variables to the file:

     ```bash
     # .env
     # Environment variables for the project

     # OpenAI API Key
        OPENAI_API_KEY=""
     # SerpAPI API Key
        SERPER_API_KEY=""
     ```

   - Replace the values with your own.

   - [SerpApi](https://serpapi.com/?gclid=CjwKCAiA0syqBhBxEiwAeNx9N5pBLmaXIF77gqaHvZWEJ-rEcEd6fQ-59mDVRxl0SD4-OX1KyvejUBoC5uYQAvD_BwE)
   - [OpenAI Api](https://openai.com/blog/openai-api)

## Installation Steps

1. **Clone or Download the Project**

   - Clone the project to your local machine, or download and extract the project archive.

2. **Navigate to Project Directory**

   - Open your terminal or command prompt.
   - Change directory to the project folder:
     ```bash
     cd path/to/project
     ```

3. **Install Dependencies**

   - Run the following command in the project directory:
     ```bash
     poetry install
     ```
   - This command installs all dependencies listed in `pyproject.toml`.

4. **Activate the Virtual Environment**

   - Poetry creates a virtual environment for the project. Activate it with:
     ```bash
     poetry shell
     ```

5. **Running the Scripts**

- Pose Estimation
  ```bash
  poetry run pose
  ```
- Object Detection
  ```bash
  poetry run object
  ```
- Depth Estimation
  ```bash
  poetry run depth
  ```
- Side by Side (Pose and Object Detection)
  ```bash
  poetry run combined
  ```
- LLM Chat
  ```bash
  poetry run chat
  ```
- LLM Agent
  ```bash
  poetry run agent
  ```

## Troubleshooting

- If you encounter installation errors, check the error messages for guidance or open an issue on the github repo.
- Verify your Python version is compatible with the project's requirements.

## Additional Commands

- To add a new dependency:
  ```bash
  poetry add package-name
  ```
