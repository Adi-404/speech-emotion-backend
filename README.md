# Emotion Recognition and Mental Health Response System

This project is a Flask-based web application that analyzes audio files to predict the user's emotion and provides a mental health response using the Gemini API.

## Features

- **Emotion Recognition**: Uses a pre-trained Convolutional Neural Network (CNN) model to predict the emotion from an audio file.
- **Speech-to-Text Conversion**: Converts the audio file to text using Google's Speech Recognition API.
- **Mental Health Response**: Generates a mental health response based on the user's emotion and query using the Gemini API.

### Setbacks
- **Length of Audio Files**: Do not give audio files smaller than 2.5s as we have considered minimum length to be of 2.5s and 2736 feat.

## Setup

### Prerequisites

- Python 3.9
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up the environment variables:
    - Create a [.env](http://_vscodecontentref_/1) file in the project root directory.
    - Add your Gemini API key to the [.env](http://_vscodecontentref_/2) file:
        ```env
        GEMINI_API_KEY="your_gemini_api_key"
        ```

### Running the Application

1. Start the Flask application:
    ```sh
    python app.py
    ```

2. The application will be available at `http://0.0.0.0:5050`.

### Using Docker

1. Build the Docker image:
    ```sh
    docker build -t emotion-recognition-app .
    ```

2. Run the Docker container:
    ```sh
    docker run -p 5050:5050 emotion-recognition-app
    ```

## API Endpoints

### Analyze Audio

- **Endpoint**: `/analyze_audio`
- **Method**: `POST`
- **Description**: Upload an audio file to analyze the emotion and get a mental health response.
- **Request**:
    - [file](http://_vscodecontentref_/3): The audio file to be analyzed.
- **Response**:
    ```json
    {
        "transcription": "Transcribed text from the audio",
        "emotion": "Predicted emotion",
        "gemini_response": "Mental health response from Gemini API"
    }
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](http://_vscodecontentref_/4) file for details.

## Acknowledgements

- Lords A & BUI for their contributions.
- The authors of the libraries and tools used in this project.

---
💻✨ **Crafted with ❤️, code, and coffee by [Adi](https://github.com/Adi-404)** ☕🚀