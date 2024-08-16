# Sentiment Analysis with LSTM on Social Media Data

This project performs sentiment analysis on social media data using a deep learning model built with TensorFlow and Keras. The datasets used for this analysis include Twitter and Reddit data. We preprocess the text data, train an LSTM (Long Short-Term Memory) network to classify sentiments into **positive**, **neutral**, and **negative** categories, and evaluate the model's performance.

## Features

- **Data Preprocessing**: 
  - Combined datasets from Twitter and Reddit.
  - Cleaned the text data by removing URLs, mentions, HTML tags, numbers, punctuation, stopwords, and performing lemmatization.
- **Tokenization**: 
  - Tokenized the text data and applied padding to ensure uniform sequence length.
- **Model Architecture**: 
  - LSTM network with embedding layer, LSTM layer, and dense output layer.
- **Model Training**: 
  - Trained using categorical cross-entropy loss and RMSProp optimizer with early stopping.
- **Evaluation**: 
  - Evaluated performance using accuracy, loss metrics, and visualized training and validation loss.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- `pip` (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone <repository-link>
    cd <repository-folder>
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Place your datasets (`Twitter_Data.csv`, `Reddit_Data.csv`) in the specified directory or update the file paths in the script.

2. Run the Jupyter Notebook or Google Colab script to execute the following steps:
   - Load and preprocess the data.
   - Train the LSTM model.
   - Evaluate model performance and visualize results.

### Output

- **Model Performance**: Classification accuracy and confusion matrix.
- **Training Visualization**: Loss and validation loss plots over epochs.

## Data

The datasets used include:
- [Twitter Data]
- [Reddit Data]

## Disclaimer

The results obtained are based on the provided datasets and the specific models used in this script. Validation with additional data and possible model refinements are recommended for practical applications.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [NLTK](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)

## Contact

For questions or comments, please open an issue on GitHub.
