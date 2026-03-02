# Semantic-Learning: Fake News Classification

This repository contains a neural network pipeline designed to classify textual data as either "Fake" or "Not Fake". Originally developed in a Jupyter Notebook (`bot.py`), the codebase has been modularized into separate Python files to enhance readability, maintainability, and deployment readiness.

## Project Structure

The project is structured into the following modules:

*   **`eda.py`**: Performs exploratory data analysis (EDA). It loads the dataset, prints basic statistics, plots the distribution of word lengths for both fake and real news, and generates word clouds to visualize frequent terms.
*   **`preprocessing.py`**: Handles text cleaning and preparation. It removes special characters and stopwords, converts text to lowercase, and applies stemming and lemmatization. It also tokenizes the text, pads the sequences for neural network input, and saves the tokenizer (`tokenizer.pkl`) for later use.
*   **`model.py`**: Defines the artificial neural network architecture. It uses a Sequential model with an Embedding layer, SpatialDropout1D, and a Bidirectional LSTM followed by a Dense classification layer.
*   **`train.py`**: Executes the training workflow. It loads and splits the preprocessed data, builds the model, trains it over a specified number of epochs, saves the trained weights (`model.h5`), and plots the training/validation accuracy and loss curves.
*   **`predict.py`**: Runs inference on new test data. It loads the preprocessed test set and the saved model, generates predictions, and exports the results to a CSV file (`submission.csv`).

## Getting Started

1.  **Dependencies**: Install the required packages listed in `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Dataset**: Ensure you have the `Dataset/train.csv` and `Dataset/test.csv` files available in the root directory.

3.  **Training**: Run the training script to train the model and generate performance plots:
    ```bash
    python train.py
    ```

4.  **Prediction**: Generate predictions on the test set:
    ```bash
    python predict.py
    ```
