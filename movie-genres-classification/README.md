# Movie Genres Classification

## Overview

This project focuses on classifying movie genres using both Machine Learning (ML) and Deep Learning (DL) approaches. We have implemented distinct methodologies to tackle the classification task, resulting in different accuracies.

## Getting Started

To run the application, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/DhruvalBhuva/movie-genres-classification.git
   cd movie-genres-classification
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Approaches

### Deep Learning (DL)

In the DL approach, we employed advanced neural network architectures such as LSTM to extract intricate features from movie data. The DL model has demonstrated remarkable performance, achieving an accuracy of 94%. The high accuracy is attributed to the model's ability to capture complex patterns and relationships within the data.

For more details on the DL approach, refer to the `DL_Approach.ipynb` notebook.

### Machine Learning (ML)

In the ML approach, traditional machine learning algorithms were utilized to classify movie genres based on selected features. While the ML approach provided valuable insights, the accuracy reached 67%, indicating that the model might face challenges in capturing nuanced patterns present in the data.

For a deeper understanding of the ML approach, consult the `ML_Approach.ipynb` notebook.

## Repository Structure

- `DL_Approach.ipynb`: Jupyter notebook containing the Deep Learning approach.
- `ML_Approach.ipynb`: Jupyter notebook outlining the Machine Learning approach.
- `movie_dataset.txt`: File containing the data used for training and evaluation.
