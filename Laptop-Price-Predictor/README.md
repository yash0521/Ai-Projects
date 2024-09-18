# Laptop Price Predictor

## Overview

The **Laptop Price Predictor** is a machine learning project that predicts the price of a laptop based on various features provided by the user. The project includes a predictive model and a user-friendly interface created using Streamlit.

## Features

- **Input Features:**

  - Company
  - TypeName
  - Ram
  - Weight
  - Touchscreen
  - IPS
  - Screen resolution
  - Screen size
  - CPU brand
  - HDD
  - SSD
  - GPU
  - OS type

- **Output:**
  - Predicted laptop price

## Getting Started

To run the application, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/DhruvalBhuva/Laptop-Price-Predictor.git
   cd laptop-price-predictor
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
    streamlit run app.py
   ```

4. Open your web browser and navigate to [http://localhost:8501](http://localhost:8501) to use the application.

## Usage

1. Launch the Streamlit app by running `streamlit run app.py`.

2. Input the relevant information about the laptop in the provided fields.

3. Click the "Predict Price" button to get the predicted price for the given laptop specifications.

## Data

The project utilizes a dataset containing information on various laptops, including features such as Company, TypeName, Ram, Weight, Touchscreen, IPS, Screen resolution, Screen size, CPU brand, HDD, SSD, GPU, and OS type.
