# Weather Prediction System for Smart Agriculture Startup

This repository is created as an answer to **Question 1 (Q1)** of **IntelliHack 5.0**. It contains the code and reports related to the weather prediction system that uses minute-by-minute data aggregation and machine learning to predict the next 21 days of rain probability.

## Files in the Repository

- **Q1_part1.ipynb**: This Jupyter notebook contains the workbook for **Part 1** of the project, including data preprocessing, exploratory data analysis (EDA), and the initial steps to train a prediction model.

- **Q1_part2.ipynb**: This Jupyter notebook contains the workbook for **Part 2** of the project, where real-time data handling and the weather prediction system are implemented. It also includes handling sensor malfunctions and making predictions for the next 21 days.

- **Q1_app.py**: This is the Python app that uses **Streamlit** to provide a web interface for users. The app shows the weather predictions and visualizations interactively.

- **Report_1**: Contains the detailed report for **Part 1** of the project, including the methodology, data processing, and model training.

- **Report_2**: Contains the detailed report for **Part 2** of the project, which focuses on real-time prediction and handling sensor malfunctions.

- **weather_data.csv**: The dataset used for training and testing the model, which includes  weather data such as temperature, humidity, wind speed, pressure, and rain status.

## How to Run the System

1. Install the necessary Python libraries:
   ```bash
   pip install streamlit pandas matplotlib scikit-learn
