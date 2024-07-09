# Project Description

This project is based on the previous project made Python-PostgreSQL-CRUD-AP, made a neural network´s prediction model for the salary with the postgreSQL database cleaned in the previous project and made a fastAPI API in order to interact and make the prediction.

## Tasks Overview

1. Extract Data from PostgreSQL: Use Python to extract data from a PostgreSQL database, the database cleaned from the previous project.
2. Prepare the data in order to make the neural networks
3. Train the model
4. Make a function in order to interact with the model
5. Make FastAPI API to interact with the model

## Project Structure

- README.md: This document provides an overview of the project and its tasks.
- requirements.txt: Includes all dependencies needed for the project.
- main.py: Contains the main code for the project.
- database: 
    - train.py: Contains model training and the function to interact with the model
    - database.py: This file is in charge of downloading the DataBase from the PostgreSQL instance and using pandas to transform it into a DataSet for processing.
- utils:
    - models.py: In this file is the request model from the API used to run main.py.

## Usage Instructions

1. Setup Environment: Ensure Python and the required dependencies from requirements.txt are installed, for install the dependencies use "pip install -r requirements.txt".
2. Run main.py: Execute main.py to initiate the project, use "uvicorn main:app --reload" to initiate the project.
3. Enter the localHost to begin working with the database and the functions of the CRUD, use the URL: "http://127.0.0.1:8000/docs" or access the endpoints.
4. The FastAPI interface will request information about the database in order to carry out any of the requests.
5. To use the GPT function, input the desired prompt. It's best to specify the database request type in the prompt. For reading and updating, include relevant information (e.g., age, location, salary). For reading and deleting, provide the row ID and specify the action (create, read, update, or delete).