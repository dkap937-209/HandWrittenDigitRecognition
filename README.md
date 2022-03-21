# Hand Written Digit Recognition

## User Guide

### Installation Instructions

1. Clone this repo: 
   
    ```sh
    git clone https://github.com/COMPSYS-302-2021/project-1-team-39.git
    ```
2. Change directory into the cloned repo:
   
    ```sh
   cd project-1-team-39
   ```
3. Use the *Python* module *pip* to install project dependencies:

    ```sh
    pip install -r requirements.txt
    ```
    *-- or --*
    ```sh
    python -m pip install -r requirements.txt
    ```
4.  Change into the source code directory:

    ```sh
    cd PythonProject
    ```
5. Run the program using python:
    ```sh
   python main.py
   ```
### Using The Program

1. Select a model from the "Model Select" option

2. If this is the first time running the application: File>Train Model,
   and from here download the MNIST dataset and train the model once the 
   data has been downloaded. If data has been downloaded and model has been 
   trained previously "Load Model Cache" can be selected to avoid training times
   
3. On the main page draw a number from 0-9 within the black canvas and afterwards 
   select the "Recognise" for the program to predict the drawn number
   
4. To repeat Step 3, select "Clear" to empty the canvas


## Developer Notes

### Direcotry Strcuture

- `PythonProject/` contains the source code
- `PythonProject/DesignFiles/` contains the Qt Designer project files. The files in this
folder where the inital starting points for all GUI pages, but their output was greatly 
  modified by hand, and they do not represent the current state of the GUI.
- `PythonProject/Model_Cache/` this is the directory where the mode's cache goes after training.
It is empty in the git repo.
- `PythonProject/mnist_data/` is not present in the repo, but it is created whren the MNIST dataset is downloaded
from the application


