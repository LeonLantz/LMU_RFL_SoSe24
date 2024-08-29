# LMU_RFL_SoSe24

Thema 5: RL-Policy-Training unter Vermeidung von Zuständen mit sehr geringer Datendichte

## 🛠️ **Setup**

1) **Install PyENV**:  
   Follow the instructions on the [PyENV GitHub page](https://github.com/pyenv/pyenv) to install PyENV.

2) **Set Up Python 3.10.6**:  
   Ensure PyEnv has Python 3.10.6 installed (or any version > 3.8.0). Run the following commands:
   ```sh
   pyenv update
   pyenv install 3.10.6
   ```
3) **Clone the Repository**:  
   Clone this repository to your local machine:
   ```sh
   git clone https://github.com/LeonLantz/LMU_RFL_SoSe24
   ```
4) **Set Python Version**:  
   Create an instance of Python 3.10.6 for your project. This will generate a `.python-version` file:
   ```sh
   pyenv local 3.10.6
   ```
5) **Determine Python Path**:  
   Find the path to your Python 3.10.6 executable:
   ```sh
   pyenv which python
   ```
   Example output: `C:\Users\<USER>\.pyenv\pyenv-win\versions\3.10.6\python.exe`
6) **Create Virtual Environment**:  
   Create a virtual environment using the specified Python version:
   ```sh
   C:\Users\<USER>\.pyenv\pyenv-win\versions\3.10.6\python.exe -m venv .venv
   ```
7) **Activate the Virtual Environment**:  
   Activate your virtual environment:
   - On Windows:
     ```sh
     .venv\scripts\activate
     ```
   - On Linux/Mac:
     ```sh
     source .venv/bin/activate
     ```
8) **Install Required Packages**:  
   Install all necessary packages from `requirements.txt` into your virtual environment:
   ```sh
   pip install -r requirements.txt
   ```
8) **Your are done!**  
   Open the project in your preferred IDE (e.g., VS Code or JupyterLab) and start working!
