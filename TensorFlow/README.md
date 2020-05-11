### Work whith virtual environment

**1. Install virtualenv to be able to create a virtual environment**

$ pip install virtualenv

**2. Create a virtual environment and IMPORTANT: specify the python version. Make sure you are in the /TensorFlow directory.**

$ virtualenv venv --python=python3

**3. To activate the virtual environment**

$ source venv/bin/activate

**4. Double check everything went well, it should say python 3**

$ (venv)  python --version

**5. Install the requirements**

$ (venv)  pip install -r requirements.txt

### To deactivate the virtual environment (exit)
$ (venv)  deactivate


__Now you will succesfully run any python script in this package using python 3!__

### (Optional) If you install more dependencies, you need to add them to the requirements.txt
$ (venv) pip freeze >> requirements.txt
