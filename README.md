# Research workflow repository of the EOAFrica/Vidi GNSS-IR paper reproducable_research_template

# Instructions for reproducing the workflows

## 1. create a virtual environment with pip in the root of this repository
In a terminal (or through a notebook cell) execute the following
```
# 1. create a new virtual environment (possibly reuse existing packages with --system-site-packages)
python -m venv  --system-site-packages pyeoaf
# 2. activate your virtual environment
source pyeoaf/bin/activate
# 3. install the modules needed for this workflow
pip install -r requirements.txt
# 4. make the virtual environment available for use in your notebooks 
ipython kernel install --user --name=pyeoaf
```


