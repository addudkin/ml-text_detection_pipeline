conda deactivate
python3.8 -m venv "$(pwd)/dbrain_td"
source dbrain_td/bin/activate
python3.8 -m pip install --upgrade pip
pip3 install -r requirements.txt