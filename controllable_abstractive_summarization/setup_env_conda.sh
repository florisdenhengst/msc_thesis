conda create -n venv_train python=3.6.5 -y
source activate venv_train
conda install pytorch==1.1.0 cudatoolkit=9.0 -c pytorch
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm

