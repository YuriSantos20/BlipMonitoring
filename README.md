Este projeto requer a criação de um ambiente Python isolado com a versão 3.11. 
Recomendo o uso do Anaconda Prompt.

### 1. Crie o ambiente virtual

Abra o **Anaconda Prompt** e execute:

conda create -n BL2 python=3.11

conda activate BL2

### 2. Conferir a versao do cuda pelo prompt

nvidia-smi

### Caso a versao do cuda seja 12.6:

conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia

### Caso nao tenha CPU:

conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 cpuonly -c pytorch

### 3. Instalar Bibliotecas

pip install transformers==4.46.2
pip install sympy==1.13.1
