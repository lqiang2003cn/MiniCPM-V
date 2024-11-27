wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -P /root
bash /root/Anaconda3-2024.02-1-Linux-x86_64.sh -b
/root/anaconda3/bin/conda init && source /root/.bashrc
/root/anaconda3/bin/conda create -n MiniCPM-V python=3.10 -y