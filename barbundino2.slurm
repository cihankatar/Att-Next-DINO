#!/bin/bash

#SBATCH -p barbun-cuda      							# Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A ckatar          							# Kullanici adi
#SBATCH -J Dinobarbuntraining2  								# Gonderilen isin ismi
#SBATCH -o dinobarbun2.out			  			        	# Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:1        							# Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                							# Gorev kac node'da calisacak?
#SBATCH -n 1                							# Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 20							# Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=24:00:00    							# Sure siniri koyun. (HH:MM:SS)
#SBATCH --mail-type=begin        						# send email when job begins
#SBATCH --mail-type=end          						# send email when job ends
#SBATCH --mail-user=cihan.katar@tau.edu.tr

eval "$(/truba/sw/centos7.3/lib/anaconda3/2021.11/bin/conda shell.bash hook)"

conda activate segmentation
cd /truba/home/ckatar/Desktop/Att-Next-DINO
python train_ssl_pretrained2.py -p=False -i=256 -b=8 -x=0.5 -o=0.5 -e=202 -l=0.0001 -r=0.1