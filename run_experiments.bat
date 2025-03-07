@echo off
echo Запуск серии экспериментов для deepfake detection
echo Время начала: %date% %time%
echo. > experiment_runs.txt

:: Создаем папку для логов
mkdir logs 2>nul

:: Базовые эксперименты с разными моделями и функциями потерь
echo Запуск эксперимента 1/10
python train.py --model vits16_dino --loss triplet --epochs 20 --lr 1e-4 --margin 0.1 --seed 42 > logs\exp1_vit_triplet.log 2>&1
echo vits16_dino triplet lr=1e-4 margin=0.1 - %date% %time% >> experiment_runs.txt

echo Запуск эксперимента 2/10
python train.py --model vits16_dino --loss arcface --epochs 20 --lr 1e-4 --margin 0.2 --scale 30 --embedding_size 512 --seed 42 > logs\exp2_vit_arcface.log 2>&1
echo vits16_dino arcface lr=1e-4 margin=0.2 - %date% %time% >> experiment_runs.txt

echo Запуск эксперимента 3/10
python train.py --model swin --loss triplet --epochs 20 --lr 5e-5 --margin 0.1 --seed 42 > logs\exp3_swin_triplet.log 2>&1
echo swin triplet lr=5e-5 margin=0.1 - %date% %time% >> experiment_runs.txt

echo Запуск эксперимента 4/10
python train.py --model convnext --loss cosface --epochs 20 --lr 1e-4 --margin 0.2 --scale 30 --embedding_size 512 --seed 42 > logs\exp4_convnext_cosface.log 2>&1
echo convnext cosface lr=1e-4 margin=0.2 - %date% %time% >> experiment_runs.txt

:: Эксперименты с аугментациями
echo Запуск эксперимента 5/10
python train.py --model vits16_dino --loss triplet --epochs 20 --lr 1e-4 --margin 0.1 --advanced_augmentation --seed 42 > logs\exp5_vit_triplet_advaug.log 2>&1
echo vits16_dino triplet adv_aug - %date% %time% >> experiment_runs.txt

echo Запуск эксперимента 6/10
python train.py --model vits16_dino --loss triplet --epochs 20 --lr 1e-4 --margin 0.1 --mixup --mix_alpha 0.2 --seed 42 > logs\exp6_vit_triplet_mixup.log 2>&1
echo vits16_dino triplet mixup - %date% %time% >> experiment_runs.txt

:: Эксперименты с настройками дипфейка
echo Запуск эксперимента 7/10
python train.py --model vits16_dino --loss triplet --epochs 20 --lr 1e-4 --margin 0.1 --deepfake_weight 0.5 --seed 42 > logs\exp7_vit_triplet_deepfake.log 2>&1
echo vits16_dino triplet deepfake_weight=0.5 - %date% %time% >> experiment_runs.txt

echo Запуск эксперимента 8/10
python train.py --model vits16_dino --loss triplet --epochs 20 --lr 1e-4 --margin 0.1 --balance_real_fake --real_ratio 0.5 --seed 42 > logs\exp8_vit_triplet_balanced.log 2>&1
echo vits16_dino triplet balanced - %date% %time% >> experiment_runs.txt

:: Эксперименты с майнерами
echo Запуск эксперимента 9/10
python train.py --model vits16_dino --loss triplet --epochs 20 --lr 1e-4 --margin 0.1 --miner hardest --seed 42 > logs\exp9_vit_triplet_hardest.log 2>&1
echo vits16_dino triplet miner=hardest - %date% %time% >> experiment_runs.txt

:: Комбинированный эксперимент
echo Запуск эксперимента 10/10
python train.py --model vits16_dino --loss triplet --epochs 35 --lr 1e-4 --margin 0.1 --advanced_augmentation --deepfake_weight 0.5 --miner hardest --seed 42 > logs\exp10_vit_triplet_combined.log 2>&1
echo vits16_dino triplet combined - %date% %time% >> experiment_runs.txt

echo Все эксперименты завершены!
echo Время окончания: %date% %time%