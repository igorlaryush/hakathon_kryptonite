# experiments.ps1
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
"Запуск экспериментов: $Timestamp" | Out-File -FilePath "experiment_runs.txt"

# Создаем папку для логов
New-Item -Path "logs" -ItemType Directory -Force | Out-Null

function Run-Experiment {
    param (
        [string]$Description,
        [string]$CommandArgs
    )
    
    $ExperimentNumber = $script:ExperimentCounter
    $script:ExperimentCounter++
    
    Write-Host "Запуск эксперимента $ExperimentNumber/$TotalExperiments: $Description"
    $LogFile = "logs\exp${ExperimentNumber}_$($Description -replace ' ', '_' -replace '=', '-').log"
    
    $Command = "python train.py $CommandArgs"
    Write-Host "Выполнение: $Command"
    
    Invoke-Expression "$Command > $LogFile 2>&1"
    
    "$Description - $(Get-Date)" | Out-File -FilePath "experiment_runs.txt" -Append
    Write-Host "Эксперимент $ExperimentNumber завершен"
    Write-Host "------------------------------------------"
}

# Счетчик экспериментов
$script:ExperimentCounter = 1
$TotalExperiments = 10

# Базовые эксперименты
Run-Experiment -Description "vits16_dino triplet" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --seed 42"
Run-Experiment -Description "vits16_dino arcface" -CommandArgs "--model vits16_dino --loss arcface --epochs 1 --lr 1e-4 --margin 0.2 --scale 30 --embedding_size 512 --seed 42"
Run-Experiment -Description "swin triplet" -CommandArgs "--model swin --loss triplet --epochs 1 --lr 5e-5 --margin 0.1 --seed 42"
Run-Experiment -Description "convnext cosface" -CommandArgs "--model convnext --loss cosface --epochs 1 --lr 1e-4 --margin 0.2 --scale 30 --embedding_size 512 --seed 42"

# Эксперименты с аугментациями
Run-Experiment -Description "vits16_dino triplet adv_aug" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --advanced_augmentation --seed 42"
Run-Experiment -Description "vits16_dino triplet mixup" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --mixup --mix_alpha 0.2 --seed 42"

# Эксперименты с дипфейком
Run-Experiment -Description "vits16_dino triplet deepfake=0.5" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --deepfake_weight 0.5 --seed 42"
Run-Experiment -Description "vits16_dino triplet balanced" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --balance_real_fake --real_ratio 0.5 --seed 42"

# Эксперименты с майнерами
Run-Experiment -Description "vits16_dino triplet miner=hardest" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --miner hardest --seed 42"

# Комбинированный эксперимент
Run-Experiment -Description "vits16_dino triplet combined" -CommandArgs "--model vits16_dino --loss triplet --epochs 1 --lr 1e-4 --margin 0.1 --advanced_augmentation --deepfake_weight 0.5 --miner hardest --seed 42"

Write-Host "Все эксперименты завершены!"