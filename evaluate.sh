#python -m evaluate --dataset AIDS           --kfold 10    --lr 1e-5  --pooling add  --width 8   --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC0          --kfold 10    --lr 1e-5  --pooling mean --width 8   --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC1          --kfold 10    --lr 1e-5  --pooling mean --width 8   --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC2          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
python -m evaluate --dataset EMLC1          --kfold 5     --lr 2e-5 --pooling mean --width 8     --ccp_alpha 1e-3 --max_steps 1000
# python -m evaluate --dataset PROTEINS       --kfold 10    --lr 1e-5  --pooling mean --width 8   --ccp_alpha 1e-2
# python -m evaluate --dataset BZR            --kfold 10    --lr 1e-5  --pooling mean --width 8   --ccp_alpha 1e-2
# python -m evaluate --dataset BAMultiShapes  --kfold 10    --lr 1e-4  --pooling mean --width 8   --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC3          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC4          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC5          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC6          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC7          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC8          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC9          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3
#python -m evaluate --dataset EMLC10          --kfold 10    --lr 1e-4  --pooling mean --width 16  --sample_size 1000 --ccp_alpha 1e-3