python -m evaluate_quick --dataset EMLC0      --conv GCN        
python -m evaluate_quick --dataset EMLC0      --conv DIR-GCN      
python -m evaluate_quick --dataset EMLC1      --conv GCN          
python -m evaluate_quick --dataset EMLC1      --conv DIR-GCN     
python -m evaluate_quick --dataset EMLC2   --conv GCN          
python -m evaluate_quick --dataset EMLC2   --conv DIR-GCN     
python -m evaluate_quick --dataset AIDS  --conv GCN            
python -m evaluate_quick --dataset AIDS  --conv DIR-GCN        
python -m evaluate_quick --dataset PROTEINS  --conv GCN 
python -m evaluate_quick --dataset PROTEINS  --conv DIR-GCN  
#
# python -m evaluate_quick --dataset EMLC0      --conv GCN        --dim 512   --layers 4   --lr 8e-4   
# python -m evaluate_quick --dataset EMLC0      --conv DIR-GCN    --dim 512   --layers 4   --lr 5e-5   
# python -m evaluate_quick --dataset EMLC1      --conv GCN        --dim 512   --layers 10 --lr 7e-3  
# python -m evaluate_quick --dataset EMLC1      --conv DIR-GCN    --dim 512   --layers 16 --lr 7e-5  
# python -m evaluate_quick --dataset EMLC2   --conv GCN          --dim 256    --layers 4  --lr 3e-4  
# python -m evaluate_quick --dataset EMLC2   --conv DIR-GCN      --dim 256   --layers 4  --lr 6e-4  
# python -m evaluate_quick --dataset AIDS  --conv GCN         --dim 256   --layers 12     --lr 4e-5    
# python -m evaluate_quick --dataset AIDS  --conv DIR-GCN     --dim 512   --layers 10     --lr 2e-5    
# python -m evaluate_quick --dataset PROTEINS  --conv GCN     --dim 128      --layers 12  --lr 5e-4  
# python -m evaluate_quick --dataset PROTEINS  --conv DIR-GCN --dim 256      --layers 4  --lr 1e-3  
