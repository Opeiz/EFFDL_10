echo "Quantizing"
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_30_F1.pth --prune_ratio 0.3 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_30_F2.pth --prune_ratio 0.3 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_30.pth --prune_ratio 0.3 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_40_F1.pth --prune_ratio 0.4 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_40_F2.pth --prune_ratio 0.4 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_40.pth --prune_ratio 0.4 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_50_F1.pth --prune_ratio 0.5 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_50_F2.pth --prune_ratio 0.5 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_60_F1.pth --prune_ratio 0.6 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_50.pth --prune_ratio 0.5 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_60_F2.pth --prune_ratio 0.6 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_60.pth --prune_ratio 0.6 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_70_F1.pth --prune_ratio 0.7 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_70_F2.pth --prune_ratio 0.7 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_70.pth --prune_ratio 0.7 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_80_F1.pth --prune_ratio 0.8 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_80_F2.pth --prune_ratio 0.8 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Custom/PreAct_Base150_B64_80.pth --prune_ratio 0.8 --fact_ratio 0#
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_30_F1.pth --prune_ratio 0.3 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_30_F2.pth --prune_ratio 0.3 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_30.pth --prune_ratio 0.3 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_40_F1.pth --prune_ratio 0.4 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_40_F2.pth --prune_ratio 0.4 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_40.pth --prune_ratio 0.4 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_50_F1.pth --prune_ratio 0.5 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_50_F2.pth --prune_ratio 0.5 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_60_F1.pth --prune_ratio 0.6 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_50.pth --prune_ratio 0.5 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_60_F2.pth --prune_ratio 0.6 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_60.pth --prune_ratio 0.6 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_70_F1.pth --prune_ratio 0.7 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_70_F2.pth --prune_ratio 0.7 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_70.pth --prune_ratio 0.7 --fact_ratio 0
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_80_F1.pth --prune_ratio 0.8 --fact_ratio 1
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_80_F2.pth --prune_ratio 0.8 --fact_ratio 2
#python calculate_Score.py --ckpt PruneRatio/Unstructured/PreAct_Base150_B64_80.pth --prune_ratio 0.8 --fact_ratio 0
#cd EFFDL_10_main
#python LongProject.py --ckpt PreAct_Base150_B64_70.pth --half
#echo "Calculating the scores"
python calculate_Score.py --ckpt PreAct_Base150_B64_DA.pth --quant 16 --prune_ratio 0 --fact_ratio 0