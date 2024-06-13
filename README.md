# GDE
Codes for the paper: How Powerful is Graph Filtering for Recommendation (KDD '24)

# Hyperparameter Setting

For SGFCF without the IGF design (sgfcf_wo_igf.py)

- Gowalla

```bash
python sgfcf_wo_igf.py --dataset='gowalla' --density='sparse' --k=650 --beta=1.3 --eps=0.34 --gamma=1.6
python sgfcf_wo_igf.py --dataset='gowalla' --density='dense' --k=110 --beta=3.1 --alpha=4.8 --gamma=0.02
```
