# SGFCF
Codes for the paper: How Powerful is Graph Filtering for Recommendation (KDD '24)

# Hyperparameter Setting

For SGFCF without the IGF design (sgfcf_wo_igf.py)

- CiteULike

```bash
python sgfcf_wo_igf.py --dataset='citeulike' --density='sparse' --k=110 --beta=4.6 --alpha=2.3--gamma=0.05
python sgfcf_wo_igf.py --dataset='citeulike' --density='dense' --k=1000 --beta=0.95 --eps=0.28 --gamma=1.5
```

- Pinterest

```bash
python sgfcf_wo_igf.py --dataset='pinterest' --density='sparse' --k=60 --beta=2.0 --eps=0.37--gamma=0.07
python sgfcf_wo_igf.py --dataset='pinterest' --density='dense' --k=300 --beta=1.0 --alpha=10 --gamma=0.3
```



- Yelp

```bash
python sgfcf_wo_igf.py --dataset='yelp' --density='sparse' --k=50 --beta=2.0 --alpha=5.3--gamma=0.025
python sgfcf_wo_igf.py --dataset='yelp' --density='dense' --k=250 --beta=1.0 --alpha=10 --gamma=0.5
```


- Gowalla

```bash
python sgfcf_wo_igf.py --dataset='gowalla' --density='sparse' --k=650 --beta=1.3 --eps=0.34 --gamma=1.6
python sgfcf_wo_igf.py --dataset='gowalla' --density='dense' --k=110 --beta=3.1 --alpha=4.8 --gamma=0.02
```

For SGFCF.py, only need to replace the beta with beta_1 and beta_2, other parameters are the same:

- CiteULike

```bash
beta_1, beta_2=0.7, 1.1 for dense
beta_1, beta_2=4.3, 5.1 for sparse
```

- Pinterest

```bash
beta_1, beta_2=0.9, 1.0 for dense
beta_1, beta_2=2.0, 2.4 for sparse
```

- Yelp

```bash
beta_1, beta_2=1.0, 1.3 for dense
beta_1, beta_2=2.0, 5.0 for sparse
```

- Gowalla

```bash
beta_1, beta_2=0.3, 2.5 for dense
beta_1, beta_2=2.5, 6.5 for sparse
```

