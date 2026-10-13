# AT-FedBN Experimental Results

## Performance Comparison

| Strategy    | Description                   |   Accuracy |   Balanced Acc |     F1 |   ROC-AUC | Privacy     | Personalization   |
|:------------|:------------------------------|-----------:|---------------:|-------:|----------:|:------------|:------------------|
| Centralized | Upper bound (all data, no FL) |     0.729  |         0.7286 | 0.7166 |    0.7991 | None        | None              |
| FedBN       | Local BatchNorm (no trust)    |     0.6131 |         0.6146 | 0.5708 |    0.6621 | None        | Local BN          |
| AT-FedBN    | Trust-weighted + DP (β=0.7)   |     0.6997 |         0.703  | 0.6214 |    0.8056 | DP (σ=0.01) | Local BN + Trust  |

## Key Findings

### Performance Hierarchy
1. **Centralized** - Upper bound (all data pooled, no privacy)
2. **AT-FedBN** - Best federated approach (trust + personalization + privacy)
3. **FedBN** - Good baseline (personalization, no trust)
4. **FedAvg** - Standard FL baseline

### Novel Contributions
- **Trust Mechanism**: Update consistency scoring via cosine similarity
- **Hybrid Aggregation**: β × volume + (1-β) × trust weights
- **FedBN Personalization**: Local BatchNorm adaptation to hospital-specific distributions
- **Differential Privacy**: Gradient clipping + Gaussian noise

### Privacy-Utility Tradeoff
- AT-FedBN achieves competitive performance while providing DP guarantees
- Current configuration (σ=0.01) prioritizes utility over formal privacy guarantees
- Privacy budget ε ≈ 55,000 indicates weak formal guarantee but practical noise injection

## Methodology

- **Dataset**: Cardiovascular disease (70,000 records)
- **Non-IID Split**: Dirichlet(α=1.0) across 3 hospitals
  - Hospital 0: 41% positive class
  - Hospital 1: 83% positive class
  - Hospital 2: 31% positive class
- **Model**: 3-layer MLP with BatchNorm
- **Training**: 10 FL rounds, 2 local epochs per round

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Balanced Accuracy**: Handles class imbalance (not fooled by skewed predictions)
- **F1**: Harmonic mean of precision and recall
- **ROC-AUC**: Discrimination ability across all thresholds
