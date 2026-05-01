1. Implement "Minimum Samples per Leaf"Currently, you have minSamplesToSplit. However, Scikit-learn's default strength often comes from min_samples_leaf.The Problem: minSamplesToSplit might be $2$, but it could result in one branch having $49$ samples and the other having only $1$. That $1$-sample leaf is almost certainly noise.The Fix: Ensure that every leaf created contains at least $5$ samples (for Iris). If a split would result in a leaf with fewer than that, discard the split entirely. This forces the tree to only care about "statistically significant" patterns.

2. Post-Pruning

3. Weighted Average Thresholds