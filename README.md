# Correlation + Selectivity Filter Builder

**The problem**: given an array of "scores" (one might also think of distance, where high score = short distance), generate a filter (a 0/1 bitmask) for a specified:

- selectivity _s_: fraction of 1s
- correlation _c_: [point-biserial](https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient)

## Random flips approach

1. Sort vectors from low -> high score
2. Label a chunk of the data:
   - If _c < 0_, label the lowest _s_ fraction of the data
   - If _c >= 0_, label the highest _s_ fraction of the data
3. Measure the correlation.
4. Randomly flip some filter labels.
