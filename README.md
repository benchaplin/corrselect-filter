# Correlation + Selectivity Filter Builder

**The problem**: given an array of "scores" (one might also think of distance, where high score = short distance), generate a filter (a 0/1 bitmask) for a specified:

- selectivity _s_: fraction of 1s
- correlation _c_: [point-biserial](https://en.wikipedia.org/wiki/Point-biserial_correlation_coefficient)

## Random flips (iterative approach)

1. Sort vectors from low -> high score
2. Label a chunk of the data:
   - If _c < 0_, label the lowest _s_ fraction of the data
   - If _c >= 0_, label the highest _s_ fraction of the data
3. Measure the correlation.
4. For a fixed number of iterations, flip some filter labels to push the correlation up/down, breaking if it's within a threshold of the target correlation.
   - If the current _c_ is too low, pick one of the worst correlated 1s and flip it to 0, then flip a random 0 to 1 to preserve _s_.
   - If the current _c_ is too high, pick one of the best correlated 1s and flip it to 0, then flip a random 0 to 1 to preserve _s_.
