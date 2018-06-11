# To do

*MDDM: Mcdiarmid Drift Detection Method*

- [x] implement a drift detector that doesn't immediately need the class
    - MDDM does not work because it requires knowledge of class to insert it into the sliding window (similar to FHDDM(S))
- [x] allow online and chunk based learning (sklearn allows 1+ tuples to be fetched)
    - determine where changes need to be made
- [x] import tornado framework datasets

## Experiments
- modify MDDM to use a sliding window of weighted prediction confidence (Bayesian model?)
-  *NOPE* online and chunk-based learning in parallel (using ≠ sized chunks)
- overlapping chunks
    - evaluation
        - each classifier does prequential on its own chunk
        - drift detected if slided window chunks don't predict same class or confidence lowers (depends on window/chunk size)
    - examples
        - chunk 1 [1,2,3] 1' [2,3,4] 1'' [3,4,5] for c1, c2, c3
        - chunk 2 [4,5,6] 2' [5,6,7] 2'' [6,7,8] for c1, c2, c3
        - chunk 3 [7,8,9] 3' [8,9,10] 3'' [9,10,11]
- max velocity of incoming tuples / run time


run a sliding (pov: stream) tumbling (pov: classifier in the ensemble) window

modify window size in real time 2015?
    perfsim

kaggle for social good?


drop only one classifier (or more)!

DS 2018 possible conference deadline June 26th

framework -> diagram and pseudocode
    [take another look at Sarah D'Ettorre] [Mohamed Kenlinde Olorunimbe] thesis