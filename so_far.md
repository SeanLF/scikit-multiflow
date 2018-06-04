# Thesis work

## Libraries used
Below is a list of all of the software libraries that I have used:
- [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) 
- [scikit-learn](http://scikit-learn.org/stable/)
- [mlxtend](https://github.com/rasbt/mlxtend)'s voting classifier

## Tying them together
Scikit-learn is used to import additional classifiers to run in the scikit-multiflow framework.
Mlxtend has a voting classifier that I used.
I then modified the voting classifier to use a different voting scheme (by summing class probabilities) and so that it can function online, with batches or sliding windows and use a drift detector for each classifier in the ensemble.

## Ideas put forward
- Using a voting ensemble that requires that its classifiers to submit class probabilities, in order to sum them and select the class with the highest average.
- Using a sliding window from the point of view of the stream, and a tumbling window from the point of view of the ensemble classifiers. (turns out to be same method proposed by Sarah D'Ettore in FG-CDCStream but she uses it only for drift detection)
- Allows for either online learning (with a window size of 1), or batch learning to speed things up using overlapping chunks.
- Modified FHDDM to detect drifts by using a sliding window of weighted prediction confidence. This reduces the reliance upon immediate knowledge of the ground truth after predictions.
- When drifts are detected, all models are dropped, and rebuilt using the last 100 tuples.