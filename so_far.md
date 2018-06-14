# Thesis work

## Libraries used
Below is a list of all of the software libraries that I have used:
- [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) 
- [scikit-learn](http://scikit-learn.org/stable/)
- [mlxtend](https://github.com/rasbt/mlxtend)'s voting classifier

## Tying them together
Scikit-learn is used to import additional classifiers to run in the scikit-multiflow framework.
Mlxtend has a voting classifier that I used.
I then modified the voting classifier to use a different voting scheme (by summing class probabilities). It was also modified so that it can function online, with batches or sliding windows. Finally, it was further modified to use a drift detector on each classifier in the ensemble.

## Ideas put forward
- [x] Voting ensemble that requires that its classifiers submit class probabilities, in order to sum them and select the class with the highest average.
- [x] Using a sliding window from the point of view of the stream, and a tumbling window from the point of view of the ensemble classifiers. (turns out to be same method proposed by Sarah D'Ettore in FG-CDCStream but she uses it only for drift detection)
    - [x] Allows for either online learning (with a window size of 1), or in batches to speed things up.
- [x] Modified FHDDM to detect drifts by using a sliding window of weighted prediction confidence. This reduces the reliance upon immediate knowledge of the ground truth after predictions.
- [x] When drifts are detected, all models are dropped, and rebuilt using the last 100 tuples.
- [x] When drifts are detected, a subset of models are dropped, and rebuilt using the last 100 tuples. (current implementation: each classifier in the voting ensemble has a 70% chance of being reset)
- [ ] When drifts are detected, query the oracle, and create a temporary replacement classifier that learns in parallel in case there is a drift according to the oracle. Could also skip tuples?

## Experiments to perform
- Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
- Compare different voting ensemble techniques against one another and against single classifiers and against other ensemble methods
- See how the modified concept drift detector performs with/without sliding tumbling windows, and/or when playing with the ensemble classifier reset logic, and finding the right balance of ground truth that can be omitted versus using predicted values as the ground truth
- Evaluate the performance, stream velocity, accuracy against other methods
- Determine if the sliding tumbling window can outperform in speed and accuracy versus simply using sliding windows
- Build summarizing classifiers using batches and give them a slightly higher weight, or determine if there is a threshold

## Questions
- Is using someone elses structure/organization for the litterature review considered plagiarism?