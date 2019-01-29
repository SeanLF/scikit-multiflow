# Thesis work

## Libraries used
Below is a list of all of the software libraries that I have used:
- [scikit-multiflow](https://github.com/scikit-multiflow/scikit-multiflow) 
- [scikit-learn](http://scikit-learn.org/stable/)
- [mlxtend](https://github.com/rasbt/mlxtend)'s voting classifier

## Tying them together
Scikit-learn is used to import additional classifiers to run in the scikit-multiflow framework.
Mlxtend has a voting classifier that I used.
I then modified the voting classifier to use a different voting scheme (by averaging weighted class probabilities). It was also modified so that it can function online, with batches or sliding windows. Finally, it was further modified to use a drift detector on each classifier in the ensemble or on a average of their results.

## Ideas put forward

- Voting ensemble that requires that its classifiers submit class probabilities, weight them, then average them and select the class with the highest average.
- Using a sliding window from the point of view of the stream, and a tumbling window from the point of view of the ensemble classifiers. (turns out to be same method proposed by Sarah D'Ettore in FG-CDCStream but she uses it only for drift detection)
    - This method allows for either true online learning (with a window size of 1), or in chunks to speed things up.
- Modified FHDDM to detect drifts by using a sliding window of weighted prediction confidence. This reduces the reliance upon immediate knowledge of the ground truth after predictions.
- Modified FHDDM to detect drifts by using a sliding window of agreement among classifiers. Also allows a reduction on the immediate knowledge of ground truth.
- When drifts are detected, all models are dropped, and rebuilt using the last NUM tuples.
- When drifts are detected, a subset of models are dropped, and rebuilt using the last NUM tuples. (current implementation: each classifier in the voting ensemble has a 70% chance of being reset)


## Experiments to perform
- Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
- The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
- Compare different voting ensemble strategies against one another and against single classifiers and against other ensemble methods. Compare outcomes
- See how the modified concept drift detector performs with/without sliding tumbling windows, and/or when playing with the ensemble classifier reset logic, and finding the right balance of ground truth that can be omitted versus using predicted values as the ground truth
- Evaluate the performance, stream velocity, accuracy against other methods
- It is good to compare to blind adaptation, i.e. a simple model reset at every x instances.
- Compare to state of the art.

## Questions
- Is using someone elses structure/organization for the litterature review considered plagiarism?

## Reminders for writing
- Sarah also uses some voting to decide on whether there is a warning/drift, but it is not clearly stated in her work. It would be good to formalize this in your thesis (Equations).

## Future work
- Replace poorly performing classifiers (find good metric)
- Modify window size in real time
- Test on real dataset
- Optimize weighting function
- Test more variations of the parameters
- Introduce active learning where an oracle can classify tuples where all classifiers are insufficiently confident in their predictions
- Build summarizing classifiers using batches and give them a slightly higher weight in the vote
- Experiments on mixed drift
- Confirm that drift detector's window size is optimal
- Drift detector's window size should be linked to learning chunk size
- Test before/after weighting with different functions

## Notes

- Could not extend MDDM supposedly because it requires knowledge of the class to insert it into its sliding window.
- Runtime includes writing to file, calculating metrics and generating the stream. 