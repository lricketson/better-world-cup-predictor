i want to make calls to the oddsportal historical odds website to create a statistic to judge how similar the odds my model gives are to the real ones. i also want a statistic on how much variance there is in the odds for this match between different betting firms.

- i should get a bunch of past Euros and copa america games so i can increase the size of the dataset and test the model's brier score

- do walk forward validation (use a dynamically calculated global q matrix that only factors in matches that happened before the one we're about to make a prediction for)

- add knn to the live in play engine

- still need to improve GPU performance with Monte Carlo simulations. E.g. testing hyperparams took 6.2 sec/iteration but it should take 0.18 sec/iteration. maybe ask antigravity to look at it

- figure out how to get enough matches such that i have a good database of historical matches.

- think about how i could make this modular enough to be able to directly apply it to the premier league rather than the wc

- set up backtesting to find out how much benefit (if any) the k-nn gives the overall model.
- test both the CTMC and the k-NN individually and then together, to see the benefit

- add in yellow/red cards
