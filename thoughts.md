do teams' historical performance data take into account the calibre of the team they were playing against? e.g. cabo verde lost to argentina but argentina are a great team

- k-NN is not very useful at predicting the outcomes of future matches because it only takes into account ELO diff which the CTMCs already take into account. So the signals are linearly dependent. This is called feature collinearity. However it would be very useful for predicting live, in-play matches. i could test the effectiveness of knn at predicting futuer matches by doing feature wrapper techniques and doing leave one out cross validation, getting the model to predict match outcomes.

- when calculating a statistic to measure variance between my model's predictions and the true bookie odds, I can't do direct comparison on decimal odds. Since decimal odds are non-linear, the difference between 1.10 and 1.20 (difference: 9.1%) is mathematically treated the same as 5.10 and 5.20 (difference: 2.0%), even thought the percentage differences are wildly different.

- current rendition assumes vig is distributed uniformly when in reality it is distributed more towards the underdogs and draws since favourites already have odds so low that people won't be tempted to bet on them

- i was getting very different results when I ran Home: Norway vs Away: England compared to Home: England vs Away: Norway. This implies that the global Q matrix has home advantage baked into it somehow. This initially seems weird because 90% of the matches that went into the global Q matrix were played by 2 neutral teams (not Canada/USA/Mexico). But the reason for it is that due to administrative seeding, Pot 1 (the best pot) teams ended up in the 'Home' slot more often (2/3 times for their group games) which therefore inflated the transition rates for Home teams.

- we will turn the M>=10,000 historical match database into a vectorised form offline.
