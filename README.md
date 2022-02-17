# pyrecmd

Pure python implementation of collaborative filtering algorithm, one of popular methods when to build recommendation system

### Installation

1. Download python package file: [pyrecmd-0.1.0.tar.gz](https://github.com/humble-data-miner/pyrecmd/blob/master/dist/pyrecmd-0.1.0.tar.gz)
1. pip install pyrecmd-0.1.0.tar.gz

### Example


```python
import pyrecmd
import os

# assume that you already have the movie-lens 100k dataset
with open(os.path.join("ml-100k","u.data"),"r") as fin:
    ratings = np.asarray([ [int(token) for token in line.split("\t")] for line in fin])
    ratings[:,[0,1]] = ratings[:,[1,0]] # switch user and item columns

with open(os.path.join("ml-100k","u.item"),"r",encoding="latin1") as fin:
    itemnames = [ line.split("|")[1] for line in fin]
    itemnames.insert(0, None)

# initialize with 100 features and ratings each record in form of (item, user, rating)
# parameter 'itemnames' and 'usernames' are optional and required to output more readable names
movierecmd = pyrecmd.PyRecmd(100, ratings, itemnames=itemnames)

# parameter 'regularization' is optional but it's worth tuning this parameter for better validation error
# parameter 'learningrate' is optional but it's worth tuning this parameter to shorten learning time
# parameter 'validationsplit' is optional and usually needed when you're tyring to tune learning parameters like 'regularization' and 'learningrate'
movierecmd.fit(regularization=0.8, iterations=10000, learningrate=3e-4, validationsplit=0.1, verbose=True)

  # learning-rate=0.0003, regularization-lambda=0.8, iterations=10000, validationsplit=0.1
  # 90700 for train, 9300 for test, total 100000 ratings
  # iteration #1: loss=0.04257519975299331, valloss=0.04123926882097799
  #iteration #1000: loss=0.003767156102278313, valloss=0.02942087829419617
  #iteration #2000: loss=0.0019754647266246525, valloss=0.029991390881000023
  #iteration #3000: loss=0.0017366852642300109, valloss=0.029905459133939213
  #iteration #4000: loss=0.001660970523927064, valloss=0.029934834364596683
  #iteration #5000: loss=0.0016250544223948558, valloss=0.03011373533301576
  #iteration #6000: loss=0.0016085053613471238, valloss=0.030277677577385687
  #iteration #7000: loss=0.001600799510030617, valloss=0.030360609212595906
  #iteration #8000: loss=0.0015958981491327761, valloss=0.03042506479935819
  #iteration #9000: loss=0.0015925062894997233, valloss=0.030413337290199106
  #iteration #10000: loss=0.0015910422433985266, valloss=0.030346999656762637

# predict all the ratings given users 1,2,3,4,5
userresult = movierecmd.predict(movierecmd.USER,[1,2,3,4,5])

  #{1: [('Toy Story (1995)', 4.85, 5.0),
  #    ('GoldenEye (1995)', 3.02, 3.0),
  #    ('Four Rooms (1995)', 3.92, 4.0),
  #    ('Get Shorty (1995)', 4.16, 3.0),
  #    ('Copycat (1995)', 2.83, 3.0),
  #         ...
  #2: [('Toy Story (1995)', 4.03, 4.0),
  #   ('GoldenEye (1995)', 2.78, None),
  #   ('Four Rooms (1995)', 2.32, None),
  #   ('Get Shorty (1995)', 2.95, None),
  #   ('Copycat (1995)', 3.08, None),
  # ...

sample_ratings = [(r[0],r[1]) for r in ratings if r[1] == 5]
# instant_fit learns a single user or item instantly and return prediction results given the sample
instant_fit_result = movierecmd.instant_fit(movierecmd.USER, sample_ratings, regularization=0.8, iterations=5000, learningrate=3e-4, verbose=True)

  #iteration #1: loss=0.16384997286426806
  #iteration #500: loss=0.04543407281702286
  #iteration #1000: loss=0.0346454254970892
  #iteration #1500: loss=0.02911853749379627
  #iteration #2000: loss=0.026296828685820364
  #iteration #2500: loss=0.02471713670331908
  #iteration #3000: loss=0.023771358973847783
  #iteration #3500: loss=0.023201403212681094
  #iteration #4000: loss=0.02287483390299474
  #iteration #4500: loss=0.02270596322264621
  #iteration #5000: loss=0.02263077845384043

print(instant_fit_result)

  #{'prediction': [('Toy Story (1995)', 5, 5.0),
  #                ('GoldenEye (1995)', 4.99, 5.0),
  #                ('Four Rooms (1995)', 5, None),
  #                ('Get Shorty (1995)', 5, None),
  #                ('Copycat (1995)', 5, None),
  # ...
  # 'weight': array([-0.0012171 ,  0.49456538,  0.4138671 ,  0.02527371,  0.22232359,
  # ...

```

### Note

The input "Ratings" is a list of tuples, each has (item, user, rating)
##### Recommended steps to follow
1. **(PyRecmd() construct)** Initialize PyRecmd object with rating information 
2. **(PyRecmd.fit())** Try several times varying with 'regularization' and 'learningrate' parameters also given 'validationsplit' parameter, 0.1 for an example 
    * 'validationsplit > 0' splits input ratings into train and test set
    * 'validationsplit > 0' outputs validation loss as well, which is important to check the model is overfitted
    * 'regularization' is weight regularization and validation loss is quite depedent to this parameter
    * 'learningrate' plays a role in convergence speed as usual machine learning method does
3. **(PyRecmd.fit())** Given the best 'learningrate' and 'regularization' parameters found, learn again the whole dataset without 'validationsplit' in order to materialize the final weights
4. **(PyRecmd.predict())** Predict all the ratings given users or items
5. **(PyRecmd.instant_fit)** Fit any external user or item not learned yet to the existing model and returns prediction results

### Limitation

* Too slow to learn a large dataset in a feasible time which size is more than 100K

### Future items(if they are in need)

* Speed up by OpenCL
* Incremental model building

### Reference

* [Wikipedia - Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
* [Coursera - Machine Learning - Recommender System](https://www.coursera.org/learn/machine-learning)