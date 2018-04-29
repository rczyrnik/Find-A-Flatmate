![Flatmate Finder](img/flatmatefinder.png?raw=true "FlatmateFinder")

# Flatmate Finder

_Exploring the way users interact on an app-based roommate matching platform._

### Introduction
I partnered with an app-based roommate matching platform to investigate what factors influence whether a user can find a roommate through their app. Because the app doesn't follow up with sers to find which have successfully matched through the app, they define success as two users starting a conversation through the in-platform messaging service. 

I developed a model that more effectively identifies whether a particular user will respond to a message. With this model, the platform can be more selective in recommending possible matches.

### The Data Set
The data set included 25,000 users who together sent over 30,000 messages.

The user data included 50 columns of data, including basic demographic information and roommate preferences. I grouped message data into conversations with information about time, users, and text of the message. 

### The Approach
I grouped the messages into conversations and considered a conversation sucessful if there was at least one response. I then gathered information about the two users from the second table and built an ensemble machine learning model with four models:

- Linear Regression

- Random Forest

- Gradient Boosting

- Adaboost

### Results
I was able to predict successful messages with an f1 score of around 40%. Given such a large class imbalance (90-10), a 40% score is an tremendous improvement and can help the company to identify which users are most likely to respond.

## What does this mean to the application?


Everytime the app suggests one user to another, it's making a prediction that those two users will be a good match. The app has to balance showing every user in the area, including users who are unlikely to respond, against showing very few users and missing out on possible matches. Where they put that cutoff depends in large part on the number of users in an area.

The ROC curve below shows this tradeoff. 
![ROC Curve](img/roc_curve_AB_temp.png?raw=true "ROC Curve")

### Point A: San Francisco, California

Tons of users, we can afford to miss out on a few opportuntiites. False positive rate is down at 10% so we are unlikely to show potential users who wouldn't be a good match. To achieve such a low fals positive rate, we have to accept a true positive rate at 50%, which isn't great; it means we're not picking up on about half of potential matches. But as long as there are lots of fish in the sea, it's not imperative that we catch them all.

### Point B: Hartford, Connecticut

I love my home state of Connecitcut, but there just isn't the density of young people looking for shared housing. This means we can't be as picky in matching users. Yes, our false positive rate jumps to 50%, meaning about half the people we suggest won't be a match, but we need that high false positive rate to maintain a 90% true positive rate, missing out on very few potential matches. 

