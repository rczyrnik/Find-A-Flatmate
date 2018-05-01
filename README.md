![Find-A-Flatmate](img/flatmatefinder.png?raw=true "Find-A-Flatmate")

# Find-A-Flatmate

_Exploring the way users interact on an app-based roommate matching platform._

## Introduction

I partnered with an app-based roommate finding platform to investigate the reasons behind a low message response rate. I built a model that can predict when users will respond, and discovered that most of the time it's not a bad match that's at fault, but simply a recipient that was never likely to engage in the first place. With my model, the app can better suggest potential roommates by avoiding matches that were never likely to respond. 

## The Data Set
The data set included 25,000 users who together sent over 30,000 messages.

The user data included 50 columns of data, including basic demographic information and roommate preferences. I grouped message data into conversations with information about time, users, and text of the message. 

## The Approach
Because the app doesn't follow up with sers to find which have successfully matched through the app, they define success as two users starting a conversation through the in-platform messaging service. 

I grouped the messages into conversations and considered a conversation sucessful if there was at least one response. I then gathered information about the two users from the second table and built an ensemble machine learning model with four models:

- Linear Regression

- Random Forest

- Gradient Boosting

- Adaboost

## Results
I was able to predict successful messages with an f1 score of around 40%. Given such a large class imbalance (90-10), a 40% score is an tremendous improvement and can help the company to identify which users are most likely to respond.

## What does this mean to the application?


Everytime the app suggests one user to another, it's making a prediction that those two users will be a good match. The app has to balance showing every user in the area, including users who are unlikely to respond, against showing very few users and missing out on possible matches. Where they put that cutoff depends in large part on the number of users in an area.

The ROC curve below shows this tradeoff. 
![ROC Curve](img/roc_curve_AB_temp.png?raw=true "ROC Curve")

#### Point A: San Francisco, California

Tons of users, we can afford to miss out on a few opportuntiites. False positive rate is down at 10% so we are unlikely to show potential users who wouldn't be a good match. To achieve such a low fals positive rate, we have to accept a true positive rate at 50%, which isn't great; it means we're not picking up on about half of potential matches. But as long as there are lots of fish in the sea, it's not imperative that we catch them all.

#### Point B: Hartford, Connecticut

I love my home state of Connecitcut, but there just isn't the density of young people looking for shared housing. This means we can't be as picky in matching users. Yes, our false positive rate jumps to 50%, meaning about half the people we suggest won't be a match, but we need that high false positive rate to maintain a 90% true positive rate, missing out on very few potential matches. 

## Discussion
A 10% response rate can be very disheartening for users of this app, and improving the response rate should be a priority.

My model showed that it's possible to predict, with good recall and precision, who will respond to a user's message. It should be possible, therefore, to run the calculation before suggesting possible matches, to rule out candidates that may look good on paper but who have a very low probability of matching.

This possibility seems even more likely when we look at the features that most influenced predictions. Of the ten most influential features, eight were true of the receiver alone, suggesting that it's not the "fit" between two users that determines whether someone will respond or not. Instead, some users are just unlikely to respond no matter what.

For example, users who didn't fill out the "about me section" of their profile are more than half as likely to respond compared to users who did fill out that section. By suggesting only users who have filled out that section, we may be able to improve the response rate from the current 10% to 16%.

User experience is key to the success of this app, and if users feel like they are sending messages out into the void without receiving responses, they will switch to another platform. By rigging the system towards users who are more likely to respond, we provide a better experience for everyone.
