![Flatmate Finder](img/flatmatefinder.png?raw=true "FlatmateFinder")

# Flatmate Finder

_Exploring the way users interact on an app-based roommate matching platform._

### Introduction
I partnered with an app-based roommate matching platform to investigate what factors influence whether a user can find a roommate through their app. Because the app doesn't follow up with sers to find which have successfully matched through the app, they define success as two users starting a conversation through the in-platform messaging service. 

I developed a model that more effectively identifies whether a particular user will respond to a message. With this model, the platform can be more selective in recommending possible matches.

### The Data Set
The data set included 25,000 users who sent over 30,000 messages.

The user data included 50 columns of data, including basic demographic information and roommate preferences. The message data was grouped into conversations, and included the time, users, and text of the message. 

### The Approach
I grouped the messages into conversations and considered a conversation sucessful if there was at least one response. I then gathered information about the two users from the second table and built an ensemble machine learning model with three models:
-Random Forest
-Gradient Boosting
-Adaboost

### Results
I was able to predict successful messages with an f1 score of around 40%. Given such a large class imbalance (90-10), a 40% score is an tremendous improvement and can help the company to identify which users are most likely to respond.

### What does this mean to the application?


Everytime the app suggests one user to another, it's making a prediction that those two users will be a good match. The app has to balance showing every user in the area, including users who are unlikely to respond, against showing very few users and missing out on possible matches. 

The ROC curve below shows this tradeoff. 
![ROC Curve](img/roc_curve_AB.png?raw=true "ROC Curve")

On the left is an application with a very low false positive rate - very unlikely to suggest users who wouldn't make a good match - but at the cost of a low true positive rate - not showing users who would be a good match.

On the right is the reverse - an application who shows lots of good matches, but at the cost of also showing a lot of duds.

Where exactly the app wants to be probably depenst on the location. Someplace like San Francisco, with a lot of users, can be more selective about who they show. We would want to place them somewhere near point "A" on the curve. The false positive rate is down at about 10% of users, meaning about 10% of the time the app recommends a user who won't be a good match, but the true positive rate is 50%, meaning the app doesn't show half the users that might be a good match. In someplace like San Francisco, that migh not be a big deal.

On the flip side might be a place like my home state of Connecticut which has many fewer users. To make sure we have enough suggestions, we should pump the true positive rate up to 90% - missing out on only 10% of users who would be a good match. The cost to this is a much higher false positive rate - around 50% - meaning users won't match with about half of the potential roommates we suggest. 



### Project Files

#### 002. reading files
  - 002a reading user data
  - 002b reading message data

#### 003. explore data
  - 003a explore user data
  - 003b explore message data
 
#### 004. get targets
  - 004a combine user and message data
  - 004b explore combined data

#### 005. models
  - 005a linear/logistic regression
  - 005b machine learning
  
