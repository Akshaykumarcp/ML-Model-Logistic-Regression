# ML-Model-Logistic-Regression
Machine Learning Logistic Regression

# Logistic Regression Mind Map
![Logistic Regression Mind Map](https://github.com/Akshaykumarcp/ML-Model-Logistic-Regression/blob/main/Logistic%20Regression.jpg)

# Logistic Regression
- Logistic regression is one such regression algorithm which can be used for performing classification problems. 
- It calculates the probability that a given value belongs to a specific class. If the probability is more than 50%, it assigns the value in that particular class else if the probability is less than 50%, the value is assigned to the other class. 
- Therefore, we can say that logistic regression acts as a binary classifier.

## Working of a Logistic Model
- Linear regression, the model is defined by: 

    ```
    y=β0+β1x - (equation 1)
    ```

    and for logistic regression, we calculate probability, i.e. y is the probability of a given variable x belonging to a certain class. Thus, it is obvious that the value of y should lie between 0 and 1.

    But, when we use equation 1 to calculate probability, we would get values less than 0 as well as greater than 1. That doesn’t make any sense . So, we need to use such an equation which always gives values between 0 and 1, as we desire while calculating the probability.

## Sigmoid function
We use the sigmoid function as the underlying function in Logistic regression. Mathematically and graphically, it is shown as: https://images.app.goo.gl/jMEBdvkbnFazRxz98

#### Why do we use the Sigmoid Function?

1) The sigmoid function’s range is bounded between 0 and 1. Thus it’s useful in calculating the probability for the Logistic function. 
2) It’s derivative is easy to calculate than other functions which is useful during gradient descent calculation. 
3) It is a simple way of introducing non-linearity to the model.

Although there are other functions as well, which can be used, but sigmoid is the most common function used for logistic regression. 

# Multinomial Logistics Regression( Number of Labels >2)
- Many times, there are classification problems where the number of classes is greater than 2. 
- We can extend Logistic regression for multi-class classification. 
- The logic is simple; we train our logistic model for each class and calculate the probability(hθx) that a specific feature belongs to that class. 
- Once we have trained the model for all the classes, we predict a new value’s class by choosing that class for which the probability(hθx) is maximum. 
- Although we have libraries that we can use to perform multinomial logistic regression, we rarely use logistic regression for classification problems where the number of classes is more than 2. 
-   There are many other classification models for such scenarios. 

# Evaluation of a Classification Model

## Refer other gihub repo

# Advantages of Logisitic Regression
- It is very simple and easy to implement.
- The output is more informative than other classification algorithms
- It expresses the relationship between independent and dependent variables
- Very effective with linearly seperable data

# Disadvantages of Logisitic Regression
- Not effective with data which are not linearly seperable
- Not as powerful as other classification models
- Multiclass classifications are much easier to do with other algorithms than logisitic regression
- It can only predict categorical outcomes