## Table of content
1. Logistic Regression Mind Map
2. About Logistic Regression
3. Logistic regression using geometric intuition
4. Weight vector
5. L2 regularization; overfitting and underfitting
6. L1 regularization & sparsity
7. Logistic regression using Loss minimization
8. Hyperparameter
9. Column/feature standardization
10. Feature importance and Model interpretation
11. Collinearity or multicollinearlity
12. Real world cases
13. What if data is not linearly seperable ?
14. Multinomial Logistics Regression
15. Advantages of Logistics Regression
16. Disadvantages of Logistics Regression
17. Generalized linear models (GLM)
18. Acknowledgements
19. Connect with me

---

## 1. Logistic Regression Mind Map
![Logistic Regression Mind Map](https://github.com/Akshaykumarcp/ML-Model-Logistic-Regression/blob/main/Logistic%20Regression.jpg)

## 2. About Logistic Regression
- Classification technique
- Simple algorithm
- Logistic regression calculates the probability that a given value belongs to a specific class. If the probability is more than 50%, it assigns the value in that particular class else if the probability is less than 50%, the value is assigned to the other class.
- Logistic regression has multiple perceptions:
    - Geometric intuition (easy and simple)
    - Probability 
    - Loss function

## 3. Logistic regression using geometric intuition

![Logistic Regression Mind Map](https://editor.analyticsvidhya.com/uploads/650901st.png)

Assumption made by LR: Data is linearly or almost linearly seperable.

![Logistic Regression Mind Map](https://miro.medium.com/max/1400/1*mmumN1XbpLPsWVLVmxx1LQ.png)

Task of LR is to find w and b which corresponds to the plain (pi) such that plain seperates positive points and negative points.

Diving into math piece.

Math optimization problem

![Logistic Regression Mind Map](https://miro.medium.com/max/618/1*OumRMqIYqx7Bws_nwIUjDA.png)

Above equation is read as, Find the optimal w such that it maximizes argmax value.

In above equation we're using signed distance to find optimal w, signed distance is prone to outlier. Due to outlier we'll end up finding wrong w.

So have to modify math formulation, along with signed distance lets use function that avoids outliers.

One such function is sigmoid function.

- Why sigmoid :question:
    - Sigmoid function tappers off when there is large value (outlier). When there is small value, sigmoid function behaves linearly
    - Provides nice probabilistic interpretation
    - The sigmoid function’s range is bounded between 0 and 1. Thus it’s useful in calculating the probability for the Logistic function
    - It’s derivative is easy to calculate than other functions which is useful during gradient descent calculation
    - It is a simple way of introducing non-linearity to the model.

Now, Optimization problem looks like:

![Logistic Regression Mind Map](https://miro.medium.com/max/1028/0*7taiX_Iha0TGssUP)

Lets use monotonic function in above optimization problem for ease to operate. After using monotonic function equation looks like below:

 <!--- ![Logistic Regression Mind Map](https://miro.medium.com/max/1182/1*yQqRdmXzNPpWr5is5MYLgQ.png) --->
 EQUATION UPDATE HERE
 
simple rule in optimization is shown below:

![Logistic Regression Mind Map](https://miro.medium.com/max/1400/1*jQ9dzMvlTUy9i8a1gujTiA.png) 

Is the optimization problem of logistic regression.

## 4. Weight vector

From optimization problem, the w* is the weight vector.

w is the d dim vector. For every feature fi, corresponding wi weight is associated.

Interpretation of w:
- when w is positive, P(yq = +1) increases
- when w is negative, P(yq = -1) decreases

Therefore, given fi, the corresponding weight vector is positive then for any query point (xqi) the value corresponding to ith value increases. Its probability of it belonging to positive class increases & vice versa.

## 5. L2 regularization; overfitting and underfitting

The weight vector (w) tends to positive and negative infinity.

#### How to control :question:
- Regularization

![Logistic Regression Mind Map](https://i.stack.imgur.com/HwqTv.png)

The L2 regularization does not let w tends to infinity. Remaining part of OP is known as loss term

Lambda is the hyperparameter in logistic regression

- when lambda = 0; no regularization. Prone to overfit (high variance)
- when lambda = large value; influence of loss term is reduced i,e not using train data to find best w. Prone to underfit (High bias)

#### Pattern in ML

Min ( loss function over training data + regularization )

#### How to find best lambda value :question:
- Using Cross validation 
    - k-fold CV
    - simple CV

## 6. L1 regularization & sparsity

![Logistic Regression Mind Map](https://i.stack.imgur.com/dp7F2.png)

Any alternatives to L2 reg is L1 reg.

- The L1 regularization does not let w tends to infinity
- Remaining part of OP is known as loss term
- Lambda is the hyperparameter in logistic regression

L1 reg and L2 reg are used for same purpose, but L1 reg has 1 major advantage i,e sparsity

#### what is sparsity :question: 
- solution to logistic regression is said to be parse if many w's in weight vector are zero.
- When we use L1 reg, in logistic regression all less important features become zero.
- When we use L2 reg, wi becomes small values but not necessary zero

Another alternate to L2 reg is elastic net.

Elastic net utilizes both L1 reg and L2 reg.

Update equation!!

## 7. Logistic regression using Loss minimization

Loss minimization interpretation:
- Loss function as logistic loss gives Logistic regression
- Loss function as hinge loss gives SVM
- Loss function as exponential loss gives Adaboost
- Loss function as squared loss gives Linear regression

## 8. Hyperparameter

Lambda in optimization problem is the hyperparameter in logistic regression. 

- when lambda = 0; overfit
- when lambda = infinity; underfit

#### How to find right lambda value :question:
- Grid search
- Random search

## 9. Column/feature standardization

In logistic regression, mandatory to perform column/feature standardization before training the model because we're computing the distance between the line/plane and query point.

## 10. Feature importance and Model interpretation

When we compute optimal w in weight vector, we can determine feature importance. 

#### How :question:
- Pick the abosolute value of large values present in w vector.

Utilizing the feature importance, another benefit is logistic regression is interpretation.

## 11. Collinearity or multicollinearlity

## 12. Real world cases

##### 12.1 Decision surface in Logistic Regression
- Linear/hyperplane

##### 12.2 Assumption in Logistic Regression
- Data is linearly/almost linearly seperable
    
##### 12.3 Feature importance and interpretability in Logistic Regression
- Top values of w weight vector
    
##### 12.4 What to do when dataset is imbalanced in Logistic Regression :question:
- Perform Up/Down sampling
    
##### 12.5 What happpens when there are outliers in Logistic Regression :question:
- logistic regression has less impact due to sigmoid function
- remove outliers and train
    
##### 12.6 What if missing values are present in Logistic Regression :question:
- Traditional strategies ([reference](https://github.com/Akshaykumarcp/ML-Feature-Engineering/tree/main/0.3_missing%20value%20imputation))

##### 12.7 Best and worst case in Logistic Regression
- Binary seperable
- Low latency
- Fast to train
- Works fairly well on large dataset
- Worst case: When data is not linearly seperable

## 13. What if data is not linearly seperable :question:
 Logistic Regression works well if linearly seperable. What if data is not linearly seperable ??
- When data is not linearly seperable, we've to perform feature transformation such that features are transformed from original feature space to transformed feature space.

#### How to know which transform to apply :question:
- We get know by learning/practising/experience.

## 14. Multinomial Logistics Regression( Number of Labels >2)
- Many times, there are classification problems where the number of classes is greater than 2. 
- We can extend Logistic regression for multi-class classification. 
- The logic is simple; we train our logistic model for each class and calculate the probability(hθx) that a specific feature belongs to that class. 
- Once we have trained the model for all the classes, we predict a new value’s class by choosing that class for which the probability(hθx) is maximum. 
- Although we have libraries that we can use to perform multinomial logistic regression, we rarely use logistic regression for classification problems where the number of classes is more than 2. 
-   There are many other classification models for such scenarios. 

## 15. Advantages of Logisitic Regression
- It is very simple and easy to implement.
- The output is more informative than other classification algorithms
- It expresses the relationship between independent and dependent variables
- Very effective with linearly seperable data

## 16. Disadvantages of Logisitic Regression
- Not effective with data which are not linearly seperable
- Not as powerful as other classification models
- Multiclass classifications are much easier to do with other algorithms than logisitic regression
- It can only predict categorical outcomes

## 17. Generalized Linear Model {GLM}
- Extension to logistic regression is the GLM.
- Logistic regression in probability perspective is the combination of gaussian naive bayes plus the bernoulli random variable assumption on class labels
- From above statement, when we change bernoulli to multinomial distribution, we get multinomial logistic regression that is used for multiclass classification
- Similarly there are linear regression and poisson regression

## 18. Acknowledgements :handshake:
- [Google Images](https://www.google.co.in/imghp?hl=en-GB&tab=ri&authuser=0&ogbl)
- [Appliedai](https://www.appliedaicourse.com/)
- [Ineuron](https://ineuron.ai/)
- Other google sites

## 19. Connect with me  :smiley:
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/linkedin.svg" />](https://www.linkedin.com/in/akshay-kumar-c-p/)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/youtube.svg" />](https://www.youtube.com/channel/UC3l8RTE3zBRzUrHbSXpx-qA)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/github.svg" />](https://github.com/Akshaykumarcp)
[<img align="left" alt="" width="22px" src="https://simpleicons.org/icons/medium.svg" />](https://medium.com/@akshai.148)
