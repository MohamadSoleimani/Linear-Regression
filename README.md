# Linear Regression

## Introduction

In many situations, we want to predict an outcome based on certain known features. For example, how much does a house cost depending on its size? To figure this out, we can gather a dataset that lists house prices along with their sizes. By fitting a function to this data, we can estimate the price of any given house. The more accurately the function fits, the better our predictions will be. In this case, we’ll use a linear function and explore two different methods to find the best parameters for it.

## Linear Hypothesis

The linear hypothesis is defined as:

$$
h_w(x) = w_0 + \sum_{i=1}^{D} w_i x_i
$$

Where \(x_i\) refers to the feature vector and \(w_i\) refers to the parameter vector of the linear hypothesis.

## Solving the Problem

To find the best linear function, we need to determine the optimal parameters. To do this, we define a cost function. The sum of squared errors is the most common cost function used in linear regression problems. Based on this, we define the cost function as follows:

$$
J(w) = \sum_{i=1}^{n} (y_i - h_w(x_i))^2
$$

## Analytical Method

Having defined an appropriate cost function for our problem, we can use an analytical method to find the optimal parameters for the linear function. To do this, we must take the derivatives of the cost function with respect to each parameter and set them equal to zero.

The cost function derivative for \(w_0\) and \(w_1\) is:

$$
\frac{\partial J}{\partial w_0} = -2 \sum_{i=1}^{n} (y_i - w_0 - w_1 x_i) = 0
$$

$$
\frac{\partial J}{\partial w_1} = -2 \sum_{i=1}^{n} (y_i - w_0 - w_1 x_i) x_i = 0
$$

By solving the derivative with respect to \(w_0\), we obtain:

$$
\frac{\partial J}{\partial w_0} = \sum_{i=1}^{n} y_i - n w_0 - w_1 \sum_{i=1}^{n} x_i = 0
$$

Thus, we get:

$$
w_0 = \frac{\sum_{i=1}^{n} y_i - w_1 \sum_{i=1}^{n} x_i}{n}
$$

Substituting this into the equation for \(w_1\), we can solve for \(w_1\) as follows:

$$
w_1 = \frac{n \sum_{i=1}^{n} y_i x_i - \sum_{i=1}^{n} y_i \sum_{i=1}^{n} x_i}{n \sum_{i=1}^{n} x_i^2 - \left( \sum_{i=1}^{n} x_i \right)^2}
$$

## Conclusion

In this analytical method, we have derived the formula for calculating the optimal parameters for linear regression. The linear model now allows us to predict unknown values from given input data.

## Requirements

- Python 3.x
- NumPy
- Matplotlib (for visualization)

