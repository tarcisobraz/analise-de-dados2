##### Chapter 5: Classification using Decision Trees and Rules -------------------

#### Part 1: Decision Trees -------------------

## Understanding Decision Trees ----
# calculate entropy of a two-class segment
-0.60 * log2(0.60) - 0.40 * log2(0.40)

curve(-x * log2(x) - (1 - x) * log2(1 - x),
      col="red", xlab = "x", ylab = "Entropy", lwd=4)

## Example: Identifying Risky Bank Loans ----
## Step 2: Exploring and preparing the data ----
credit <- read.csv("credit.csv")
str(credit)

# look at two characteristics of the applicant
table(credit$checking_balance)
table(credit$savings_balance)

# look at two characteristics of the loan
summary(credit$months_loan_duration)
summary(credit$amount)

# look at the class variable
table(credit$default)

# create a random sample for training and test data
# use set.seed to use the same random number sequence as the tutorial
set.seed(12345)
credit_rand <- credit[order(runif(1000)), ]

# compare the credit and credit_rand data frames
summary(credit$amount)
summary(credit_rand$amount)
head(credit$amount)
head(credit_rand$amount)

# split the data frames
credit_train <- credit_rand[1:900, ]
credit_test  <- credit_rand[901:1000, ]

# check the proportion of class variable
prop.table(table(credit_train$default))
prop.table(table(credit_test$default))

## Step 3: Training a model on the data ----
# build the simplest decision tree
library(C50)
credit_model <- C5.0(credit_train[-17], credit_train$default)

# display simple facts about the tree
credit_model

plot(credit_model)

# display detailed information about the tree
summary(credit_model)

## Step 4: Evaluating model performance ----
# create a factor vector of predictions on test data
credit_pred <- predict(credit_model, credit_test)

# cross tabulation of predicted versus actual classes
library(gmodels)
CrossTable(credit_test$default, credit_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

## Step 5: Improving model performance ----

## Boosting the accuracy of decision trees
# boosted decision tree with 10 trials
credit_boost10 <- C5.0(credit_train[-17], credit_train$default,
                       trials = 10)
credit_boost10
summary(credit_boost10)

credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

# boosted decision tree with 100 trials (not shown in text)
credit_boost100 <- C5.0(credit_train[-17], credit_train$default,
                        trials = 100)
credit_boost_pred100 <- predict(credit_boost100, credit_test)
CrossTable(credit_test$default, credit_boost_pred100,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

## Making some mistakes more costly than others
# create a cost matrix
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2)
error_cost

# apply the cost matrix to the tree
credit_cost <- C5.0(credit_train[-17], credit_train$default,
                          costs = error_cost)
credit_cost_pred <- predict(credit_cost, credit_test)

CrossTable(credit_test$default, credit_cost_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))

#http://www.patricklamle.com/Tutorials/Decision%20tree%20R/Decision%20trees%20in%20R%20using%20C50.html
#https://www.packtpub.com/books/content/support/13251

#Caret
library(caret)
credit_caret <- train(default ~ ., data=credit_train, method="C5.0")

credit_caret_pred <- predict(credit_caret, credit_test)

CrossTable(credit_test$default, credit_caret_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual default', 'predicted default'))