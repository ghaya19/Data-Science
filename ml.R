setwd("D:/Files/Documents/UM/Masters/SEM1/PRINCIPLE OF DATA SCIENCE/G.ass/app2")

library(tidyverse)
library(mlbench)
library(caret)
require(caTools)
library(rpart)
library(klaR)  #to use NaiveBayes()

data <- read.csv("D:/Files/Documents/UM/Masters/SEM1/PRINCIPLE OF DATA SCIENCE/G.ass/app2/survey_lung_cancer.csv", sep = ",", header = TRUE)
#Change the levels of the target variable to "1" and "0" which stand for YES and NO respectively
data$LUNG_CANCER=ifelse(data$LUNG_CANCER=="YES",1,0)
data$LUNG_CANCER=as.factor(data$LUNG_CANCER)
table(data$LUNG_CANCER)
intrain <- createDataPartition(y = data$LUNG_CANCER, p= 0.7, list = FALSE)
training <- data[intrain,]
testing <- data[-intrain,]

#To train control
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

#To SVM
seed = 101
metric <- "Accuracy"
preProcess=c("center", "scale")

#SVM
set.seed(seed)
fit.svm <- train(LUNG_CANCER ~., data = training, method = "svmLinear",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneLength = 10)

# Linear Discriminant Analysis
set.seed(seed)
fit.lda <- train(LUNG_CANCER~., data=training, method="lda", metric=metric, preProc=c("center", "scale"), trControl=trctrl)
#gbm
set.seed(seed)
fit.gbm <- train(LUNG_CANCER~., data=training, method="gbm", metric=metric, trControl=trctrl, verbose=FALSE)
# kNN
set.seed(seed)
fit.knn <- train(LUNG_CANCER~., data=training, method="knn", metric=metric, preProc=c("center", "scale"), trControl=trctrl)

save(fit.svm, fit.lda, fit.gbm, fit.knn,file = "models.RData")

results <- resamples(list(svm=fit.svm, lda=fit.lda, gbm=fit.gbm, knn=fit.knn))

#bagging=fit.treebag,

save(results, file = "models_accuracy.RData")
load("models_accuracy.RData")
load("models.RData")
test_results <- data.frame(testing$LUNG_CANCER)

test_results$predicted_svm <- predict(fit.svm, testing)
test_results$predicted_lda <- predict(fit.lda, testing)
test_results$predicted_gbm <- predict(fit.gbm, testing)
test_results$predicted_knn <- predict(fit.knn, testing)


confusionMatrix(test_results$testing.LUNG_CANCER, test_results$predicted_svm)
confusionMatrix(test_results$testing.LUNG_CANCER, test_results$predicted_lda)
confusionMatrix(test_results$testing.LUNG_CANCER, test_results$predicted_gbm)
#confusionMatrix(test_results$as.factor(testing.LUNG_CANCER), test_results$as.factor(predicted_knn))



# Table comparison
summary(results)
# boxplot comparison
bwplot(results)
# Dot-plot comparison
dotplot(results)

