# ML models implementation of Capstone project


#------------------loading required packages--------
#install.packages("randomForest")
library(caTools)  #for splitting the data in training and test sets
library(class)      #for KNN algorithm
library(e1071)   #for both Naive Bayes and SVM
library(rpart)   #for decision tree
library(rpart.plot)   #for plotting tree for decision tree
library(randomForest)  #for random forest algorithm


#----------importing the data-------
traffic <- read.csv("C:/Users/Pawan/OneDrive/Desktop/Capstone Project/city_traffic_sample.csv")
#View(traffic)


#--------data pre-processing---------
#removing unwanted columns (i.e. X)
traffic <- traffic[,-4]
traffic <- traffic[,-9]

#checking for NA values and omitting them
#na_value <- sum(is.na(traffic))
#traffic <- na.omit(traffic)

#labeling categorical data (or converting categorical data to numerical data for using them in the model)
#table(traffic$Day.Of.Week)
traffic$City <- as.numeric(factor(traffic$City,
                            levels = c('Bengalore','Chennai','Delhi','Kolkata','Mumbai','Pune'),
                            labels = c(1:6)))
traffic$Vehicle.Type <- as.numeric(factor(traffic$Vehicle.Type,
                                  levels = c('Bike','Bus','Car','Rickshaw'),
                                  labels = c(1:4)))
traffic$Weather <- as.numeric(factor(traffic$Weather,
                                  levels = c('Clear','Hot','Rainy','Snowy','Storm'),
                                  labels = c(1:5)))
traffic$Day.Of.Week <- as.numeric(factor(traffic$Day.Of.Week,
                                  levels = c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'),
                                  labels = c(1:7)))

#converting numerical data to categorical data
traffic$Traffic.Density <- ifelse(traffic$Traffic.Density <= 0.33, "Low Traffic",
                                  ifelse(traffic$Traffic.Density <= 0.66, "Moderate Traffic", "High Traffic"))
#View(traffic)
#table(traffic$Traffic.Density)


#----------------------dividing data in training and testing--------
set.seed(456)
split = sample.split(traffic$Traffic.Density, SplitRatio = 0.7) 
traffic_train = subset(traffic, split == TRUE) 
traffic_test = subset(traffic, split == FALSE) 


#-----------------feature scaling-----
#feature scaling (to bring data on the same scale so algorithm do not give preference to larger value during prediction)
traffic_train[,-9] <- scale(traffic_train[,-9])
traffic_test[,-9] <- scale(traffic_test[,-9])
#View(traffic_train)
#View(traffic_test)


#--------Machine learning models implementation for traffic prediction-----

traffic_predict <- function(train_data, test_data) {
  # 1. K nearest neighbors
  knn_train <- train_data
  knn_test <- test_data
  knn_predict <- knn(train = knn_train[,-9],
                     test = knn_test[,-9],
                     cl = knn_train[,9],
                     k = 93)
  knn_accuracy <- (sum(knn_predict == knn_test[,9]) / length(knn_test[,9])) * 100
  #knn_accuracy
  
  # 2. Naive Bayes
  nb_train <- train_data
  nb_test <- test_data
  nbc <- naiveBayes(Traffic.Density ~ .,
                    data = nb_train)
  nb_predict <- predict(nbc, nb_test[,-9])
  nb_accuracy <- (sum(nb_predict == nb_test[,9]) / length(nb_test[,9])) * 100
  #nb_accuracy
  
  # 3. Decision Tree
  dt_train <- train_data
  dt_test <- test_data
  tree <- rpart(formula = Traffic.Density ~ .,
                data = dt_train,
                method = "class")
  rpart.plot(tree)
  dt_predict <- predict(tree, dt_test, type = "class")
  dt_accuracy <- (sum(dt_predict == dt_test[,9]) / length(dt_test[,9])) * 100
  #dt_accuracy
  
  # 4. Support Vector Machine
  svm_train <- train_data
  svm_test <- test_data
  svm_train$Traffic.Density <- as.factor(svm_train$Traffic.Density)
  svm_test$Traffic.Density <- as.factor(svm_test$Traffic.Density)
  svm_classifier <- svm(formula = Traffic.Density ~ .,
                        data = svm_train,
                        cost = 8.1)
  svm_predict <- predict(svm_classifier,
                         newdata = svm_test[,-9])
  svm_accuracy <- (sum(svm_predict == svm_test[,9]) / length(svm_test[,9])) * 100
  #svm_accuracy
  
  # 5. Random Forest
  rf_train <- train_data
  rf_test <- test_data
  rf_train$Traffic.Density <- as.factor(rf_train$Traffic.Density)
  rf_test$Traffic.Density <- as.factor(rf_test$Traffic.Density)
  rf_model <- randomForest(Traffic.Density ~ .,
                           data = rf_train, ntree = 1000)
  rf_predict <- predict(rf_model, 
                        newdata = rf_test[,-9])
  rf_accuracy <- (sum(rf_predict == rf_test[,9]) / length(rf_test[,9])) * 100
  #rf_accuracy
  
  
  comparison <- format(data.frame("Algorithms_Used" = c("K-nearest neighbors","Naive Bayes","Decision Tree","Support Vector Machine","Random Forest"),
                           "Accuracy" = c(knn_accuracy,nb_accuracy,dt_accuracy,svm_accuracy,rf_accuracy)), width = 10)
  print("-------ACCURACY OF THE ALGORITHMS PREDICTING THE TRAFFIC ARE-------")
  print(comparison)
  
}


#----------Printing the result-------
traffic_predict(traffic_train, traffic_test)


