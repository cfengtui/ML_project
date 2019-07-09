library(data.table)
library(caret)
library(dplyr)
library(randomForest)
library(xgboost)
library(rpart)
library(randomForestExplainer)
## load daa
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","testing.csv")

training_data<-fread("training.csv")
testing_data<-fread("testing.csv")

## convert classe as factor
training_data$classe<-as.factor(training_data$classe)

## Remove zero features
nsv<-nearZeroVar(training_data, saveMetrics = TRUE)
logic0<-nsv[,4]==FALSE
training_data<-training_data[,..logic0]


## Remove variables contains a lot of NA's
na_count <-sapply(training_data, function(y) sum(is.na(y)))
na_count <- data.frame(na_count)
features_left<-rownames(na_count[na_count[,1]<19000,,drop=FALSE])


training_new<-training_data %>% select(one_of(features_left))

## delete the first 5 columns in the data set since they are not features
training_new<-training_new[,-1:-6]

testing<-testing<-testing_data %>% select(one_of(colnames(training_new)))

## split training data into training and cross validation set ##############
set.seed(1)
dpTrain <- createDataPartition(training_new$classe, p=0.7, list=FALSE)
training <- training_new [dpTrain,]
validation<- training_new [-dpTrain,]


## random forest  
set.seed(1)
rffit<-randomForest(classe~.,data=training)
varImpPlot(rffit)
dev.copy(tiff,"random_forest.tiff")
dev.off()


##XGB
# use cross-validation to tune parameters
label = as.numeric(training$classe)-1
xgb.cv(data = data.matrix(select(training,-classe)), 
               label = label,
               nfold=10,
               eta = 0.5,
               max_depth = 10, 
               nround=40, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 5,
               nthread = 3
)
xgbfit<-xgboost(data = data.matrix(select(training,-classe)), 
                label = label,
               eta = 0.5,
               max_depth = 10, 
               nround=40, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 5,
               nthread = 3
)
importance <- xgb.importance(model = xgbfit)
print(xgb.plot.importance(importance_matrix =importance, top_n = 20))
dev.copy(tiff,"XGB.tiff")
dev.off()


## decision tree
dtfit <- rpart(classe ~ ., data = training, method="class")
library(rattle)
fancyRpartPlot(dtfit)
dev.copy(tiff,"decision_tree.tiff",height=1500, width=2500)
dev.off()

##evaluate the models by using validation set
confusionMatrix(predict(dtfit,newdata=validation,type = "class"),validation$classe) # decision tree
confusionMatrix(predict(rffit,newdata=validation,type = "class"),validation$classe) # random forest
mat<- as.matrix(as.data.frame(lapply(select(validation,-classe), as.numeric)))
xgbpred<-as.data.frame(matrix(predict(xgbfit,mat),ncol =5,byrow = T))
xgbmax<-max.col(xgbpred, 'first')
xgbclass<-ifelse(xgbmax==1,"A",ifelse(xgbmax==2,"B",ifelse(xgbmax==3,"C",ifelse(xgbmax==4,"D","E"))))
confusionMatrix(as.factor(xgbclass),validation$classe) # XGB


## apply the best model (random forest) to test set
predict(rffit,newdata=testing,type="class")
