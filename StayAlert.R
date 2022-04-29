#Author=ORCID:0000-0002-3324-2255
#Task=KAGGLE:Stay Alert! The Ford Challenge 

#Structure=
#1.loading libraries and data, splitting the dataset
#2. lasso for variable selection
#3. KMO and PCA for linear comb.
#4. GBDT for binary classification
#5. Testing AUC and writing the submission file

#1.loading libraries and data, splitting the dataset
#1.loading libraries and data, splitting the dataset

#deleting memory.
rm(list = ls())

#packages.
#install.packages("glmnet")
library("glmnet",verbose=TRUE) # for lasso
#install.packages("pROC")
library("pROC", verbose=TRUE) # for calculating ROC
#install.packages("xgboost")
library("xgboost", verbose=TRUE) #for GBDT
#install.packages("psych")
library("psych", verbose=TRUE) # for KMO and PCA

#reading file.
ford1 <- read.csv(file = "...fordtrain.csv", header=TRUE, as.is=T, sep=",", fileEncoding = "UTF-8")

#50% split.
vm1 <- sample( c(1:dim(ford1)[1]) , 302165) #uneven n for medians, n=302165

training1 <- data.frame( ford1[ vm1, -(c(1:2)) ] )
testing1 <- data.frame(ford1[-vm1,-(c(1:2)) ] )  


#2. lasso for variable selection
#2. lasso for variable selection

glmmod <- glmnet( as.matrix(training1[,-1]), training1$IsAlert, alpha = 1, family = "binomial")

plot(glmmod, xvar="lambda", label=TRUE, , ylab="Reg Betas", xlab="ln(λ)", cex.lab=1.2, cex.axis=1.2, cex.main=1.2, cex.sub=1.2, xaxt="n")
axis(1, at = seq(-7.5, -2.5, by = 0.5), las=2, cex.axis=1.2)
abline(h=-0, lty="dotted" )

#finding the best lambda with repetad CV
blamda <- NULL 
for (c1 in 1:5)
{
  cv_model <- cv.glmnet(as.matrix(training1[,-1]), training1$IsAlert, alpha = 1, family="binomial", kfold=10 ,    type.measure = "auc") 
  best_lambda <- cv_model$lambda.min
  blamda[c1] <- best_lambda
  cat(best_lambda, "--")
}

hist(blamda)
summary(blamda)
real_best_lambda <- median(blamda)

#running the lasso with lambda identified.
lss_model <- glmnet(as.matrix(training1[,-1]), training1$IsAlert, alpha = 1, lambda = real_best_lambda, family="binomial")
coef(lss_model)

#p2, p8, v5, v6, v7, v9 betas are 0 or very low, so to be deleted
training2 <- training1[,-(c(9, 25, 26, 27,29))]
testing2 <-  testing1[,-(c(9, 25, 26,27,29))]
rm(training1, testing1, blamda) #need memory

#3. KMO and PCA for dat ext. 
#3. KMO and PCA for dat ext. 

KMO(training2[,c(2:8)]) #too low.
KMO(training2[,c(9:19)]) #acceptable for PCA.
KMO(training2[,c(20:26)]) #acceptable for PCA

pcE = psych::principal(training2[,c(9:19)], nfactors=1, rotate = "varimax", scores = TRUE)
pcV = psych::principal(training2[,c(20:26)], nfactors=1, rotate = "varimax", scores = TRUE)
training3 <- cbind(training2, pcE$scores, pcV$scores)

pcEtest = psych::principal(testing2[,c(9:19)], nfactors=1, rotate = "varimax", scores = TRUE)
pcVtest = psych::principal(testing2[,c(20:26)], nfactors=1, rotate = "varimax", scores = TRUE)
testing3 <- cbind(testing2, pcEtest$scores, pcVtest$scores)

#correcting col.names
colnames(training3)[27] <- "pce" 
colnames(testing3)[27] <- "pce"
colnames(training3)[28] <- "pcv"
colnames(testing3)[28] <- "pcv"

rm(training2, testing2, pcE, pcV, pcEtest, pcVtest) #need 4 memory

#4. GBDT for binary classification
#4. GBDT for binary classification
trainingx1 <- training3[,-1]
traininglabelx1 <- training3$IsAlert
testingx1 <- testing3[,-1]
testinglabelx1 <- testing3$IsAlert

dtrain <- xgb.DMatrix(as.matrix(trainingx1), label=as.matrix(traininglabelx1))
dtest <- xgb.DMatrix(as.matrix(testingx1), label=as.matrix(testinglabelx1))

#modell set
watchlist1 <- list(train=dtrain, test=dtest)
xgbmod <- xgb.train(
  data = dtrain, 
  watchlist = watchlist1,
  max.depth = 3, 
  eta = 0.5,  
  nthread = 1, 
  nrounds = 100,  
  objective = "binary:logitraw",
  eval_metric = "auc", 
  min_child_weight = 100)

#chekcing on variable imp.
vimp1 <- as.data.frame(xgb.importance(model = xgbmod))
rownames(vimp1) <- vimp1$Feature
vimp1

#5. Testing AUC and writing the submission file
#5. Testing AUC and writing the submission file
xgbmod$evaluation_log$test_auc[100]
xgbmod$evaluation_log$train_auc[100]

#reading the eval file
fordt <- read.csv(file = "...fordTest.csv", header=TRUE, as.is=T, sep=",", fileEncoding = "UTF-8") # "../input/stayalert/fordTest.csv"
#removing unused columns
fordt2 <-  fordt[,-(c(11, 27, 28,29,31))]
#adding component data
pcEtest = psych::principal(fordt2[,c(11:21)], nfactors=1, rotate = "varimax", scores = TRUE)
pcVtest = psych::principal(fordt2[,c(22:28)], nfactors=1, rotate = "varimax", scores = TRUE)
fordt3 <- cbind(fordt2, pcEtest$scores, pcVtest$scores)

#correcting col.names
colnames(fordt3)[29] <- "pce" 
colnames(fordt3)[30] <- "pcv"

fordt4 <- xgb.DMatrix( data=as.matrix(fordt3[,-(c(1:3))]), label=rep(NaN, dim(fordt3)[1])) 
pred <- predict(xgbmod, fordt4)
prediction <- as.numeric(pred > 0.5)
summary(prediction)

Prediction <-prediction #on cap missing
submissiondf <- data.frame(fordt$TrialID, fordt$ObsNum, Prediction)
colnames(submissiondf) <- c("TrialID", "ObsNum", "Prediction")

write.table(submissiondf, file = "..../submission.csv", append = FALSE, quote = FALSE,sep = "," , row.names = FALSE, fileEncoding = "UTF-8")

#deleting memory.
rm(list = ls())
