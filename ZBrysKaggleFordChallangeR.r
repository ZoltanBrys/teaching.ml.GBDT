#Author=ORCID:0000-0002-3324-2255
#Task=KAGGLE:Stay Alert! The Ford Challenge 

#Structure=
#1.loading libraries and data, splitting the dataset to 3 disjunct set
#2. boruta and lasso for variable selection
#3. KMO and PCA for linear comb of inf.
#4. GBDT for binary classification
#5. Testing AUC on the val dataset

#1.loading libraries and data, splitting the dataset to 3
#1.loading libraries and data, splitting the dataset to 3

rm(list = ls()) #deleting memory.
 
#packages.
#install.packages("glmnet")
library("glmnet",verbose=TRUE) # for lasso
#install.packages("pROC")
library("pROC", verbose=TRUE) # for calculating ROC
#install.packages("xgboost")
library("xgboost", verbose=TRUE) #for GBDT
#install.packages("psych")
library("psych", verbose=TRUE) # for KMO and PCA
#install.packages('Boruta')
library("Boruta", verbose=TRUE) # for featire eval.

#reading file.
ford1 <- read.csv(file = "...train.csv", header=TRUE, as.is=T, sep=",", fileEncoding = "UTF-8")

#50%-25%-25% split.
vm1 <- sample( c(1:dim(ford1)[1]) , replace= FALSE, 302165) #uneven n for medians, n=302165
restli <- c(1:dim(ford1)[1])[-vm1]  #temporary variable
vm2 <- sample(restli, 150523 ) #n=150523
vm3 <- restli[-which(restli %in% vm2)]
rm(restli) #temporary variable

training1 <- data.frame( ford1[ vm1, -(c(1:2)) ] )
testing1 <- data.frame(ford1[vm2,-(c(1:2)) ] )  
validation1 <- data.frame(ford1[vm3,-(c(1:2)) ] )
rm(vm1, vm2, vm3)


#2. lasso for variable selection
#2. lasso for variable selection

#boruta basic selection
boruta1 <- Boruta(IsAlert ~ ., data=training1, doTrace=1, maxRuns=11, pValue=0.001) 
plot(boruta1)
bor_atts<-attStats(boruta1)
bor_atts

#saveRDS(boruta1, file = "...boruta1_20220430.RDS")
rm(boruta1)

#p8, v7, v9 rejected, so we remove it from the data
dcn <- c(which( colnames(training1)=="P8" ) , which( colnames(training1)=="V7" ) ,which( colnames(training1)=="V9" ))

training2 <- training1[,-dcn]
testing2 <-  testing1[,-dcn]
validation2 <-  validation1[,-dcn]
rm(training1, testing1, validation1, dcn) #need4memory

#lasso selection
glmmod <- glmnet(as.matrix(training2[,-1]), training2$IsAlert, alpha = 1, family = "binomial")

plot(glmmod, xvar="lambda", label=TRUE, , ylab="Reg Betas", xlab="ln(λ)", cex.lab=1.2, cex.axis=1.2, cex.main=1.2, cex.sub=1.2, xaxt="n")

plot(glmmod, xvar="lambda", label=TRUE, , ylab="log(reg betas)", xlab="ln(λ)", cex.lab=1.2, cex.axis=1.2, cex.main=1.2, cex.sub=1.2, xaxt="n",  log="y")

#finding the best lambda with repetad CV
blamda <- NULL 
for (c1 in 1:10)
{
  cv_model <- cv.glmnet(as.matrix(training2[,-1]), training2$IsAlert, alpha = 1, family="binomial", kfold=4 ,    type.measure = "auc") 
  best_lambda <- cv_model$lambda.min
  blamda[c1] <- best_lambda
  cat(best_lambda, "--")
}

hist(blamda)
summary(blamda)
real_best_lambda <- median(blamda)

rm(blamda,cv_model,c1, best_lambda)

#running the lasso with lambda identified.
lss_model <- glmnet(as.matrix(training2[,-1]), training2$IsAlert, alpha = 1, lambda = real_best_lambda, family="binomial")
coef(lss_model)

#E11, V4 are out based on LASSO
dcn <- c(which( colnames(training2)=="E11" ) , which( colnames(training2)=="V4" ))
training3 <- training2[,-dcn]
testing3 <-  testing2[,-dcn]
validation3 <-  validation2[,-dcn]
rm(training2, testing2, validation2, dcn, real_best_lambda, glmmod, lss_model) #need4memory


#3. KMO and PCA for dat ext. 
#3. KMO and PCA for dat ext. 

KMO(training3[,c(2:8)]) #P block , KMO too low and all MSA are low.
KMO(training3[,c(9:18)]) #E block, acceptable for PCA, except E1, E2, E5, E4, E6
KMO(training3[,c(11,15:18)]) #partly E block(11,15:18), acceptable for PCA
KMO(training3[,c(19:26)]) #V block acceptable for PCA, KMO ok, MSA ok

pcE = psych::principal(training3[,c(11,15:18)], nfactors=1, rotate = "varimax", scores = TRUE)
pcV = psych::principal(training3[,c(19:26)], nfactors=1, rotate = "varimax", scores = TRUE)
training4 <- cbind(training3, pcE$scores, pcV$scores)

pcEtest = psych::principal(testing3[,c(11,15:18)], nfactors=1, rotate = "varimax", scores = TRUE)
pcVtest = psych::principal(testing3[,c(19:26)], nfactors=1, rotate = "varimax", scores = TRUE)
testing4 <- cbind(testing3, pcEtest$scores, pcVtest$scores)

pcEtest = psych::principal(validation3[,c(11,15:18)], nfactors=1, rotate = "varimax", scores = TRUE)
pcVtest = psych::principal(validation3[,c(19:26)], nfactors=1, rotate = "varimax", scores = TRUE)
validation4 <- cbind(validation3, pcEtest$scores, pcVtest$scores)

#correcting col.names
colnames(training4)[27] <- "pce" 
colnames(testing4)[27] <- "pce"
colnames(validation4)[27] <- "pce"
colnames(training4)[28] <- "pcv"
colnames(testing4)[28] <- "pcv"
colnames(validation4)[28] <- "pcv"

rm(training3, testing3, validation3, pcE, pcV, pcEtest, pcVtest) #need 4 memory

#4. GBDT for binary classification
#4. GBDT for binary classification
trainingx1 <- training4[,-1]
traininglabelx1 <- training4$IsAlert
testingx1 <- testing4[,-1]
testinglabelx1 <- testing4$IsAlert
validationx1 <- validation4[,-1]
validationlabelx1 <- validation4$IsAlert

dtrain <- xgb.DMatrix(as.matrix(trainingx1), label=as.matrix(traininglabelx1))
dtest <- xgb.DMatrix(as.matrix(testingx1), label=as.matrix(testinglabelx1))
dval <- xgb.DMatrix(as.matrix(validationx1), label=as.matrix(validationlabelx1))

#modell set
watchlist1 <- list(train=dtrain, test=dtest)
xgbmod <- xgb.train(
  data = dtrain, 
  watchlist = watchlist1,
  max.depth = 9, 
  eta = 0.5,  
  nthread = 1, 
  nrounds = 9,  
  objective = "binary:logitraw",
  eval_metric = "auc", 
  min_child_weight = 1024)

#chekcing on variable imp.
vimp1 <- as.data.frame(xgb.importance(model = xgbmod))
rownames(vimp1) <- vimp1$Feature
vimp1

#dec.tree
xgb.dump(xgbmod, with_stats = TRUE) # GBDT-dontesi fa.


#5. Testing VAL AUC 
#5. Testing VAL AUC 

#reading the eval file
pred <- predict(xgbmod, dval)
prediction <- as.numeric(pred > 0.5)
summary(prediction)

#ROC
roc(validationlabelx1,prediction,
    smoothed = TRUE,
    ci=TRUE, ci.alpha=0.9, stratified=FALSE,
    plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
    print.auc=TRUE, show.thres=FALSE)

#Area under the curve: 0.8933
#95% CI: 0.8917-0.8948 (DeLong)


#deleting memory.
rm(list = ls())
