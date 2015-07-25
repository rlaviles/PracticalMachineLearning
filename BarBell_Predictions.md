### Weight Lifting Exercise Prediction Using Human Activity Recognition (HAR) Data

#### Abstract

The objective of this work is to predict the way an exercise has been
performed by using the 'classe' variable in the training dataset
collected by the HAR project, part of <Groupware@LES>, "a research and
development group on technologies for team work", in Universidad
Catolica de Rio de Janeiro, Brasil. In this dataset, a group of 6 people
has been asked to perform barbell lifts in 5 diferent ways, the right
one and 4 wrong ones. One goal was to teach people how to workout the
right way, for both health reasons and as better use of their time.

1.  About the Data.  
    Training and Testing are matrices of (19622,160) and (20,160) {rows,
    columns} respectively. After inspection, we can discard columns like
    X, raw\_timestamp\_part\_1, raw\_timestamp\_part\_2,
    cvtd\_timestamp. There are many NA values and also, many 'zero or
    near zero variance' predictors. Now we load libraries and start
    preparing the data:

<!-- -->

    library(caret); library(knitr); library(plyr)  
    DTraining<-read.csv("pml-training.csv", stringsAsFactors=FALSE) # load training and testing datasets
    DTesting<-read.csv("pml-testing.csv", stringsAsFactors=FALSE)
    dtraining<-subset(DTraining, select = -c(X,raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp))
    nzv<-nearZeroVar(dtraining, saveMetrics=TRUE)
    nzv<-subset(nzv, nzv==TRUE)     # a subset of 60 columns of the original data
    dtraining<-subset(dtraining, select = !names(dtraining) %in% rownames(nzv))
    dNA<-function(vector){ return(sum(is.na(vector))) }
    NAnum<-sapply(dtraining, dNA)
    noNA<-names(NAnum[NAnum / nrow(dtraining) == 0])
    dtraining<-subset(dtraining, select = names(dtraining) %in% noNA)
    print(matrix(noNA,ncol=5))  # the remaining useful columns of information after 'cleaning'.

    ##       [,1]               [,2]              [,3]                  
    ##  [1,] "user_name"        "accel_belt_z"    "accel_arm_x"         
    ##  [2,] "num_window"       "magnet_belt_x"   "accel_arm_y"         
    ##  [3,] "roll_belt"        "magnet_belt_y"   "accel_arm_z"         
    ##  [4,] "pitch_belt"       "magnet_belt_z"   "magnet_arm_x"        
    ##  [5,] "yaw_belt"         "roll_arm"        "magnet_arm_y"        
    ##  [6,] "total_accel_belt" "pitch_arm"       "magnet_arm_z"        
    ##  [7,] "gyros_belt_x"     "yaw_arm"         "roll_dumbbell"       
    ##  [8,] "gyros_belt_y"     "total_accel_arm" "pitch_dumbbell"      
    ##  [9,] "gyros_belt_z"     "gyros_arm_x"     "yaw_dumbbell"        
    ## [10,] "accel_belt_x"     "gyros_arm_y"     "total_accel_dumbbell"
    ## [11,] "accel_belt_y"     "gyros_arm_z"     "gyros_dumbbell_x"    
    ##       [,4]                [,5]                 
    ##  [1,] "gyros_dumbbell_y"  "total_accel_forearm"
    ##  [2,] "gyros_dumbbell_z"  "gyros_forearm_x"    
    ##  [3,] "accel_dumbbell_x"  "gyros_forearm_y"    
    ##  [4,] "accel_dumbbell_y"  "gyros_forearm_z"    
    ##  [5,] "accel_dumbbell_z"  "accel_forearm_x"    
    ##  [6,] "magnet_dumbbell_x" "accel_forearm_y"    
    ##  [7,] "magnet_dumbbell_y" "accel_forearm_z"    
    ##  [8,] "magnet_dumbbell_z" "magnet_forearm_x"   
    ##  [9,] "roll_forearm"      "magnet_forearm_y"   
    ## [10,] "pitch_forearm"     "magnet_forearm_z"   
    ## [11,] "yaw_forearm"       "classe"

1.  Next steps:  

-   **preprocess** the data  
-   **partitionate** it and  
-   do a **preliminary check** of the Training subset

<!-- -->

    preProVals <-preProcess(dtraining[,-c(1,55)], method = c("center", "scale"))
    preProVals$user_name <- dtraining$user_name
    preProVals$classe <- dtraining$classe
    toTrain <- createDataPartition(preProVals$classe, p=0.75, list=F) #p = default value.check it!
    training<-dtraining[toTrain,]
    testing<-dtraining[-toTrain]
    ggplot(training, aes(x = classe, fill = classe)) + geom_bar(aes(classe)) + facet_grid(.~user_name)                                                                                       

![](BarBell_Predictions_files/figure-markdown_strict/unnamed-chunk-2-1.png)
3.Is time to **build and test models.** Let's use **4** methods for the
construction and further *evaluation of models*: Random Forest, radial
Support Vector Machines, K-Nearest Neighbours and Naive Bayes. KNN was
the fastest of the 3 methods (few time and small CPU for so many (183!)
possible choices: "names(getModelInfo())").

    library(randomForest); library(kernlab); library(klaR)  
    training$classe <- as.factor(training$classe)
    cvControl <-trainControl(method = "cv", number = 5, allowParallel = TRUE, verboseIter = TRUE)
    model1 <- train(classe ~., training, method = "rf", trControl = cvControl) #model with random forest
    model2 <- train(classe ~., training, method = "svmRadial", trControl = cvControl) # model with radial SVM
    model3 = train(classe ~., training, method = "knn", trControl = cvControl)
    model4 = train(classe ~., training, method = "nb", trControl = cvControl)

    model1[4]; model1[11]

    ## $results
    ##   mtry  Accuracy     Kappa   AccuracySD     KappaSD
    ## 1    2 0.9938171 0.9921785 0.0014086760 0.001781515
    ## 2   30 0.9972142 0.9964762 0.0009429438 0.001193067
    ## 3   58 0.9936812 0.9920071 0.0019160514 0.002423951

    ## $finalModel
    ## 
    ## Call:
    ##  randomForest(x = x, y = y, mtry = param$mtry) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 30
    ## 
    ##         OOB estimate of  error rate: 0.22%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 4184    1    0    0    0 0.0002389486
    ## B    7 2836    4    1    0 0.0042134831
    ## C    0    6 2561    0    0 0.0023373588
    ## D    0    0    7 2404    1 0.0033167496
    ## E    0    1    0    4 2701 0.0018477458

    model2[4]; model2[11]

    ## $results
    ##        sigma    C  Accuracy     Kappa  AccuracySD     KappaSD
    ## 1 0.01174385 0.25 0.8688681 0.8342002 0.004023541 0.005183003
    ## 2 0.01174385 0.50 0.9043351 0.8788745 0.003217659 0.004124142
    ## 3 0.01174385 1.00 0.9319882 0.9138562 0.002291787 0.002942151

    ## $finalModel
    ## Support Vector Machine object of class "ksvm" 
    ## 
    ## SV type: C-svc  (classification) 
    ##  parameter : cost C = 1 
    ## 
    ## Gaussian Radial Basis kernel function. 
    ##  Hyperparameter : sigma =  0.0117438478139355 
    ## 
    ## Number of Support Vectors : 7090 
    ## 
    ## Objective Function Value : -1051.113 -805.3369 -744.9408 -428.5018 -1015.412 -558.2042 -656.9748 -1079.438 -666.0021 -594.2111 
    ## Training error : 0.055714

    model3[4]; model3[11]

    ## $results
    ##   k  Accuracy     Kappa  AccuracySD     KappaSD
    ## 1 5 0.9020248 0.8760225 0.003861991 0.004865511
    ## 2 7 0.8857183 0.8553788 0.004363310 0.005537797
    ## 3 9 0.8700233 0.8355652 0.005465978 0.006895857

    ## $finalModel
    ## 5-nearest neighbor classification model
    ## Training set class distribution:
    ## 
    ##    A    B    C    D    E 
    ## 4185 2848 2567 2412 2706

    model4[4]

    ## $results
    ##   usekernel fL  Accuracy     Kappa AccuracySD    KappaSD
    ## 1     FALSE  0 0.5576169 0.4427306 0.01162287 0.01431810
    ## 2      TRUE  0 0.7554013 0.6920216 0.01361968 0.01697867

I have measured the CPU time requested to run each model: Random Forest,
20 minutes; radial SVM, 12 minutes; k Nearest Neighbour, 2 minutes;
Naive Bayes, 5 minutes (a number of warings have been 'discarded' since
my objective was only to test NB, not to use it.) And now, check the
results (in terms of accuracy) for these 4 models:

    accuracy <- data.frame(Model=c("Random Forest", "SVM (radial)", "KNN", "Naive Bayes"),
                           Accuracy=c(round(max(head(model1$results)$Accuracy), 5),
                                       round(max(head(model2$results)$Accuracy), 5),
                                       round(max(head(model3$results)$Accuracy), 5),
                                       round(max(head(model4$results)$Accuracy),5)))
    accuracy

    ##           Model Accuracy
    ## 1 Random Forest  0.99721
    ## 2  SVM (radial)  0.93199
    ## 3           KNN  0.90202
    ## 4   Naive Bayes  0.75540

As we can see, the best model is number one, constructed with the Random
Forest method (accuracy\~99.7%) and the 'worst' was Naive Bayes
(accuracy \~ 76.2%) Now we can use those 4 models and do predictions
using the samples in the 'testing' dataset.

    DTesting <- DTesting[, names(DTesting) %in% names(training)]
    testmodelRF <- predict(model1, DTesting)
    testmodelSVM<- predict(model2, DTesting)
    testmodelKNN<- predict(model3, DTesting)
    testmodelBN<- predict(model4, DTesting)

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 1

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 2

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 3

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 4

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 5

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 6

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 7

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 8

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 9

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 10

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 11

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 12

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 13

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 14

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 15

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 16

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 17

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 18

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 19

    ## Warning in FUN(1:20[[20L]], ...): Numerical 0 probability for all classes
    ## with observation 20

    ## Construct a table and check if the models agree in their predictions
    predTable <- data.frame(rfPred = testmodelRF, svmPred = testmodelSVM, knnPred = testmodelKNN, bnPred = testmodelBN)
    predTable$agree <- with(predTable, rfPred == svmPred && rfPred == knnPred && rfPred == bnPred)
    Agreement <- all(predTable$agree)

    kable(predTable)  ## an ordered summary of results

<table>
<thead>
<tr class="header">
<th align="left">rfPred</th>
<th align="left">svmPred</th>
<th align="left">knnPred</th>
<th align="left">bnPred</th>
<th align="left">agree</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">C</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">A</td>
<td align="left">B</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">D</td>
<td align="left">D</td>
<td align="left">D</td>
<td align="left">D</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">C</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">C</td>
<td align="left">C</td>
<td align="left">C</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">B</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">FALSE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">FALSE</td>
</tr>
<tr class="even">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">FALSE</td>
</tr>
</tbody>
</table>

    ## table(predTable)  ## a bit more of details  

As we can see, if we include **model 4 (naive Bayes)** the prediction
table display FALSE's. So discard Naive Bayes and we get:

    DTesting <- DTesting[, names(DTesting) %in% names(training)]
    testmodelRF <- predict(model1, DTesting)
    testmodelSVM<- predict(model2, DTesting)
    testmodelKNN<- predict(model3, DTesting)
    ## Construct a table and check if the models agree in their predictions
    predTable <- data.frame(rfPred = testmodelRF, svmPred = testmodelSVM, knnPred = testmodelKNN)
    predTable$agree <- with(predTable, rfPred == svmPred && rfPred == knnPred)
    Agreement <- all(predTable$agree)

    kable(predTable)  ## an ordered summary of results

<table>
<thead>
<tr class="header">
<th align="left">rfPred</th>
<th align="left">svmPred</th>
<th align="left">knnPred</th>
<th align="left">agree</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">A</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">D</td>
<td align="left">D</td>
<td align="left">D</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">C</td>
<td align="left">C</td>
<td align="left">C</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">E</td>
<td align="left">E</td>
<td align="left">E</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">A</td>
<td align="left">A</td>
<td align="left">A</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="odd">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
<tr class="even">
<td align="left">B</td>
<td align="left">B</td>
<td align="left">B</td>
<td align="left">TRUE</td>
</tr>
</tbody>
</table>

    ## table(predTable)  ## too much detail, I decided not to show this!  

Now we are ready to submit our results, considering only our Random
Forest model. Final remarks: we have taken up to a 75% (p=0.75) of data
for training and have tested the dataset with 4 different
methods/models. After test, we have discarded only Naive Bayes and
probed that Random Forest with cross validation is the better method for
the construction of a good model (higher accuracy.)

References:  
1.
<https://www.unt.edu/rss/class/Jon/Benchmarks/CrossValidation1_JDS_May2011.pdf>  
2.
<http://www.r-bloggers.com/near-zero-variance-predictors-should-we-remove-them>  
3. <https://cran.r-project.org/web/packages/caret/vignettes/caret.pdf>
NEW!  
4. <https://cran.r-project.org/web/views/MachineLearning.html>  
5. <https://cran.r-project.org/web/packages/klaR/index.html>  
(I am really sorry that I was unable (time!) to find the right way to
eliminate (not to show) 'useless' lines.)
