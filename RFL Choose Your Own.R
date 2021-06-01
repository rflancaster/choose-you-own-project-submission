### Choose Your Own Project: Richard Lancaster
### Predicting Countries of Universities based on Rankings Indicators

#Packages and Libraries
install.packages("caret", repos = "http://cran.us.r-project.org")
install.packages("dplyr", repos = "http://cran.us.r-project.org")
library(dplyr)
library(caret)

## Step 1
## Read data files
##

THEdata <- read.csv("./timesData.csv")

##
## STEP 2: remove unnecessary data and NAs, empty cells and ‘-‘s
## I will keep the (2016) data for the countries from North America, Asia, and Europe with the most universities

## 2A: order by most universities (per country) in the rankings
THEdata <- THEdata %>% filter(year == 2016) #only using 2016 data
top <- data.frame(table(THEdata$country)) #gives number of universities for each country
top[order(top$Freq, decreasing = TRUE),] #puts into descending order

##Note: The top 3 countries are USA, UK and Japan, which are already one from each of the 3 continents I will use

## 2B: remove all data except for the 3 chosen countries
top3 <- c("United States of America", "United Kingdom", "Japan") 
THEdata <- THEdata %>% filter(country %in% top3)
THEdata$country <- factor(THEdata$country) #removes the now unused factors for the other countries to avoid errors later

## 2C - Remove total_score column (I am not using this, and it has a lot of missing values which will cause problems later, and I want to avoid removing all rows with NAs for this column)
THEdata <- subset(THEdata, select = -total_score)

## 2D - Set all empty cells and those with dash to NA, which makes it easier to remove them
THEdata[THEdata==""] <- NA
THEdata[THEdata=="-"] <- NA

## 2E - remove all rows with NAs (now including empty cells and dashes)
THEdata <- na.omit(THEdata)

##
## STEP 3: convert value types, and remove unwanted characters
##

## 3A: Set seed
set.seed(501) 

## 3B: Convert all international_students variables to character, remove %, then numeric
THEdata$international_students <- as.character(THEdata$international_students)
THEdata$international_students <- strsplit(THEdata$international_students, "%")
THEdata$international_students <- as.numeric(THEdata$international_students)

## 3C: Convert international variables to character then numeric
THEdata$international <- as.character(THEdata$international)
THEdata$international <- as.numeric(THEdata$international)

## 3D: Convert income variables to character then numeric
THEdata$income <- as.character(THEdata$income)
THEdata$income <- as.numeric(THEdata$income)


## 3E: Check for NAs in international, international_students, and income (none found)
sum(is.na(THEdata$international))
sum(is.na(THEdata$international_students))
sum(is.na(THEdata$income))


##
## STEP 4: create test, train and validation sets
##

## 4A: Create validation set (for final test)
use_index <- createDataPartition(y = THEdata$country, times = 1, p = 0.5, list = FALSE)
use_set <- THEdata[-use_index,]
validation_set <- THEdata[use_index,]

## 4B: Create test and train sets for each country
test_index <- createDataPartition(y = use_set$country, times = 1, p = 0.9, list = FALSE)
train_set <- THEdata[-test_index,]
test_set <- THEdata[test_index,]

##Note: These values for p give sets which are reasonably similar in size (see below) which helps for adjusting for the confusion matrix

## 4C: Check lengths of test, train and validation sets
length(test_set$country) #109
length(train_set$country) #129
length(validation_set$country) #119

## 4D: Make test, train and validation sets the same length (109) to avoid problems with confusion matrix
train_set <- train_set[1:109,]
validation_set <- validation_set[1:109,]

##
## STEP 5: use knn algorithm
##

## 5A: predict country based on the values for international students, international score, and income
knn_fit <- knn3(train_set$country ~ train_set$international_students + train_set$international + train_set$income, data = train_set)

y_hat_knn <- predict(knn_fit, test_set, type = "class")


##
## STEP 6: judge quality of algorithm using confusion matrix
##

confusionMatrix(data = y_hat_knn, reference = test_set$country)$overall["Accuracy"]

###Accuracy = 0.4036697

##
## STEP 7: Optimize algorithm for improved accuracy (find best k value)
##

## 7A - Create empty list to store accuracy values
knn_acc_list = list()

## 7B - Use for loop to test k values (from 1 to 40)
for(i in 1:40){
  fit <- knn3(train_set$country ~ train_set$international_students + train_set$international + train_set$income, data = train_set, k = i)
  
  y_hat <- predict(fit, train_set, type = "class")
  
  x <-  confusionMatrix(data = y_hat, reference = test_set$country)$overall["Accuracy"]
  
  knn_acc_list[[i]] = x
}

## 7C: Show accuracy for each k value and find max

knn_acc_list
which.max(knn_acc_list) #32	
k_best <- which.max(knn_acc_list)


##
## STEP 8: Run again with best k and test
##

## 8A: Use knn with k = k_best
knn_fit <- knn3(train_set$country ~ train_set$international_students + train_set$international + train_set$income, data = train_set, k=k_best)

y_hat_knn <- predict(knn_fit, validation_set, type = "class")

## 8B: Check accuracy (=0.487)
confusionMatrix(data = y_hat_knn, reference = validation_set$country)$overall["Accuracy"]

## 8C: Give full confusion matrix output
confusionMatrix(data = y_hat_knn, reference = validation_set$country)



###Accuracy = 0.4678899



