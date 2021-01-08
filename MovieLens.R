##########################################################
# Create Helper Functions
##########################################################

#Function to calculate the Root Mean Squared Error
calcRMSE <- function(predictedResult, actualResult){
  # calculate difference between the predicted and actual results
  error <- predictedResult - actualResult
  # square the error
  squaredError <- error^2
  #find the mean of all the squared errors
  meanSquaredError <- mean(squaredError)
  # take the square root of the mean
  sqrt(meanSquaredError)
}

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes
# Install Required Packages
options(install.packages.compile.from.source = "always")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(forecast)) install.packages('forecast', dependencies = TRUE, repos = "http://cran.us.r-project.org")



library(tidyverse)
library(caret)
library(data.table)
library(forecast)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

#download, read and label files
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##########################################################
# Data Exploration
##########################################################

# View General Structure of Data 
head(edx)

# Tests if observations are properly randomized within data set
edxRowIndexes <- 1:length(edx$rating)
rowCorrelation <- cor(edxRowIndexes, edx$rating)
rowCorrelation

# Summary Statistics for Ratings 
All_Ratings <- edx$rating
# Helper function for Mode statistic
getMode <- function(list){
  unique(list)[which.max(tabulate(match(list, unique(list))))]  
}
#Number entries
nrow(All_Ratings)

## mean
ratingMean <- mean(All_Ratings)
ratingMean
## median
ratingMedian <- median(All_Ratings)
ratingMedian
## mode
ratingMode <- getMode(All_Ratings)
ratingMode

# Histograms

# Distribution of All Ratings
hist(All_Ratings)
# Distribution of Summary Statistics per User
## Set-Up
userStats <- edx %>% group_by(userId) %>% summarise(average=mean(rating), median=median(rating), mode=getMode(rating), amt=n())
Mean_Ratings_for_Each_User <- userStats$average
Median_Ratings_for_Each_User <- userStats$median
Mode_Ratings_for_Each_User <- userStats$mode
Number_Of_Ratings_for_Each_User <- userStats$amt

# Histogram of User Mean Scores
hist(Mean_Ratings_for_Each_User)
# Histogram of User Median Scores
hist(Median_Ratings_for_Each_User)

# Histogram of User Mode Scores
hist(Mode_Ratings_for_Each_User)

# Histogram of Number of Scores per User
hist(Number_Of_Ratings_for_Each_User)

# Distribution of Summary Statistics per Movie
## Set-Up
movieStats <- edx %>% group_by(movieId) %>% summarise(average=mean(rating), median=median(rating), mode=getMode(rating), amt=n())
Mean_Ratings_for_Each_Movie <- movieStats$average
Median_Ratings_for_Each_Movie <- movieStats$median
Mode_Ratings_for_Each_Movie <- movieStats$mode
Number_Of_Ratings_for_Each_Movie <- movieStats$amt

# Histogram of Movie Mean Scores
hist(Mean_Ratings_for_Each_Movie)
# Histogram of Movie Median Scores
hist(Median_Ratings_for_Each_Movie)
# Histogram of Movie Mode Scores
hist(Mode_Ratings_for_Each_Movie)
# Histogram of Number of Scores per Movie
hist(Number_Of_Ratings_for_Each_Movie)

# Distribution of Summary Statistics per unique genre set
## Set-Up
genresStats <- edx %>% group_by(genres) %>% summarise(average=mean(rating), median=median(rating), mode=getMode(rating), amt=n())
Mean_Ratings_for_Each_Set_of_Genres <- genresStats$average
Median_Ratings_for_Each_Set_of_Genres <- genresStats$median
Mode_Ratings_for_Each_Set_of_Genres <- genresStats$mode
Number_Of_Ratings_for_Each_Set_of_Genres <- genresStats$amt

# Histogram of Movie Mean Scores
hist(Mean_Ratings_for_Each_Set_of_Genres)
# Histogram of Movie Median Scores
hist(Median_Ratings_for_Each_Set_of_Genres)
# Histogram of Movie Mode Scores
hist(Mode_Ratings_for_Each_Set_of_Genres)
# Histogram of Number of Scores per Genres
hist(Number_Of_Ratings_for_Each_Set_of_Genres)


##########################################################
# Data Preparation
##########################################################

# Get 3 most popular Genres for Mutation

## find unique genre sets
uniqueGenres <- unique(edx$genres)
## find unique genres used in sets
baseGenres <- uniqueGenres[!str_detect(uniqueGenres, "\\|")]
## Find frequencies of each genre
genreFrequencies <- sapply(baseGenres, function(genre){
  sum(str_detect(edx$genres, genre))
})
# sort and print
sortGenreFreq <- sort(genreFrequencies, decreasing=TRUE)
sortGenreFreq

edx <- edx %>% mutate(isDrama=str_detect(edx$genres, "Drama"))
edx <- edx %>% mutate(isComedy=str_detect(edx$genres, "Comedy"))
edx <- edx %>% mutate(isAction=str_detect(edx$genres, "Action"))

##########################################################
# create train and test set
##########################################################

test_index2 <- createDataPartition(edx$userId, times = 1, p = 0.5, list = FALSE)
crossValidation1 <- edx[test_index2, ]
crossValidation2 <- edx[-test_index2, ]

##########################################################
# Create Model w/ Train Set 
##########################################################

# Create Simplistic Prediction Model based off mean score for all movies

means <- rep(mean(crossValidation1$rating), nrow(crossValidation1))

crossValidation1 <- crossValidation1 %>% mutate(prediction=means)

# Test Model
baseRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
baseRMSE
# Modify the model by adding the average effect for 
# a user's average rating to their predicted ratings

userEffect <- crossValidation1 %>% 
  group_by(userId) %>%
  summarize(userEffect = mean(rating-prediction))

crossValidation1 <- crossValidation1 %>%
  left_join(userEffect, by="userId")

crossValidation1 <- crossValidation1 %>% mutate(prediction=prediction + userEffect)

# Test Model
userRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
userRMSE
# Create a model adding the average effect for 
# a genre's average rating to its predicted ratings

genresEffect <- crossValidation1 %>% 
  group_by(genres) %>%
  summarize(genresEffect = mean(rating-prediction))

crossValidation1 <- crossValidation1 %>%
  left_join(genresEffect, by="genres")

crossValidation1 <- crossValidation1 %>% mutate(prediction=prediction + genresEffect)

# Test Model
genreRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
genreRMSE
# Create a model adding the average error for 
# a movie's average rating to its predicted ratings

movieEffect <- crossValidation1 %>% 
  group_by(movieId) %>%
  summarize(movieEffect = mean(rating-prediction))

crossValidation1 <- crossValidation1 %>%
  left_join(movieEffect, by="movieId")

crossValidation1 <- crossValidation1 %>% mutate(prediction=prediction + movieEffect)

# Test Model
movieRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
movieRMSE
# Create a model adding the average error for 
# comedy vs non-comedy movie average ratings to its predicted ratings

isComedyEffect <- crossValidation1 %>% 
  group_by(isComedy) %>%
  summarize(isComedyEffect = mean(rating-prediction))

crossValidation1 <- crossValidation1 %>%
  left_join(isComedyEffect, by="isComedy")

crossValidation1 <- crossValidation1 %>% mutate(prediction=prediction + isComedyEffect)

# Test Model
isComedyRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
isComedyRMSE

# Create a model adding the average error for 
# drama vs non-drama movie average ratings to its predicted ratings

isDramaEffect <- crossValidation1 %>% 
  group_by(isDrama) %>%
  summarize(isDramaEffect = mean(rating-prediction))

crossValidation1 <- crossValidation1 %>%
  left_join(isDramaEffect, by="isDrama")

crossValidation1 <- crossValidation1 %>% mutate(prediction=prediction + isDramaEffect)

# Test Model
isDramaRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
isDramaRMSE

# Create a model adding the average error for 
# action vs non-action movie average ratings to its predicted ratings

isActionEffect <- crossValidation1 %>% 
  group_by(isAction) %>%
  summarize(isActionEffect = mean(rating-prediction))

crossValidation1 <- crossValidation1 %>%
  left_join(isActionEffect, by="isAction")

crossValidation1 <- crossValidation1 %>% mutate(prediction=prediction + isActionEffect)

# Test Model
isActionRMSE <- calcRMSE(crossValidation1$prediction, crossValidation1$rating)
isActionRMSE

##########################################################
# Create Regularized Model w/ Train Set 
##########################################################
# Create Simplistic Prediction Model based off mean score for all movies

#Create an array of possible lambda values
lambda <- seq(2, 5, .1)

# Iterate over lambda values and train the normalized model with each
RegularizeRMSE <- sapply(lambda, function(l){
  
  means <- rep(mean(crossValidation2$rating), nrow(crossValidation2))
  
  crossValidation2 <- crossValidation2 %>% mutate(prediction=means)
  
  # Modify the model by adding the average error for 
  # a user's average rating to their predicted ratings w/ normalization
  
  userEffect <- crossValidation2 %>% 
    group_by(userId) %>%
    summarize(userEffect = sum(rating-prediction)/(n()+l))
  
  crossValidation2 <- crossValidation2 %>%
    left_join(userEffect, by="userId")
  
  crossValidation2 <- crossValidation2 %>% mutate(prediction=prediction + userEffect)
  
  # Create a model adding the average error for 
  # a genre's average rating to its predicted ratings w/ normalization
  
  genresEffect <- crossValidation2 %>% 
    group_by(genres) %>%
    summarize(genresEffect = sum(rating-prediction)/(n()+l))
  
  crossValidation2 <- crossValidation2 %>%
    left_join(genresEffect, by="genres")
  
  crossValidation2 <- crossValidation2 %>% mutate(prediction=prediction + genresEffect)
  
  # Create a model adding the average error for 
  # a movie's average rating to its predicted ratings w/ normalization
  
  movieEffect <- crossValidation2 %>% 
    group_by(movieId) %>%
    summarize(movieEffect = sum(rating-prediction)/(n()+l))
  
  crossValidation2 <- crossValidation2 %>%
    left_join(movieEffect, by="movieId")
  
  crossValidation2 <- crossValidation2 %>% mutate(prediction=prediction + movieEffect)
  
  # Test Model
  calcRMSE(crossValidation2$prediction, crossValidation2$rating)
})
# correspond lambda values to RMSE values
RMSE_For_Lambda <- RegularizeRMSE
RegularizeRMSE <- cbind(lambda, RMSE_For_Lambda)
# plot lambda values by RMSE
plot(RegularizeRMSE)
#Print Values
RegularizeRMSE
#Find index of min RMSE 
minIndex <- which.min(RegularizeRMSE[,2])
#Print Lambda values corresponding to min RMSE
RegularizeRMSEOptimizedLambda <- RegularizeRMSE


##########################################################
#	Validate RMSE
##########################################################
#Start clock for timing function
startTime <- proc.time()

minLambda = 3.5

means <- rep(mean(validation$rating), nrow(validation))

validation <- validation %>% mutate(prediction=means)

baseValidationRMSE <- calcRMSE(validation$prediction, validation$rating)

# Modify the model by adding the average error for 
# a user's average rating to their predicted ratings

userEffect <- validation %>% 
  group_by(userId) %>%
  summarize(userEffect = sum(rating-prediction)/(n()+minLambda))

validation <- validation %>%
  left_join(userEffect, by="userId")

validation <- validation %>% mutate(prediction=prediction + userEffect)

# Create a model adding the average error for 
# a genre's average rating to its predicted ratings

genresEffect <- validation %>% 
  group_by(genres) %>%
  summarize(genresEffect = sum(rating-prediction)/(n()+minLambda))

validation <- validation %>%
  left_join(genresEffect, by="genres")

validation <- validation %>% mutate(prediction=prediction + genresEffect)

# Create a model adding the average error for 
# a movie's average rating to its predicted ratings

movieEffect <- validation %>% 
  group_by(movieId) %>%
  summarize(movieEffect = sum(rating-prediction)/(n()+minLambda))

validation <- validation %>%
  left_join(movieEffect, by="movieId")

validation <- validation %>% mutate(prediction=prediction + movieEffect)

totalTime <- proc.time() - startTime

##########################################################
#Performance
##########################################################

# Test Model
validationRMSE <- calcRMSE(validation$prediction, validation$rating)
validationRMSE
# Total Time Taken in Seconds
totalTimeElapsed <- totalTime["elapsed"]
totalTimeElapsed
