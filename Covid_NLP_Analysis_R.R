library(olsrr)
library(tm)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(readr)
library(cowplot)
library(olsrr)
library(readr) 
library(caret)
library(pscl)
library(lmtest)
library(ipred)
library(survival)
library(ResourceSelection)
library(survey)
library(lmtest)
library(pROC)
library(DescTools)
library(wordcloud)
library(RColorBrewer)
library(caTools)
library(e1071)

#loading in CSV File
covid_tweets <- read.csv(file = "/Users/mattdolan/Documents/DatasetsR/Covid NLP Analysis/covid19_tweets_2021.csv", stringsAsFactors = F)

#looking at dynamics of data
head(covid_tweets) 
summary(covid_tweets) 

#looking at number of user verified tweets
table(covid_tweets$user_verified)

#checking the shape of the data 
dim(covid_tweets)
str(covid_tweets) 

#changing data type of tweets to allow for cleaning
covid_tweets$text <- as.character(covid_tweets$text)

#Changing categorical variable of user_verified to 0 for false, 1 for true
covid_tweets$user_verified <- as.integer(covid_tweets$user_verified=="True")

#vectorising text_corpus of tweets for transformation
text_corpus <- text_corpus(VectorSource(covid_tweets$text))

#lowering case of all characters within text_corpus
text_corpus <- tm_map(text_corpus, tolower)

#removing punctuation marks within text_corpus 
text_corpus <- tm_map(text_corpus, removePunctuation)

#stemming text_corpus, as opposed to lemmatising due to messy nature of tweets
#i.e. removing suffixes and prefixes to words that may hinder analytical value
text_corpus <- tm_map(text_corpus, stemDocument)

#producing a word-cloud of top 150 words to see impact of data transformations
wordcloud(text_corpus,max.words=150,random.order=FALSE, rot.per=0.15, colors=brewer.pal(8,"Dark2"))

#creating a document-term matrix to identify the frequency of particular terms in the text_corpus 
freq <- DocumentTermMatrix(text_corpus)
freq

#removing sparse terms from the document term matrix to clean data to more common phrases to avoid 
#noisy data that may distort analytical value of our bayesian model 
freq <- removeSparseTerms(freq, 0.90)
freq

#creates a dataframe from a matrix of our cleaned data 
newtext_corpus <- as.data.frame(as.matrix(freq))

#adding column names
colnames(newtext_corpus) <- make.names(colnames(newtext_corpus))

#adding user verified values
newtext_corpus$user_verified <- covid_tweets$user_verified

#splitting dataset into train/test
set.seed(1)
split <- sample.split(newtext_corpus$user_verified, SplitRatio = 0.7)
train <- subset(newtext_corpus, split==TRUE)
test <- subset(newtext_corpus, split==FALSE)

#Training the naive bayesian classifier
naiveuserverified <- naiveBayes(formula, as.factor(user_verified)~., data=train)
naiveuserverified

#using the naive bayesian
predictnaiveuserverified <- predict(naiveuserverified, newdata = test, type="class")

#creating confusion matrix
tab <- table(predictnaiveuserverified, test$user_verified)
caret::confusionMatrix(tab)  

#plotting confusion matrix
ggplot(test, aes(user_verified, predictnaiveuserverified, color = user_verified)) +
  geom_jitter(width = 0.2, height = 0.1, size=2) +
  labs(title="Confusion Matrix", 
       subtitle="Predicted vs. Observed from Covid Tweetst", 
       y="Predicted", 
       x="Truth")







