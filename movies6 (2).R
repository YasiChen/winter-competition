#### Load libraries
setwd("~/OneDrive - Duke University/MQM Winter Competition/ml-25m")
source("DataAnalyticsFunctions.R")
library(data.table)
library(tidyverse)
library(caret) ## findCorrelation
library(stringr) # add leading 0 
library(rvest) # scrape website
library(splitstackshape) # create dummy variables for genres
library(glmnet) # Lasso
library(randomForest) # Random Forest
library(httr) ## for API
library(jsonlite) ## handle JSON
library(NLP)
library(ldatuning) ## find best number of topics
library(tm) ## create Corpus
library(SnowballC) ## convert to document term matrix
library(syuzhet) ## Sentiment Analysis
library(ggthemes) ## theme for ggplot
library(topicmodels) ## Topic Modeling
library(tidytext) ## Topic Modeling
library(glmnet) ## Lasso
library(caTools) ## smaple.split
library(xgboost) ## xgboost
set.seed(20190118)





##################################################################
############## Read the original downloaded dataset ##############
##################################################################
#### Read the six csv files 
links <- fread("links.csv")
top.cast <- fread("Top 1000 Actors and Actresses.csv")
top.cast$rank <- c(rep(1,100),rep(2,100),rep(3,800))




##################################################################
############## Web scraping from www.themoviedb.org ##############
##################################################################
#### Create a function to parse "www.themoviedb.org" API # tmdb.id <- "50091"
parse.tmdb <- function(tmdb.id,url){
  ## path
  path <- paste0("3/movie/",tmdb.id,"?api_key=b73ee4ad980aac51ec7a18742b8887a3&append_to_response=credits,keywords,reviews")
  ## Executing an API call with the GET flavor
  raw.result <- GET(url = url, path = path)
  ## Exam the status code
  if (raw.result$status_code == 200){
    ## Translate to readable text format
    raw.content <- rawToChar(raw.result$content)
    ## Parse into Json file
    content <- fromJSON(raw.content)
    ## Basic information
    content1 <- content[c(1:2,4,6:13,16:18,20:25)]
    ## Collections
    collection <- content$belongs_to_collection
    if (length(collection) != 0) {
      content1 <- c(content1, collection = paste(collection$name,collapse  = "|"))
    } else {content1 <- c(content1, collection = NA)}
    ## Genres
    genre <- content$genres
    if (length(genre) != 0) {
      content1 <- c(content1, genre = paste(genre$name,collapse  = "|"))
    } else {content1 <- c(content1, genre = NA)}
    ## Production companies
    prod.co <- content$production_companies
    if (length(prod.co) != 0) {
      content1 <- c(content1, prod.co.name = paste(prod.co$name,collapse  = "|"), prod.co.country = paste(unique(prod.co$origin_country)[!unique(prod.co$origin_country) %in% ""],collapse  = "|"))
    } else {content1 <- c(content1, prod.co.name = NA, prod.co.country = NA)}
    ## Production countries
    prod.country <- content$production_countries
    if (length(prod.country) != 0) {
      content1 <- c(content1, prod.country = paste(prod.country$iso_3166_1,collapse  = "|"))
    } else {content1 <- c(content1, prod.country = NA)}
    ## Spoken languages
    spoken.lang <- content$spoken_languages
    if (length(spoken.lang) != 0) {
      content1 <- c(content1, spoken.lang = paste(spoken.lang$iso_639_1,collapse  = "|"))
    } else {content1 <- c(content1, spoken.lang = NA)}
    ## Casts
    cast <- content$credits$cast
    if (length(cast) != 0) {
      cast <- as.data.table(cast[1:3,][,'name']) 
      names(cast) <- 'Name'
      cast <- merge(cast, top.cast, by = 'Name', all.x = TRUE)[,c('Name','rank')]
      cast[,rank := ifelse(is.na(rank),4,rank)]
      cast1 <- dim(cast[which(rank == 1),])[1]
      cast2 <- dim(cast[which(rank == 2),])[1]
      cast3 <- dim(cast[which(rank == 3),])[1]
      cast4 <- dim(cast[which(rank == 4),])[1]
      content1 <- c(content1, cast1 = cast1, cast2 = cast2, cast3 = cast3, cast4 = cast4)
    } else {content1 <- c(content1, cast1 = NA, cast2 = NA, cast3 = NA, cast4 = NA)}
    ## Crews
    crew <- content$credits$crew
    if (length(crew) != 0) {
      content1 <- c(content1, 
                    director = paste(crew$name[which(crew$job == "Director")],collapse  = "|"), 
                    screenplay = paste(crew$name[which(crew$job == "Screenplay" | crew$job == "Writer")],collapse  = "|"),
                    story = paste(crew$name[which(crew$job == "Story")],collapse  = "|"))
    } else {content1 <- c(content1, director = NA, screenplay = NA, story = NA)}
    ## Keywords
    keywords <- content$keywords$keywords
    if (length(keywords) != 0) {
      content1 <- c(content1, keyword = paste(keywords$name,collapse  = "|"))
    } else {content1 <- c(content1, keyword = NA)}
    ## Reviews
    reviews <- content$reviews$results
    if (length(reviews) != 0) {
      content1 <- c(content1, review = paste(reviews$content,collapse  = "|"))
    } else {content1 <- c(content1, review = NA)}
    content1[sapply(content1, is.null)] <- NA
  } else {
    content1 <- NULL
  }
  return(content1)
}

#### Run for loop to parse all movies
start_time <- Sys.time()

tmdb <- data.frame()
tmdb.na <- c()
url  <- "https://api.themoviedb.org"
for (tmdbid in links$tmdbId){
  tmdb_ <- parse.tmdb(tmdbid,url)
  if (is.null(tmdb_)){
    tmdb.na <- append(tmdb.na,tmdbid)
  } else {
    tmdb = rbind(tmdb, data.frame(tmdb_,stringsAsFactors = FALSE))
  }
}

end_time <- Sys.time()
end_time - start_time 

remove(tmdb_,url,links)
remove(end_time,start_time)





###############################################################
############## Format www.themoviedb.org dataset ##############
###############################################################
#### Format "tmdb" table
tmdb1 <- tmdb %>% distinct() %>% as.data.table()

#### Format "tmdb" table: Remove movies that are not released 
tmdb1 <- tmdb1[which(tmdb1$status == 'Released'),]

#### Format "tmdb" table: Remove movies that were released the forth quater of 2019 
tmdb1 <- tmdb1[which(tmdb1$release_date < '2019-10-01'),]

#### Format "tmdb" table: Remove movies that have no revenue information
tmdb1 <- tmdb1[which(revenue > 0),] ## & budget > 0

#### Format "tmdb" table: Remove movies that have no cast information
tmdb1 <- tmdb1[which(!is.na(tmdb1$cast1)),]

#### Format "tmdb" table: Create binery variables for "homepage" and "collection"
tmdb1[, homepage := ifelse(homepage %like% 'http', 1, 0)]
tmdb1[, collection := ifelse(map(collection, is.na),0, 1)]

#### Format "tmdb" table: Convert "adult" and "video" from bool to binary
tmdb1[, adult := ifelse(adult == TRUE,1,0)]
tmdb1[, video := ifelse(video == TRUE,0,1)]

#### Format "tmdb" table: Convert "release date" to season
tmdb1$release_month <- month(as.Date(tmdb1$release_date))
tmdb1 <- tmdb1[which(!is.na(tmdb1$release_month)),]

####Format "tmdb" table:  Create Dummy variables for genres 
tmdb1 <- cSplit_e(tmdb1, "genre", "|", type = "character", fill = 0, drop = T)

#### Fill the missing runtime with average runtime
tmdb1[,runtime := ifelse(map(runtime, is.na),mean(tmdb1[which(!is.na(runtime)),]$runtime),runtime)]

#### Format "tmdb" table: Drop irrelevant columns for "tmdb"
tmdb1[, c("adult","backdrop_path","imdb_id","original_language","original_title","popularity","poster_path", ##"budget",
          "release_date","status","tagline","title","vote_average","prod.co.name","prod.co.country",##"revenue",
          "prod.country","spoken.lang","director","screenplay","story") := NULL]

#### Correlation analysis
cor <- cor(tmdb1[,c(-3,-4,-5,-14,-15)])
cor.col <- findCorrelation(cor,cutoff = 0.7,verbose = FALSE,names = TRUE,exact = FALSE)
tmdb1[, (cor.col):=NULL]
rm(cor,cor.col)

#### Format "tmdb" table: group genres that have less than 5% appereance into others
genres <- tmdb1[,genre_Action:genre_Western]
genres.var <-setdiff(names(tmdb1), names(genres))
tmdb1 <- tmdb1[,..genres.var]
genres.colS <- colSums(as.matrix(genres))
genres.colS <- data.table(name = attributes(genres.colS)$names, count = genres.colS)
nOthers <- dim(tmdb1)[1]*0.05
genres.var <- c(genres.colS[count<nOthers]$name)
genres[, genres_Others := ifelse(rowSums(genres[, ..genres.var]) > 0, 1, 0)]
genres <- genres[, (genres.var):=NULL]
tmdb1 <- cbind(tmdb1,genres)
rm(genres,genres.colS,genres.var)


#### Format "tmdb" table: Rename columns by adding "tmdb"
names(tmdb1) <- paste0("tmdb.",names(tmdb1))
names(tmdb1)[3] <- "tmdbId"




###########################################################
############## Prepare the data for modeling ##############
###########################################################

#### Create table 
movies1 <- links[,c(1,3)]
movies1 <- merge(movies1, tmdb1, by = "tmdbId", all.y = TRUE)
movies1 <- movies1[!which(is.na(movieId))]
remove(links,top.cast)

################ Feature Engineering ################
## fill empty values in the list for "keyword"
movies1[, tmdb.keyword := ifelse(map(tmdb.keyword, is.na),"aempty",tmdb.keyword)]
## count number of "keywords" per movie
movies1$tmdb.keyword <- as.list(strsplit(movies1$tmdb.keyword, '\\|'))
movies1[,tmdb.keyword_count := unlist(lapply(tmdb.keyword, length))]
## count number of words in "overview"
movies1[,tmdb.overview_word_count := str_count(tmdb.overview,pattern = "\\w+")]
## count total length of "overview"
movies1[,tmdb.overview_len := str_count(tmdb.overview)]
# ##similarity between $$
# movies1[,lev_sim := levenshteinDist(budget,revenue)]
# ## remove movies that have no overviews
# movies1 <- movies1[which(movies1$tmdb.overview_word_count > 0),]
dim(movies1) ## genome+rating+tags+tmdb+count (kw+overview)


################ Analysis for "Overview" ################
tmdb.overview <- movies1$tmdb.overview
#### "Overview": Emotion Analysis
overview.e <-get_nrc_sentiment(tmdb.overview)
names(overview.e) <- paste0("overview.",names(overview.e))
movies2 <- cbind(movies1, overview.e) ## genome+rating+tags+tmdb+count (kw+overview)+emotion
rm(tmdb.overview,overview.e)

# #### "Overview": Sentiment Analysis
# overview.syuzhet <- get_sentiment(tmdb.overview, method="syuzhet")
# overview.bing <- get_sentiment(tmdb.overview, method="bing")
# overview.afinn <- get_sentiment(tmdb.overview, method="afinn")
# overview.nrc <- get_sentiment(tmdb.overview, method="nrc") 
# movies3 <- cbind(movies2, overview.nrc) ## genome+rating+tags+tmdb+count (kw+overview)+emotion+sentiment
# # rm(overview.e, overview.nrc)
movies3 <- movies2

######## "Keywords": Transform into one keyword per row format ########
## extract variables from keyword
movies.k <- data.table(movieId = rep(unlist(movies3$movieId), lapply(movies3$tmdb.keyword, length)), tmdb.keyword = unlist(movies3$tmdb.keyword))
head(movies.k)
## convert keyword to lower
movies.k[, tmdb.keyword := unlist(lapply(tmdb.keyword, tolower))]
## calculate count for every feature
movies.k[,count := .N, tmdb.keyword]
## keep keywords which occur 10 or more times
nKappearance <- dim(movies3)[1]*0.01
movies.k <- movies.k[count >= nKappearance]
## Convert each keyword into a separate column to use those as variables in model training. 
movies.k <- dcast(data = movies.k, formula = movieId ~ tmdb.keyword, fun.aggregate = length, value.var = "tmdb.keyword")
names(movies.k)[-1] <- paste0("kw.",names(movies.k[,-1]))
dim(movies.k)
## Merge with existing table
movies4 <- merge(movies3, movies.k, by = "movieId")#, all.y = TRUE) ## genome+rating+tags+tmdb+count (kw+overview)+emotion+sentiment+kw
rm(movies.k)


######## "Overview": Topic Modeling ########
overview_corpus <- Corpus(VectorSource(movies4$tmdb.overview))
## clean Corpus
overview_corpus <- tm_map(overview_corpus, tolower) #change to lower case
overview_corpus <- tm_map(overview_corpus, removePunctuation) #remove punctuation
overview_corpus <- tm_map(overview_corpus, removeNumbers) #remove numbers
overview_corpus <- tm_map(overview_corpus, removeWords, c(stopwords('english'))) #remove stopwords
overview_corpus <- tm_map(overview_corpus, stemDocument,language = "english") #perform stemming
dropword <- c("find","new","one","get","becom","take","two","year","will","must",
              "make","can","set","soon","day","three","come")
overview_corpus <- tm_map(overview_corpus,removeWords,dropword) #remove irrelevant words
overview_corpus <- tm_map(overview_corpus, stripWhitespace) #remove whitespaces
overview_corpus_dtm <- DocumentTermMatrix(overview_corpus)
overview_corpus_dtm_colS <- colSums(as.matrix(overview_corpus_dtm))
overview_corpus_dtm_doc_features <- data.table(name = attributes(overview_corpus_dtm_colS)$names, count = overview_corpus_dtm_colS)
## Graph the frequency of the words
overview_corpus_dtm_doc_features[count>250] %>%
  mutate(name = fct_reorder(name, count)) %>%
  ggplot(aes(name, count)) +
  geom_bar(stat = "identity",fill='lightblue',color='black')+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_economist()+
  scale_color_economist()+
  geom_bar(stat = "identity") +
  xlab(NULL) +
  coord_flip()
rm(overview_corpus_dtm_colS,overview_corpus_dtm_doc_features)
## Remove terms with more than 0.97 sparsity
overview_corpus_dtm_new <- removeSparseTerms(overview_corpus_dtm,sparse = 0.97)
overview_corpus_dtm_new <-as.matrix(overview_corpus_dtm_new)
## Remove Document with now term
rowTotals <- apply(overview_corpus_dtm_new , 1, sum) #Find the sum of words in each Document
overview_corpus_dtm_new   <- overview_corpus_dtm_new[rowTotals > 0, ]
## Find the best number of topics
result <- FindTopicsNumber(
  overview_corpus_dtm_new,
  topics = seq(from = 10, to = 90, by = 10),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 2L,
  verbose = TRUE
)
## Visualize to help find the best number of potics
FindTopicsNumber_plot(result)
## Run the LDA model
lda <- LDA(overview_corpus_dtm_new, k = 40) # k is the number of topics to be found.
lda_td <- tidy(lda)
## Visualization to find the 10 terms that are most common within each topic
top_terms <- lda_td %>% group_by(topic) %>% top_n(10, beta) %>% ungroup() %>% arrange(topic, -beta)
top_terms %>% filter(topic >=1 & topic <=6) %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  scale_x_reordered() +
  ggtitle("First 6 topics for Overview")+
  theme(plot.title = element_text(hjust = 0.5))
## Document-topic Prob
documents <- tidy(lda, matrix = "gamma")
documents <- documents %>% as.data.table()
documents <- reshape(documents, timevar="topic",idvar="document",direction="wide")
## Create final model for modeling
movies4 <- tibble::rownames_to_column(movies4, "document")
movies5 <- merge(movies4,documents,by = "document")
rm(overview_corpus,overview_corpus_dtm,overview_corpus_dtm_new,result,dropword,lda,lda_td)
movies5 <- movies4

##############################################################################
############## Quantify Review Column ##############
##############################################################################
bingpositive <- get_sentiments("bing") %>% 
  filter(sentiment == "positive")

review_words <- movies5 %>%
  unnest_tokens(word, tmdb.review) %>%
  filter(!word %in% stop_words$word,
         str_detect(word, "^[a-z']+$"))

wordcounts <- review_words %>%
  group_by(movieId) %>%
  summarize(words = n())

review_words_count <- review_words %>%
  semi_join(bingpositive) %>%
  group_by(movieId) %>%
  summarize(postivivewords = n()) %>%
  left_join(wordcounts, by = c('movieId')) %>%
  mutate(ratio = postivivewords/words) %>%
  ungroup()

movies6 <- merge(movies5, review_words_count, by = 'movieId', all.x = TRUE)

## count number of reviews
movies6$tmdb.review <- as.list(strsplit(movies6$tmdb.review, '\\|'))
movies6[,tmdb.review_count := ifelse(is.na(tmdb.review),0,unlist(lapply(tmdb.review, length)))]
movies6[,c("postivivewords","words") := NULL]
movies6[,ratio := ifelse(is.na(ratio),0,ratio)]
names(movies6) <- make.names(names(movies6))
names(movies6) <- make.unique(names(movies6), sep="_")
rm(bingpositive,review_words,wordcounts,review_words_count)









################################################
############## Fitting the models ##############
################################################


r2 <- data.frame(model.name=rep(NA,0), r2=rep(NA,0), num.var=rep(NA,0))

## dataset for modeling
movies.ke <- movies6 %>% select(-c(tmdbId,tmdb.overview,tmdb.keyword,tmdb.review,document))
movies.ke$tmdb.release_month<-as.factor(movies.ke$tmdb.release_month)
## split the data set into train and test
smp_size <- floor(0.75 * nrow(movies.ke))
train_ind <- sample(seq_len(nrow(movies.ke)), size = smp_size)
train <- movies.ke[train_ind, ]
test <- movies.ke[-train_ind, ]


#### Linear Regression: all included model
## fitting the model
model_all<-glm(tmdb.revenue ~.-movieId, data = train)
summary(model_all)
##significant test
alpha <- 0.05
TotalN <- sum(!is.na(summary(model_all)$coefficients[,4]))
sum(summary(model_all)$coefficients[,4] < alpha/TotalN)/TotalN
sum(summary(model_all)$coefficients[,4] < alpha)/TotalN
data.frame(summary(model_all)$coefficients[,c(1,4)])[summary(model_all)$coefficients[,4] < alpha/TotalN,]
data.frame(summary(model_all)$coefficients[,c(1,4)])[summary(model_all)$coefficients[,4] < alpha,]
var.to.keep <- sum(summary(model_all)$coefficients[,4] < alpha/TotalN) - 1
var.to.keep
predlm  <- predict(model_all, newdata=test, type="response")
r.sq <- R2(y=test$tmdb.revenue, pred=predlm)
r2[nrow(r2) + 1,] = c("Liner Regression",round(r.sq,4),var.to.keep)


#### Lasso
movies.ke<- movies.ke[complete.cases(movies.ke),]
Mx_lm <- model.matrix(tmdb.revenue ~ ., data=movies.ke[,-1])[,-1]
My_lm <- movies.ke$tmdb.revenue
lasso_lm <- glmnet(Mx_lm,My_lm)
lassoCV_lm <- cv.glmnet(Mx_lm,My_lm)
## min cvm & ise cvm
features.min.lm <- support(lasso_lm$beta[,which.min(lassoCV_lm$cvm)])
length(features.min.lm)
features.1se.lm <- support(lasso_lm$beta[,which.min((lassoCV_lm$lambda-lassoCV_lm$lambda.1se)^2)])
length(features.1se.lm) 
## Plot the cross validation of Lasso
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(lassoCV_lm, main="Fitting Graph for CV Lasso \n # of non-zero coefficients  ", 
     xlab = expression(paste("log(",lambda,")")))
## new dataset
# data.min.lm <- data.frame(Mx_lm[,features.min.lm],My_lm)
data.1se.lm <- data.frame(Mx_lm[,features.1se.lm],My_lm)
## Fitting
lasso1se_lm  <- glmnet(Mx_lm[train_ind,],My_lm[train_ind],lambda = lassoCV_lm$lambda.1se)
predlasso1se_lm  <- predict(lasso1se_lm, newx=Mx_lm[-train_ind,], type="response")
r.sq <- R2(y=My_lm[-train_ind], pred=predlasso1se_lm)
r2[nrow(r2) + 1,] = c("Lasso",round(r.sq,4),length(features.1se.lm))







#### Post Lasso
r1se_lm <- glm(My_lm ~ ., data=data.1se.lm[train_ind,])
pred1se_lm  <- predict(r1se_lm, newdata=data.1se.lm[-train_ind,], type="response")
r.sq <- R2(y=My_lm[-train_ind], pred=pred1se_lm)
r2[nrow(r2) + 1,] = c("Post Lasso",round(r.sq,4),dim(data.1se.lm)[2]-1)







#### Random Forest
train_rf <- train[,-1]
train_rf$tmdb.release_month <- as.integer(train_rf$tmdb.release_month)
rf <- randomForest(tmdb.revenue ~., data=train_rf[,-1], nodesize=5, ntree = 500, mtry = 5)
plot(rf, main = 'Error for Random Forest with 500 trees')
# It is overfitting with 500 trees as error went up after the 100th tree. 
rf <- randomForest(tmdb.revenue ~., data=train_rf[,-1], nodesize=5, ntree = 100, mtry = 5)
test$tmdb.release_month <- as.integer(test$tmdb.release_month)
predlm  <- predict(rf, newdata=test, type="response")
r.sq <- R2(y=test$tmdb.revenue, pred=predlm)
r2[nrow(r2) + 1,] = c("Random Forest",round(r.sq,4),rf$ntree)





#### Gradient Boost
error <- data.frame(version = rep(NA,0), model = rep(NA,0), max_depth = rep(NA,0), nrounds = rep(NA,0), rmse.val = rep(NA,0), rmse.train = rep(NA,0), best_ntreelimit = rep(NA,0), r2 = rep(NA,0))
train$tmdb.release_month<-as.integer(train$tmdb.release_month)
test$tmdb.release_month<-as.integer(test$tmdb.release_month)
## returns 70% indexes from train data
sp <- sample.split(Y = train$tmdb.revenue,SplitRatio = 0.75)
## create data for xgboost
xg_val <- train[sp]
id <- train$movieId
target <- train$tmdb.revenue
xg_val_target <- target[sp]

d_train <- xgb.DMatrix(data = as.matrix(train[,-c("movieId","tmdb.revenue"),with=F]),label = target)
d_val <- xgb.DMatrix(data = as.matrix(xg_val[,-c("movieId","tmdb.revenue"),with=F]), label = xg_val_target)
d_test <- xgb.DMatrix(data = as.matrix(test[,-c("movieId","tmdb.revenue"),with=F]))

watch <- list(val=d_val, train=d_train)

for (booster in c('gbtree','gblinear','dart')){
  for (max_depth in c(4,5,6)){
    param <- list(booster=booster, objective="reg:squarederror", eval_metric="rmse",
                  eta = .02, gamma = 1, max_depth = max_depth, min_child_weight = 1,
                  subsample = .7,  colsample_bytree = .7)
    for (nround in c(300,400,500)){
      xgb2cv  <- xgb.cv(params = param, data = d_train, nrounds = nround, watchlist=watch, 
                        print_every_n = 50, nfold = 5,early_stopping_rounds = 10)
      xgb2 <- xgb.train(params = param, data = d_train, nrounds = xgb2cv$best_ntreelimit, watchlist=watch, 
                        print_every_n = 50, early_stopping_rounds = 10) 
      xg_pred <- as.data.table(predict(xgb2, d_test))
      colnames(xg_pred) <- c("pred.revenue")
      xg_pred <- cbind(data.table(movieId = test$movieId, tmdb.revenue = test$tmdb.revenue),xg_pred)
      r2 <- R2(y=xg_pred$tmdb.revenue, pred=xg_pred$pred.revenue)
      print(paste(p, booster, max_depth, nround,xgb2$evaluation_log$train_rmse[xgb2cv$best_ntreelimit],xgb2cv$best_ntreelimit,r2))
      error[dim(error)[1]+1,] <- c(p, booster, max_depth, nround,xgb2$evaluation_log$val_rmse[xgb2cv$best_ntreelimit],
                                   xgb2$evaluation_log$train_rmse[xgb2cv$best_ntreelimit],xgb2cv$best_ntreelimit,round(r2,4))
    }
  }
}


error[which.max(error$r2),]
xgb2 <- xgb.train(params = list(booster="gbtree", objective="reg:squarederror", eval_metric="rmse",
                                eta = .02, gamma = 1, max_depth = 6, min_child_weight = 1,
                                subsample = .7,  colsample_bytree = .7), 
                  data = d_train, nrounds = 414, watchlist=watch, 
                  print_every_n = 50, early_stopping_rounds = 10) 
xg_pred <- as.data.table(predict(xgb2, d_test))
colnames(xg_pred) <- c("pred.revenue")
xg_pred <- cbind(data.table(tmdbID = test$movieId, tmdb.revenue = test$tmdb.revenue),xg_pred)
r.sq <- R2(y=test$tmdb.revenue, pred=xg_pred$pred.revenue)
r2[nrow(r2) + 1,] = c("XGBoost",round(r.sq,4),424)


xgb.importance(model = xgb2)
library(DiagrammeR)
xgb.plot.importance(xgb.importance(model = xgb2),top_n = 10,measure = 'Frequency')
xgb.plot.importance(xgb.importance(model = xgb2),top_n = 10,measure = 'Gain')
xgb.plot.importance(xgb.importance(model = xgb2),top_n = 10,measure = 'Cover')


paste0("Dataset Sparsity: ",round(sum(movies.ke == 0)/(dim(movies.ke)[1]*dim(movies.ke)[2]),4))




#### comparison between the 3 models
movies.ke <- movies6 %>% select(-c(document,tmdbId,tmdb.overview,tmdb.keyword,tmdb.review)) ##
movies.ke$tmdb.release_month<-as.factor(movies.ke$tmdb.release_month)

min_lambda <- lassoCV_lm$lambda.min
se1_lambda <- lassoCV_lm$lambda.1se
lassobeta <- lasso_lm$beta
features.min <- list()
lengthmin <- data.frame(betamin=rep(NA,length(lassobeta)))
features.1se <- list()
length1se <- data.frame(beta1se=rep(NA,length(lassobeta)))

data.min <- list()
data.1se <- list()

num.features <- ncol(Mx_lm)

features.min <- support(lasso_lm$beta[,which.min(lassoCV_lm$cvm)])
length(features.min)
data.min <- data.frame(Mx_lm[,features.min],My_lm)

#### K-fold validation
nfold<- 10
n<-nrow(Mx_lm)
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]
##post lassso/lasso/ random forest/ XGBoost model
OOS <- data.frame(m.lr=rep(NA,nfold), m.lr.l=rep(NA,nfold), m.lr.pl=rep(NA,nfold),m.rf=rep(NA,nfold),m.xgb=rep(NA,nfold), m.average=rep(NA,nfold)) 
data<-movies.ke

for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ### Logistic regression
  m.lr <-glm(tmdb.revenue~., data=movies.ke, subset=train)
  pred.lr <- predict(m.lr, newdata=data[-train,], type="response")
  OOS$m.lr[k] <- R2(y=data[-train]$tmdb.revenue, pred=pred.lr)
  
  ### the Post Lasso Estimates
  m.lr.pl <- glm(My_lm~., data=data.min, subset=train)
  pred.lr.pl <- predict(m.lr.pl, newdata=data.min[-train,], type="response")
  OOS$m.lr.pl[k] <- R2(y=data[-train]$tmdb.revenue, pred=pred.lr.pl)
  
  ### the Lasso estimates  
  m.lr.l  <- glmnet(Mx_lm[train,],My_lm[train],lambda = lassoCV_lm$lambda.min)
  pred.lr.l <- predict(m.lr.l, newx=Mx_lm[-train,], type="response")
  OOS$m.lr.l[k] <- R2(y=data[-train]$tmdb.revenue, pred=pred.lr.l)
  
  ### the Random Forest estimates  
  m.rf  <- randomForest(tmdb.revenue ~., data=movies.ke[train,][,-1], nodesize=5, ntree = 100, mtry = 5)
  pred.rf <- predict(m.rf, newdata=data[-train,], type="response")
  OOS$m.rf[k] <- R2(y=data[-train]$tmdb.revenue, pred=pred.rf)
  
  ### the XGBoost estimates  
  ## create data for xgboost
  xgb_train <- movies.ke[train,]
  xgb_train$tmdb.release_month <- as.integer(xgb_train$tmdb.release_month)
  xgb_test <- movies.ke[-train,]
  xgb_test$tmdb.release_month <- as.integer(xgb_test$tmdb.release_month)
  
  sp <- sample.split(Y = xgb_train$tmdb.revenue,SplitRatio = 0.75)
  
  id <- xgb_train$movieId
  target <- xgb_train$tmdb.revenue
  xg_val <- xgb_train[sp]
  xg_val_target <- target[sp]
  
  d_train <- xgb.DMatrix(data = as.matrix(xgb_train[,-c("movieId","tmdb.revenue"),with=F]),label = target)
  d_val <- xgb.DMatrix(data = as.matrix(xg_val[,-c("movieId","tmdb.revenue"),with=F]), label = xg_val_target)
  d_test <- xgb.DMatrix(data = as.matrix(xgb_test[,-c("movieId","tmdb.revenue"),with=F]))
  
  watch <- list(val=d_val, train=d_train)
  
  m.xgb  <- xgb.train(params = list(booster="gbtree", objective="reg:squarederror", eval_metric="rmse",
                                    eta = .02, gamma = 1, max_depth = 6, min_child_weight = 1,
                                    subsample = .7,  colsample_bytree = .7), 
                      data = d_train, nrounds = 414, watchlist=watch, 
                      print_every_n = 50, early_stopping_rounds = 10) 
  pred.xgb <- predict(m.xgb, newdata=d_test, type="response")
  OOS$m.xgb[k] <- R2(y=xgb_test$tmdb.revenue, pred=pred.xgb)

  ### null model estimates
  pred.m.average <- rowMeans(cbind( pred.lr.l, pred.lr.pl, pred.lr, pred.lr))
  OOS$m.average[k] <- 1-sum((pred.lr-data[-train]$tmdb.revenue)^2)/sum((mean(data[-train]$tmdb.revenue)-data[-train]$tmdb.revenue)^2)

  print(paste("Iteration",k,"of",nfold,"completed"))
  
  
}

par(mar=c(6.0,6.0,6.0,6.0))
bp <- barplot(round(colMeans(OOS),4), las=2, xpd=FALSE , xlab="Models",  
        ylim=c(0.975*min(colMeans(OOS)),1.05*max(colMeans(OOS))), ylab = "OOS R2",
        main="Performance: Out of Sample" ~ R^2)
text(x=bp,y=round(colMeans(OOS),4)+0.015, labels=as.character(round(colMeans(OOS),4)))

rm(data)









#### Random Forest
train_rf <- train
names(train_rf) <- make.names(names(train_rf))
names(train_rf) <- make.unique(names(train_rf), sep="_")
rf <- randomForest(rating ~., data=train_rf, nodesize=5, ntree = 500, mtry = 5)
plot(rf, main = 'Error for Random Forest with 500 trees')
# It is overfitting with 500 trees as error went up after the 100th tree. 
# We rerun the model with 100 trees
rf <- randomForest(Absenteeism.time.in.hours ~., data=abs_rf, nodesize=5, ntree = 100, mtry = 4)
plot(rf, main = 'Error for Random Forest with 100 trees')
