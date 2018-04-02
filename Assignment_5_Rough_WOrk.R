
h2o.shutdown()

library(data.table)
library(magrittr)

getwd()

setwd("D:/Data World/Business Analytics/Macine Learning/Assignment 5")

data <- fread("no-show-data.csv")

# some data cleaning
data[, c("PatientId", "AppointmentID", "Neighbourhood") := NULL]
setnames(data, 
         c("No-show", 
           "Age", 
           "Gender",
           "ScheduledDay", 
           "AppointmentDay",
           "Scholarship",
           "Hipertension",
           "Diabetes",
           "Alcoholism",
           "Handcap",
           "SMS_received"), 
         c("no_show", 
           "age", 
           "gender", 
           "scheduled_day", 
           "appointment_day",
           "scholarship",
           "hypertension",
           "diabetes",
           "alcoholism",
           "handicap",
           "sms_received"))

# for binary prediction, the target variable must be a factor
data[, no_show := factor(no_show, levels = c("Yes", "No"))]
data[, handicap := ifelse(handicap > 0, 1, 0)]

# create new variables
data[, gender := factor(gender)]
data[, scholarship := factor(scholarship)]
data[, hypertension := factor(hypertension)]
data[, alcoholism := factor(alcoholism)]
data[, handicap := factor(handicap)]

data[, scheduled_day := as.Date(scheduled_day)]
data[, appointment_day := as.Date(appointment_day)]
data[, days_since_scheduled := as.integer(appointment_day - scheduled_day)]

# clean up a little bit
data <- data[age %between% c(0, 95)]
data <- data[days_since_scheduled > -1]
data[, c("scheduled_day", "appointment_day", "sms_received") := NULL]


library(h2o)
h2o.init()

data <- as.h2o(data)

data_split <- h2o.splitFrame(data, ratios = 0.5, seed = 123)
data_train <- data_split[[1]]
data_test <- data_split[[2]]

data_split <- h2o.splitFrame(data_test, ratios = 0.5, seed = 123)

data_val <- data_split[[1]]
data_test <- data_split[[2]]

y <- "no_show"
X <- setdiff(names(data_train), y)

glm_model <- h2o.glm(
  X, y,
  training_frame = data_train,
  family = "binomial",
  alpha = 1, 
  lambda_search = TRUE,
  seed = 123,
  nfolds = 5
)

print(h2o.auc(h2o.performance(glm_model, newdata = data_val)))

?h2o.deeplearning

dl_model_basic <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_basic, newdata = data_val)))
#0.7185214

dl_model_layers <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(2, 5),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_layers, newdata = data_val)))
#0.6847421

dl_model_activation <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  activation = "Tanh",
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_activation, newdata = data_val)))
#0.7188071

dl_model_activation2 <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  activation = "RectifierWithDropout",
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_activation2, newdata = data_val)))
#0.7154182

##Will use the default Activation

dl_model_dropout <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  activation = "RectifierWithDropout",
  input_dropout_ratio = 0.2,
  hidden_dropout_ratios = c(0.6, 0.7),
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_dropout, newdata = data_val)))
#0.6689904

#Will use default dropout ratios

dl_model_regularization <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  l2 = 0.001,
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_regularization, newdata = data_val)))

#For Lasso 02, 0.5, 07, AUC dropped below 0.5

#l2=0.0001, 0.7226817
#   l2 = 0.00001, 0.7186687
#l2 = 0.001,,  0.7214606

dl_model_stop <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  l2 = 0.0001,
  stopping_rounds = 5,
  stopping_tolerance = 0.002,
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_stop, newdata = data_val)))

#Stopping round =10,  0.7210128
#Stopping round =2, 0.7184971

dl_model_epochs <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  validation_frame = data_val,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5, 
  stopping_metric = "AUC",
  l2 = 0.0001,
  stopping_rounds = 5,
  epochs = 20,
  reproducible = TRUE
)

print(h2o.auc(h2o.performance(dl_model_epochs, newdata = data_val)))
#0.7232043

print(h2o.auc(h2o.performance(dl_model_epochs, newdata = data_test)))
#0.7175229

###############################################################################

glm_model <- h2o.glm(
  X, y,
  training_frame = data_train,
  family = "binomial",
  alpha = 1, 
  lambda_search = TRUE,
  seed = 123,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

print(h2o.auc(h2o.performance(glm_model, newdata = data_val)))
#0.6582649

gbm_model <- h2o.gbm(
  X, y,
  training_frame = data_train,
  ntrees = 200, 
  max_depth = 10, 
  learn_rate = 0.1, 
  seed = 123,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)

print(h2o.auc(h2o.performance(gbm_model, newdata = data_val)))
#0.7235518

?h2o.randomForest

rf_model <- h2o.randomForest(
  X, y,
  training_frame = data_train,
  ntrees = 100,
  seed = 123,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)

print(h2o.auc(h2o.performance(rf_model, newdata = data_val)))
#0.7242968

dl_model <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  hidden = c(32, 8),
  seed = 123,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

print(h2o.auc(h2o.performance(dl_model_basic, newdata = data_val)))

gbm_model <- h2o.gbm(
  X, y,
  training_frame = data_train,
  ntrees = 200, 
  max_depth = 10, 
  learn_rate = 0.1, 
  seed = 123,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)


ensemble_model <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  base_models = list(glm_model,
                     rf_model,
                     gbm_model,
                     dl_model))

# inspect test set correlations of scores
predictions <- data.table(
  "glm" = as.data.frame(h2o.predict(glm_model, newdata = data_val)$Y)$Y,
  "rf" = as.data.frame(h2o.predict(rf_model, newdata = data_val)$Y)$Y,
  "gbm" = as.data.frame(h2o.predict(gbm_model, newdata = data_val)$Y)$Y,
  "dl" = as.data.frame(h2o.predict(dl_model, newdata = data_val)$Y)$Y
)

ggcorr(predictions, label = TRUE, label_round = 2)

# validation set performances
print(h2o.auc(h2o.performance(glm_model, newdata = data_val)))
print(h2o.auc(h2o.performance(rf_model, newdata = data_val)))
print(h2o.auc(h2o.performance(gbm_model, newdata = data_val)))
print(h2o.auc(h2o.performance(deeplearning_model, newdata = data_val)))

##Other meta learners
ensemble_model_gbm <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  metalearner_algorithm = "gbm",
  base_models = list(glm_model, 
                     rf_model,
                     gbm_model,
                     dl_model))

print(h2o.auc(h2o.performance(ensemble_model_gbm, newdata = data_test)))