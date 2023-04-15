#Title: Predicting the diabetes readmission rate and number of medication taken
#Year: 1999-2008
#The purpose of dataset:
#The dataset shows the hospital admission of diabetic patients. The dataset also shows their race,gender, age and other important variables.

#Objective/ goal of processing this dataset? To predict the readmission rate of diabetes patients and the number of medication
#Questions:
#1. How much is the diabetes readmission rate?
#2. What is the expected the number of medication taken?
#3. What are the features that significantly influence the diabetes readmission rate and the number of medication in the prediction model?

#1. Load dataset
getwd()
setwd("D:\\Files\\Documents\\UM\\Masters\\SEM1\\PROGRAMMING FOR DATA SCIENCE\\Group proj")
diabetes <- read.csv(file = 'diabetic_data.csv', head=TRUE, sep=",")
diabetes

#2. Explore dataset:
#dimension:
#content
#structure
#summary
head(diabetes) #To view the head of the variables with first 5 observations
tail(diabetes) #To view the tail of the variables with last 5 observations.
View(diabetes) #To view in table form.
ncol(diabetes) #There are 50 columns
nrow(diabetes) #There 101766 rows
dim(diabetes) #There are 101766 observations and 50 variables.
str(diabetes) #There are 101766 observations and 50 variables. 
summary(diabetes)

library(dbplyr)
glimpse(diabetes)

#3. Finding the missing values
is.na(diabetes) 
which(is.na(diabetes))
colSums(is.na(diabetes))#It states that there are no NA in this dataset. However, there are observation labelled "?"

#Which section of the data do you need to tidy?
#4. Data cleansing: Remove non-useful or irrelevant columns

diabetes1 <- subset(diabetes,select=-c(encounter_id, patient_nbr, examide, citoglipton,weight, payer_code, medical_specialty)) 
str(diabetes1)
diabetes1 # There are 101766 observations with 43 variables

#5. Data cleansing: Remove "?" in each variables
#5.1 Variable : race
diabetes2 <- diabetes1[diabetes1$race != "?",] # Number of observations drops by 2273
diabetes2
summary(diabetes1$race) #There are 101766 observation for diabetes1$race
summary(diabetes2$race) #There are 99493 observations for diabetes2$race

#5.2 Variable:diagnosis_1
diabetes2 <- diabetes2[diabetes2$diag_1 != "?",] # Number of observations drops by 19
diabetes2
summary(diabetes$diag_1) #There are 101766 observation for diabetes$diag_1
summary(diabetes2$diag_1) #There are 99139 observations for diabetes2$diag_1

#5.3 Variable:diagnosis_2
diabetes2 <- diabetes2[diabetes2$diag_2 != "?",] # No of observations drops by 335 (99474-99139)
diabetes2
summary(diabetes$diag_2) #There are 101766 observation for diabetes$race
summary(diabetes2$diag_2) #There are 99474 observations for diabetes2$race

#5.4 Variable:diagnosis_3
diabetes2 <- diabetes2[diabetes2$diag_3 != "?",] # No of observations drops by 1086 (99139 -98053)
diabetes2
summary(diabetes$diag_3) #There are 101766 observation for diabetes$race
summary(diabetes2$diag_3) #There are 98053 observations for diabetes2$race

#After data cleaning, there are 3713 observation was removed. 
#The remaining number of observations are 98053.

#6. Binary representation of readmitted within 30 days
#"0" representation for NO readmission after 30 days
#"1" representation for readmission within 30 days

diabetes2$readmittedbin <- ifelse(diabetes2$readmitted == "<30",1,0) 
table(diabetes2$readmittedbin)
#86987 no. of patients readmit after 30 days or no readmission
#11066 no. of patients readmit within 30 days

#7.1 To identify class of the variables
library(dbplyr)
glimpse(diabetes2)
names(diabetes2[c(7:13,17)])
class(diabetes2$time_in_hospital)
class(diabetes2$num_lab_procedures)
names(diabetes2[c(1:6,14:16,18:44)])
class(diabetes2$diag_2)
class(diabetes2$readmitted)

#7.2 To make factor of levels
#cbind columns with numerical variables with diabetes2[c(7:13,17)] and categorical variables with diabetes2[c(1:6,14:16,18:44)]
diabetes3 <- cbind(diabetes2[c(7:13,17)], lapply(diabetes2[c(1:6,14:16,18:44)],factor))
head(diabetes3)

#Proceed to use diabetes3





