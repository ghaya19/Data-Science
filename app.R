setwd("D:/Files/Documents/UM/Masters/SEM1/PRINCIPLE OF DATA SCIENCE/G.ass/app2")
# Define UI logic
ui <- fluidPage(
    # Application title
    titlePanel("Prediction of Lung Cancer Risk"),
    
    # Sidebar with a slider and numeric input  
    sidebarLayout(
      sidebarPanel(
        selectInput("GENDER", "GENDER: ", choices = list("M", "F")
        ),
        numericInput("AGE","Age :",min = min(data$AGE),max = max(data$AGE),value = as.integer(mean(data$AGE))
        ),
        numericInput("SMOKING","SMOKING HABIT :",min = min(data$SMOKING),max = max(data$SMOKING),value = as.integer(mean(data$SMOKING))
        ),
        numericInput("YELLOW_FINGERS","PRESENCE OF YELLOW FINGER :",min = min(data$YELLOW_FINGERS),max = max(data$YELLOW_FINGERS),value = as.integer(mean(data$YELLOW_FINGERS))
        ),
        numericInput("ANXIETY","PRESENCE OF ANXIETY :",min = min(data$ANXIETY),max = max(data$ANXIETY),value = as.integer(mean(data$ANXIETY))
        ),
        numericInput("PEER_PRESSURE","INFLUENCE OF ANY PEER PRESSURE :",min = min(data$PEER_PRESSURE),max = max(data$PEER_PRESSURE),value = as.integer(mean(data$PEER_PRESSURE))
        ),
        numericInput("CHRONIC.DISEASE","PRESENCE OF ANY CHRONIC DISEASE :",min = min(data$CHRONIC.DISEASE),max = max(data$CHRONIC.DISEASE),value = as.integer(mean(data$CHRONIC.DISEASE))
        ),
        numericInput("FATIGUE","PRESENCE OF ANY FATIGUE :",min = min(data$FATIGUE),max = max(data$FATIGUE),value = as.integer(mean(data$FATIGUE))
        ),
        numericInput("ALLERGY","PRESENCE OF ANY ALLERGY :",min = min(data$ALLERGY),max = max(data$ALLERGY),value = as.integer(mean(data$ALLERGY))
        ),
        numericInput("WHEEZING","PRESENCE OF ANY WHEEZING :",min = min(data$WHEEZING),max = max(data$WHEEZING),value = as.integer(mean(data$WHEEZING))
        ),
        numericInput("ALCOHOL.CONSUMING","ANY ALCOHOL CONSUMING HABIT :",min = min(data$ALCOHOL.CONSUMING),max = max(data$ALCOHOL.CONSUMING),value = as.integer(mean(data$ALCOHOL.CONSUMING))
        ),
        numericInput("COUGHING","PRESENCE OF ANY COUGH :",min = min(data$COUGHING),max = max(data$COUGHING),value = as.integer(mean(data$COUGHING))
        ),
        numericInput("SHORTNESS.OF.BREATH","PRESENCE OF ANY SHORTNESS OF BREATH :",min = min(data$SHORTNESS.OF.BREATH),max = max(data$SHORTNESS.OF.BREATH),value = as.integer(mean(data$SHORTNESS.OF.BREATH))
        ),
        numericInput("SWALLOWING.DIFFICULTY","PRESENCE OF ANY DIFFICULTY IN SWALLOWING :",min = min(data$SWALLOWING.DIFFICULTY),max = max(data$SWALLOWING.DIFFICULTY),value = as.integer(mean(data$SWALLOWING.DIFFICULTY))
        ),
        numericInput("CHEST.PAIN","PRESENCE OF ANY PAIN IN THE CHEST :",min = min(data$CHEST.PAIN),max = max(data$CHEST.PAIN),value = as.integer(mean(data$CHEST.PAIN))
        ),      
        submitButton('Predict')
      ),
    mainPanel(
      tabsetPanel(
        tabPanel('Introduction',
                 p(),
                 h4("Introduction tab explains the application features to ease the users."),
                 h4("This application is used to predict the lung cancer risk based on selected attributes that includes symptoms, factors that influenced and habits."),
                 h4("User can use the menu on the left to input their gender, age, symptoms, habits and so on."),
                 h4("The prediction tab shows the result of the lung cancer risk."),
                 h4("The prediction is done using four different machine learning models")),
        tabPanel('Prediction', plotOutput("PieChart"),tableOutput("observation"))))))
      

# Define server logic 
server <- function(input, output) {
    
    output$observation <- renderTable({
      mlinputs <- reactiveValuesToList(input)
      
      mlmodel <- "SUPPORT VECTOR MACHINE"
      pred <- as.character(predict(fit.svm, mlinputs))
      
      
      mlmodel <- append(mlmodel, "LINEAR DISCRIMINANT ANALYSIS")
      pred <-  append(pred,
                      as.character(predict(fit.lda, mlinputs)))      
     
      mlmodel <- append(mlmodel, "K NEAREST NEIGHBOURS")
      pred <-  append(pred,
                      as.character(predict(fit.knn, mlinputs)))
      
      mlmodel <- append(mlmodel, "GRADIENT BOOSTING MACHINE")
      pred <-  append(pred,
                      as.character(predict(fit.gbm, mlinputs)))
      
      prediction <- data.frame(mlmodel, pred)
      colnames(prediction)[1] <- "Model"
      colnames(prediction)[2] <- "Prediction"
      
    output$PieChart <- renderPlot({
        
      ggplot(prediction, aes(x=factor(1), fill = Prediction)) +
        geom_bar(width = 5 ) +
        coord_polar("y") +
        labs(title = "Lung cancer risk (Yes (1)/No(0))")
    })
      prediction
      
    })
}


shinyApp(ui, server)