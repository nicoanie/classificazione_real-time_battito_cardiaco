from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import *
from sklearn.metrics import confusion_matrix
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
import time


def MultiLogisticRegressionCV(trainingData, testData):
    start_time = time.time()

    # Creo istanza per la regressione logistica (Vedere dalla documentazione i parametri per l'auto tuning)
    lr = LogisticRegression()

    # Parametri su cui effettuare il tuning:
    # regParam: parametro lambda utilizzato nel L2-Form(fitting del modello)
    # elasticNetParam: parametro alpha (Learning Rate)
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [1, 0]) \
        .addGrid(lr.elasticNetParam, [1, 0]) \
        .addGrid(lr.maxIter, [100, 1000]) \
        .build()

    cv = CrossValidator(estimator=lr,
                        estimatorParamMaps=paramGrid,
                        evaluator=MulticlassClassificationEvaluator(),
                        numFolds=5)

    ###################################### Addestro il modello #####################################
    model1 = cv.fit(trainingData)

    ####################################### Testo il modello #######################################
    prediction = model1.transform(testData)
    result = prediction.select('features', 'label', 'prediction')

    #################################### Valutazione del modello ###################################
    #Precision
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    f1score = evaluator.evaluate(prediction)

    #Confusion Matrix
    class_temp = prediction.select("label").groupBy("label")\
                            .count().sort('count', ascending=False).toPandas()
    class_temp = class_temp["label"].values.tolist()

    y_true = prediction.select("label")
    y_true = y_true.toPandas()

    y_pred = prediction.select("prediction")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)
    print("Accuracy K-Fodls: ", accuracy)
    print("F1-Score K-Fodls: ", accuracy)
    bestModel = model1.bestModel
    print("alpha : ", bestModel._java_obj.getElasticNetParam())
    print("beta : ", bestModel._java_obj.getRegParam())
    print("iteration : ", bestModel._java_obj.getMaxIter())
    print("Confusion Matrix: ")
    print(cnf_matrix)
    print("Multi Logisti Regression K-Folds Execution TIME:", time.time() - start_time)

    return (f1score, cnf_matrix, cv)

def MultiLogisticRegression(trainingData, testData):
    start_time = time.time()
    print(" ")
    print("--------------------- MULTI LOGISTIC REGRESSION ---------------------")
    #Creo istanza per la regressione logistica
    lr = LogisticRegression()

    #Parametri su cui effettuare il tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [1, 0]) \
        .addGrid(lr.elasticNetParam, [1, 0]) \
        .addGrid(lr.maxIter, [100, 1000]) \
        .build()

    #Tuning sui vari parametri per scegliere il modello migliore
    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=MulticlassClassificationEvaluator(),
                               #Validation test: 80% traning, 20% validation.
                               trainRatio=0.8)

    ###################################### Addestro il modello #####################################
    model1 = tvs.fit(trainingData)

    ####################################### Testo il modello #######################################
    prediction = model1.transform(testData)
    result = prediction.select('features', 'label', 'prediction')

    #################################### Valutazione del modello ###################################
    #Precision
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1")
    f1score = evaluator.evaluate(prediction)

    #Confusion Matrix
    class_temp = prediction.select("label").groupBy("label")\
                            .count().sort('count', ascending=False).toPandas()
    class_temp = class_temp["label"].values.tolist()

    y_true = prediction.select("label")
    y_true = y_true.toPandas()

    y_pred = prediction.select("prediction")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)


    bestModel = model1.bestModel
    print("Accuracy Hold-Out: ", accuracy)
    print("F1-Score Hold-Out: ", f1score)
    print("alpha : ", bestModel._java_obj.getElasticNetParam())
    print("beta : ", bestModel._java_obj.getRegParam())
    print("iteration : ", bestModel._java_obj.getMaxIter())
    print("Confusion Matrix: ")
    print(cnf_matrix)
    print("Multi Logisti Regression HoldOut Execution TIME:", time.time() - start_time)

    #Richiamo la Multi Logisti Regression che utilizza la validazione K-Folds
    f1score_cv, cnf_matrix_cv, cv = MultiLogisticRegressionCV(trainingData, testData)

    #Restituisco il modello migliore tra Hold Out e K-Folds
    if (f1score < f1score_cv):
        return (f1score_cv, cnf_matrix_cv, cv)
    else:
        return (f1score, cnf_matrix, tvs)


