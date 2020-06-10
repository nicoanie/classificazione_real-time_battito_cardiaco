from pyspark.ml.classification import OneVsRest, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import *
from sklearn.metrics import confusion_matrix
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
import time

def SVMCV(trainingData, testData):
    start_time = time.time()

    svm = LinearSVC()
    ovr = OneVsRest(classifier=svm)

    # Parametri su cui effettuare il tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(svm.regParam, [1, 0]) \
        .addGrid(svm.maxIter, [100, 1000]) \
        .build()

    cv = CrossValidator(estimator=ovr, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(),
                        numFolds=5)

    model = cv.fit(trainingData)

    prediction = model.transform(testData)

    result = prediction.select('features', 'label', 'prediction')

    # Calcolo accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1score = evaluator.evaluate(prediction)

    # Confusion Matrix
    class_temp = prediction.select("label").groupBy("label") \
        .count().sort('count', ascending=False).toPandas()
    class_temp = class_temp["label"].values.tolist()

    y_true = prediction.select("label")
    y_true = y_true.toPandas()

    y_pred = prediction.select("prediction")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)
    print("Accuracy K-Fodls: ", accuracy)
    print("F1-Score K-Fodls: ", f1score)
    print("Confusion Matrix: ")
    print(cnf_matrix)
    print("SVM K-Folds Execution TIME:", time.time() - start_time)
    return (f1score, cnf_matrix, cv)

def SVM(trainingData, testData):
    start_time = time.time()
    print(" ")
    print("--------------------- SUPPORT VECTOR MACHINE ---------------------")

    svm = LinearSVC()
    ovr = OneVsRest(classifier=svm)

    # Parametri su cui effettuare il tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(svm.regParam, [1, 0]) \
        .addGrid(svm.maxIter, [100, 1000]) \
        .build()

    # Tuning sui vari parametri per scegliere il modello migliore
    tvs = TrainValidationSplit(estimator=ovr,
                               estimatorParamMaps=paramGrid,
                               evaluator=MulticlassClassificationEvaluator(),
                               # Validation test: 80% traning, 20% validation.
                               trainRatio=0.8)

    model = tvs.fit(trainingData)

    prediction = model.transform(testData)

    result = prediction.select('features', 'label', 'prediction')

    # Calcolo accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(prediction)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1score = evaluator.evaluate(prediction)

    # Confusion Matrix
    class_temp = prediction.select("label").groupBy("label") \
        .count().sort('count', ascending=False).toPandas()
    class_temp = class_temp["label"].values.tolist()

    y_true = prediction.select("label")
    y_true = y_true.toPandas()

    y_pred = prediction.select("prediction")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)


    print("Accuracy Hold-Out: ", accuracy)
    print("F1-Score Hold-Out: ", f1score)
    print("")
    print("")
    print("Doc Parameters : [", model.explainParams(), "]")
    print("")
    print("")
    print("Confusion Matrix: ")
    print(cnf_matrix)
    print("SVM HoldOut Execution TIME:", time.time() - start_time)

    # Richiamo SVM che utilizza la validazione K-Folds
    f1score_cv, cnf_matrix_cv, cv = SVMCV(trainingData, testData)

    # Restituisco il modello migliore tra Hold Out e K-Folds
    if (f1score <= f1score_cv):
        return (f1score_cv, cnf_matrix_cv, cv)
    else:
        return (f1score, cnf_matrix, tvs)

