from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
import time

def NeuralNetworkCV(trainingData, testData):
    start_time = time.time()

    layers = [187, 8, 5]

    nn = MultilayerPerceptronClassifier(layers=layers)

    # Parametri su cui effettuare il tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(nn.stepSize, [1, 0.01]) \
        .addGrid(nn.maxIter, [100, 1000]) \
        .build()

    cv = CrossValidator(estimator=nn, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(),
                        numFolds=5)

    model = cv.fit(trainingData)

    prediction = model.transform(testData)
    predictionAndLabels = prediction.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictionAndLabels)
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1score = evaluator.evaluate(predictionAndLabels)

    # Confusion Matrix
    class_temp = prediction.select("label").groupBy("label") \
        .count().sort('count', ascending=False).toPandas()
    class_temp = class_temp["label"].values.tolist()

    y_true = prediction.select("label")
    y_true = y_true.toPandas()

    y_pred = prediction.select("prediction")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)
    print(cnf_matrix)
    print("Accuracy K-Folds: ", accuracy)
    print("F1-Score K-Folds: ", f1score)
    print("")
    print("")
    print("Doc Parameters : [", model.explainParams(), "]")
    print("")
    print("")
    print("Confusion Matrix: ")
    print(cnf_matrix)
    print("Neural Network K-Folds Execution TIME:", time.time() - start_time)
    return f1score, cnf_matrix, cv


def NeuralNetwork(trainingData, testData):
    start_time = time.time()
    print(" ")
    print("--------------------- NEURAL NETWORK ---------------------")

    layers = [187, 8, 5]

    nn = MultilayerPerceptronClassifier(layers=layers)

    # Parametri su cui effettuare il tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(nn.stepSize, [1, 0.01]) \
        .addGrid(nn.maxIter, [100, 1000]) \
        .build()

    # Tuning sui vari parametri per scegliere il modello migliore
    tvs = TrainValidationSplit(estimator=nn,
                               estimatorParamMaps=paramGrid,
                               evaluator=MulticlassClassificationEvaluator(),
                               # Validation test: 80% traning, 20% validation.
                               trainRatio=0.8)

    model = tvs.fit(trainingData)

    prediction = model.transform(testData)
    predictionAndLabels = prediction.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    accuracy = evaluator.evaluate(predictionAndLabels)
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1score = evaluator.evaluate(predictionAndLabels)

    # Confusion Matrix
    class_temp = prediction.select("label").groupBy("label") \
        .count().sort('count', ascending=False).toPandas()
    class_temp = class_temp["label"].values.tolist()

    y_true = prediction.select("label")
    y_true = y_true.toPandas()

    y_pred = prediction.select("prediction")
    y_pred = y_pred.toPandas()

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_temp)


    print("Accuracy Hold out: ", accuracy)
    print("F1-Score Hold out: ", f1score)
    print("")
    print("")
    print("Doc Parameters : [", model.explainParams(), "]")
    print("")
    print("")
    print("Confusion Matrix: ")
    print(cnf_matrix)
    print("Neural Network HoldOut Execution TIME:", time.time() - start_time)

    # Richiamo NN che utilizza la validazione K-Folds
    f1score_cv, cnf_matrix_cv, cv = NeuralNetworkCV(trainingData, testData)

    # Restituisco il modello migliore tra Hold Out e K-Folds
    if (f1score <= f1score_cv):
        return (f1score_cv, cnf_matrix_cv, cv)
    else:
        return (f1score, cnf_matrix, tvs)
