from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col, lit

from LogisticRegression import *
from SVM import *
from NeuralNetwork import *


start_time = time.time()
spark = SparkSession.builder.appName('secondproject').getOrCreate()

#Creo dataframe contenente il datatset dai CSV
data = spark.read.csv('/Users/nicoanie/Desktop/2Project/heartbeat/', header=False, sep=",")

#Input in HDFS
# data = spark.read.csv(
#     "hdfs://localhost:9000/user/input/",
#     header=False,
#     sep=",")

#Converto i valori da scientific notation in float
data = data.select(*(col(c).cast("float").alias(c) for c in data.columns))

#Creo dense vector e lo converto in dataframe
transform = data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'label'])

# Split dataset Hold-Out (90-10)
(trainingData, testData) = transform.randomSplit([0.9, 0.1])


#Richiamo la Regressione Multi Logistica
f1_multi_logistic, cnf_matrix, MultiLogisticModel = MultiLogisticRegression(trainingData, testData)

#Richiamo SVM
f1_svm, cnf_matrix_svm, svm_model = SVM(trainingData, testData)

#Richiamo NN
f1_nn, cnf_matrix_nn, nn_model = NeuralNetwork(trainingData, testData)

print(" ")
print(" ")
print("Total Tuning Execution TIME:", time.time() - start_time)

#Addestro il modello che ha ottenuto la miglior accuracy
if(f1_multi_logistic > f1_svm and f1_multi_logistic > f1_nn):
    print("MODELLO SCELTO: Multi Logistic Regression")

    #Addestro il modello della regressione logistica multivariata sul dataset completo
    model = MultiLogisticModel.fit(transform)
elif(f1_svm > f1_nn):
    print("MODELLO SCELTO: SVM")

    #Addestro il modello svm sul dataset completo
    model = svm_model.fit(transform)
else:
    print("MODELLO SCELTO: Neural Network")

    # Addestro la rete neurale sul dataset completo
    model = nn_model.fit(transform)

#-------------------------------------------------- Spark Streaming ---------------------------------------------------#
#Termino la sessione di spark in quanto bisogna avviare quella di SparkStreaming
spark.stop()

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row

#Funzione per creare una Sessione Spark
def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']

#Funzione per convertire lo Streaming dei dati in Dataframe
def process(time, rdd):
    print("========= %s =========" % str(time))
    try:
        # Creo una singleton instanza della SparkSession
        spark = getSparkSessionInstance(rdd.context.getConf())

        # Converto RDD[String] in RDD[Row] e successivamente in DataFrame
        rowRdd = rdd.map(lambda w: Row(word=w)).toDF(['features'])

        #Effettuo la predizione dei dati ricevuti tramite SparkStreaming
        prediction = model.transform(rowRdd).select(['prediction']).show()
    except:
        pass

# Creo un StreamingContext locale con due thread di lavoro e intervallo batch di 1 secondo
sc = SparkContext("local[12]", "NetworkWordCount")
ssc = StreamingContext(sc, 1)

# Creo uno StremingData che si connetterà a localhost:9999
lines = ssc.socketTextStream("localhost", 9999)
# Effettuo lo split e creo un dense vector
lines = lines.flatMap(lambda line: [Vectors.dense(line.split(","))])
# Per ogni RDD viene avviata la funzione process
lines = lines.foreachRDD(process)

#Avvio e termino la sessione di SparkStreaming
ssc.start()
ssc.awaitTerminationOrTimeout(50)()
#----------------------------------------------------------------------------------------------------------------------#
