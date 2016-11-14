import org.apache.spark.{SparkConf, SparkContext, SparkFiles}
import org.apache.spark.SparkContext._
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.functions._

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

import org.apache.spark.ml.feature.{HashingTF, IDF}
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.WrappedArray

import org.apache.spark.mllib.linalg.{DenseVector, SparseVector, Vector, Vectors}

object ClassificationAdhoc {

  def appendFeature(sv: SparseVector, adhoc_feature1: Int, adhoc_feature2: Int) : Vector = {

      //case sv: SparseVector =>
      val inputValues = sv.values
      val inputIndices = sv.indices
      val inputValuesLength = inputValues.length
      val dim = sv.size

      var adhoc_features = Array(adhoc_feature1,adhoc_feature2)
      val addhoc_size = adhoc_features.length

      val outputValues = Array.ofDim[Double](inputValuesLength + addhoc_size)
      val outputIndices = Array.ofDim[Int](inputValuesLength + addhoc_size)

      System.arraycopy(inputValues, 0, outputValues, 0, inputValuesLength)
      System.arraycopy(inputIndices, 0, outputIndices, 0, inputValuesLength)

      for (i <- 1 to addhoc_size) {
        outputValues(inputValuesLength-1+i) = adhoc_features(i-1).toDouble
      }
      outputIndices(inputValuesLength) = dim

      Vectors.sparse(dim + addhoc_size, outputIndices, outputValues)
      //case _ => throw new IllegalArgumentException(s"Do not support vector type ${vector.getClass}")
       
  }

  //get type of var utility 
  def manOf[T: Manifest](t: T): Manifest[T] = manifest[T]

  def main(args: Array[String]) {

    val t0 = System.nanoTime()
    val conf = new SparkConf().setAppName("KaggleDato")
    val spark = new SparkContext(conf)
    val sqlContext = new SQLContext(spark)

    val PATH_TO_JSON = "file:///home/alexeys/MLmeetup2016/preprocess/html.avro"
    val PATH_TO_TRAIN_LABELS = "file:///home/alexeys/MLmeetup2016/preprocess/labels.avro"

    val train_label_df = sqlContext.read.format("com.databricks.spark.avro").load(PATH_TO_TRAIN_LABELS)
    val input_df = sqlContext.read.format("com.databricks.spark.avro").load(PATH_TO_JSON)
    //input_df.printSchema()
    //train_label_df.printSchema()
    //input_df.show()
    //print input_df.count()

    //Make DF with labels    
    var train_wlabels_df = input_df.join(train_label_df,"id")
    train_wlabels_df.repartition(col("label"))
    train_wlabels_df.explain
    train_wlabels_df.printSchema()
    train_wlabels_df.show() 

    //register UDFs
    sqlContext.udf.register("countfeat", (s: WrappedArray[String]) => s.length)
    def countfeat_udf = udf((s: WrappedArray[String]) => s.length)

    //create udf that returns length of the list, 
    val links_cnt_df = train_wlabels_df.withColumn("links_cnt",countfeat_udf(col("links")))
    links_cnt_df.printSchema()
    links_cnt_df.show()

    //image cnt features
    val images_cnt_df = links_cnt_df.withColumn("images_cnt",countfeat_udf(col("images")))
    images_cnt_df.printSchema()
    images_cnt_df.show()


    //tokenizer = Tokenizer(inputCol="text", outputCol="words")
    var tokenizer = new RegexTokenizer().setInputCol("text").setOutputCol("words").setPattern("\\W")
    val tokenized_df = tokenizer.transform(images_cnt_df)
    //tokenized_df.show()

    //remove stopwords 
    var remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val filtered_df = remover.transform(tokenized_df).drop("words")
    //filtered_df.printSchema()
    //filtered_df.show()

    //FIXME still need to try to squeeze ngrams in 
    //ngram = NGram(n=2, inputCol="filtered", outputCol="ngram")
    //ngram_df = ngram.transform(tokenized_df)

    //hashing
    var hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurized_df = hashingTF.transform(filtered_df).drop("filtered")

    //idf weighting
    var idf = new IDF().setInputCol("rawFeatures").setOutputCol("pre_features")
    var idfModel = idf.fit(featurized_df)
    val rescaled_df = idfModel.transform(featurized_df).drop("rawFeatures")
    rescaled_df.printSchema()
 
    //make adhoc DF
    sqlContext.udf.register("addhocfeat_appender", appendFeature _)
    def appendFeature_udf = udf(appendFeature _)
    var adhoc_df = rescaled_df.withColumn("features", appendFeature_udf(col("pre_features"),col("links_cnt"),col("images_cnt"))) 
    adhoc_df.printSchema()
    adhoc_df.show()

    //One can add more classifiers here
    //Random forest example
    //Index labels, adding metadata to the label column.
    //Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(adhoc_df)

    adhoc_df = labelIndexer.transform(adhoc_df) //.drop("rawFeatures")

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(adhoc_df)

    adhoc_df = featureIndexer.transform(adhoc_df)

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
      .setImpurity("gini")
      .setMaxDepth(4)
      .setMaxBins(32)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    //train CV split, stratified sampling
    //1 is under represented class
    val fractions = Map(0 -> 1.0, 1 -> 0.5)
    val sampledData = adhoc_df.stat.sampleBy("label", fractions, 36L)
    val Array(train, cv) = sampledData.randomSplit(Array(0.8, 0.2))

    //Prepare evaluator
    val metricName = "areaUnderPR"
    var ev = new BinaryClassificationEvaluator().setMetricName(metricName)

    //parameter search grid
    //One can add more parameters to the grid
    var paramGrid = new ParamGridBuilder()
                    .addGrid(hashingTF.numFeatures, Array(10, 20, 100))
                    .addGrid(rf.numTrees, Array(3, 5, 10))
                    .build()
  
    //set estimator 
    var crossval = new CrossValidator().setEstimator(rf).
                              setEstimatorParamMaps(paramGrid).
                              setEvaluator(ev).
                              setNumFolds(3)

    //Below is the single model vs parameter search switch 
    var model = crossval.fit(train)
    //var model = rf.fit(train)

    println("Evaluate model on test instances and compute test error...")
    var prediction = model.transform(cv)
    prediction = labelConverter.transform(prediction) 
    
    prediction.select("label", "text", "probability", "prediction").show(5)

    val result = ev.evaluate(prediction)
    println(metricName+": "+result)

    val cvErr = prediction.filter(prediction("label") === prediction("prediction")).count() / cv.count().toDouble
    println("CV Error = " + cvErr.toString)

   }
}
