name := "KaggleDato"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
    // Spark dependency
    "org.apache.spark"  % "spark-core_2.10" % "1.6.1" % "provided",
    "org.apache.spark"  % "spark-sql_2.10" % "1.6.1" % "provided",
    "org.apache.spark"  % "spark-mllib_2.10" % "1.6.1" % "provided"
)
