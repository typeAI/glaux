import sbt.Keys._
import sbt._


object Dependencies {

  object Versions {
    val nd4j = "0.0.3.5.5.5"
  }

  val resolvers = Seq(
    "scalaz-bintray" at "http://dl.bintray.com/scalaz/releases"
  )


  val akka = Seq (
    "com.typesafe.akka" %% "akka-actor" % "2.3.11",
    "com.typesafe.akka" %% "akka-testkit" % "2.3.11" % "test"
  )

  val test = Seq (
    "org.specs2" %% "specs2-core" % "3.6.1" % "test"
  )

  val math = Seq (
    "org.nd4j" % "nd4j-jblas" % Versions.nd4j,  //switch between the following line to use GPU
//  "org.nd4j" % "nd4j-jcublas-7.0" % Versions.nd4j,
    "org.nd4j" % "nd4j-api" % Versions.nd4j
  )

}

