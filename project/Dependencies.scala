import sbt._


object Dependencies {

  object Versions {
    val nd4j = "0.0.3.5.5.4-SNAPSHOT"
  }

  val akka = Seq (
    "com.typesafe.akka" %% "akka-actor" % "2.3.11",
    "com.typesafe.akka" %% "akka-testkit" % "2.3.11" % "test"
  )

  val test = Seq (
    "org.specs2" %% "specs2-core" % "3.6" % "test"
  )

  val math = Seq (
    "org.nd4j" % "nd4j-jblas" % Versions.nd4j,
    "org.nd4j" % "nd4j-scala-api" % Versions.nd4j
  )

}

