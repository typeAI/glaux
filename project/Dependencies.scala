import sbt.Keys._
import sbt._

object Dependencies {
  object Versions {
    val specs2 = "3.6.6"
    val nd4j = "0.0.3.5.5.5"
    val akka = "2.4.1"
  }


  val shapeless = Seq("com.chuusai" %% "shapeless" % "2.2.5")
  val cat = Seq("org.spire-math" %% "cats" % "0.3.0")

  val (test, integration) = {
    val specs = Seq(
      "org.specs2" %% "specs2-core" % Versions.specs2,
      "org.specs2" %% "specs2-mock" % Versions.specs2,
      "org.specs2" %% "specs2-scalacheck" % Versions.specs2
    )

    (specs.map(_ % "test"), specs.map(_ % "integration"))
  }

  val nd4j = Seq (
    //tobble between the following 2 lines to use GPU
    "org.nd4j" % "nd4j-jblas" % Versions.nd4j,
    //  "org.nd4j" % "nd4j-jcublas-7.0" % Versions.nd4j,
    "org.nd4j" % "nd4j-api" % Versions.nd4j
  )


  val akka = Seq (
    "com.typesafe.akka" %% "akka-actor" % Versions.akka,
    "com.typesafe.akka" %% "akka-testkit" % Versions.akka % "test"
  )

  val apacheCommonMath = Seq(
    "org.apache.commons" % "commons-math3" % "3.5"
  )

  val mongodb = Seq (
    "org.reactivemongo" %% "reactivemongo" % "0.11.7",
    "com.typesafe.play" %% "play-functional" % "2.4.2"
  )

  val config = Seq(
    "com.typesafe" % "config" % "1.2.1"
  )

  val commonSettings = Seq(
    scalaVersion in ThisBuild := "2.11.7",
    addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.7.1"),
    resolvers ++= Seq(
      Resolver.sonatypeRepo("releases"),
      Resolver.sonatypeRepo("snapshots"),
      Resolver.bintrayRepo("scalaz", "releases"),
      Resolver.bintrayRepo("typesafe", "maven-releases")
    )
  )

  val coreModuleSettings = commonSettings ++ Seq(
    libraryDependencies ++= shapeless ++ cat ++ test
  )


}
