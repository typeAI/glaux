import sbt.Keys._
import sbt._

object Dependencies {
  object Versions {
    val specs2 = "3.6.6"
    val nd4j = "0.0.3.5.5.5"
  }


  val shapeless = Seq("com.chuusai" %% "shapeless" % "2.2.5")
  val cat = Seq("org.spire-math" %% "cats" % "0.3.0")
  val test = Seq(
    "org.specs2" %% "specs2-core" % Versions.specs2 % "test",
    "org.specs2" %% "specs2-mock" % Versions.specs2 % "test"
  )

  val nd4j = Seq (
    //tobble between the following 2 lines to use GPU
    "org.nd4j" % "nd4j-jblas" % Versions.nd4j,
    //  "org.nd4j" % "nd4j-jcublas-7.0" % Versions.nd4j,
    "org.nd4j" % "nd4j-api" % Versions.nd4j
  )


  val apacheCommonMath = Seq(
    "org.apache.commons" % "commons-math3" % "3.5"
  )

  val commonSettings = Seq(
    scalaVersion in ThisBuild := "2.11.7",
    addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.7.1")
  )

  val testSettings = commonSettings ++ Seq(
    libraryDependencies ++= test
  )

  val settings = commonSettings ++ testSettings ++ Seq(
    libraryDependencies ++= shapeless ++ cat,
    resolvers ++= Seq(
      Resolver.sonatypeRepo("releases"),
      Resolver.sonatypeRepo("snapshots"),
      Resolver.bintrayRepo("scalaz", "releases")
    )
  )

}
