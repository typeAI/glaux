import sbt._
import Keys._

object Projects extends Build {
  
  lazy val glaux = project.in(file("."))
    .settings(moduleName := "root")
    .aggregate(linearAlgebra, neuralNetwork, reinforcementLearning, statistics, akkaInterfaceService, akkaInterfaceAPI, persistenceMongoDB )
    .settings(Common.settings:_*)
    .settings(noPublishing: _*)

  lazy val linearAlgebra = project.in(file("linear-algebra"))
    .dependsOn(statistics)
    .settings(moduleName := "glaux-linear-algebra")
    .settings(
      libraryDependencies ++= Dependencies.nd4j
    )
    .settings(coreModuleSettings:_*)

  lazy val neuralNetwork = project.in(file("neural-network"))
    .dependsOn( linearAlgebra, statistics)
    .settings(moduleName := "glaux-neural-network")
    .settings(coreModuleSettings:_*)

  lazy val reinforcementLearning = project.in(file("reinforcement-learning"))
    .dependsOn( neuralNetwork, statistics)
    .settings(moduleName := "glaux-reinforcement-learning")
    .settings(coreModuleSettings:_*)

  lazy val statistics = project.in(file("statistics"))
    .settings(moduleName := "glaux-statistics")
    .settings(
      libraryDependencies ++= Dependencies.apacheCommonMath
    )
    .settings(coreModuleSettings:_*)

  lazy val akkaInterfaceService = project.in(file("akka-interface-service"))
    .dependsOn(neuralNetwork, statistics, persistenceMongoDB, akkaInterfaceAPI, reinforcementLearning) //todo: remove this direct dependency on persistence MongoDB
    .settings(moduleName := "glaux-akka-interface")
    .settings(
      libraryDependencies ++= Dependencies.akka
    )
    .settings(coreModuleSettings:_*)

  lazy val akkaInterfaceAPI = project.in(file("akka-interface-api"))
    .dependsOn(neuralNetwork, statistics, reinforcementLearning) //todo: remove this direct dependency on persistence MongoDB
    .settings(moduleName := "glaux-akka-interface-api")
    .settings(
      libraryDependencies ++= Dependencies.akka
    )
    .settings(coreModuleSettings:_*)

  lazy val persistenceMongoDB = project.in(file("persistence-mongodb"))
    .dependsOn(neuralNetwork, statistics, akkaInterfaceAPI, linearAlgebra)
    .settings(moduleName := "glaux-persistence-mongodb")
    .settings(
      libraryDependencies ++= Dependencies.mongodb
    )
    .settings(coreModuleSettings:_*)

  val coreModuleSettings = Common.settings ++
                           Publish.settings ++ 
                           Format.settings ++ 
                           Testing.settings ++ 
                           Dependencies.coreModuleSettings

  val noPublishing = Seq(publish := (), publishLocal := (), publishArtifact := false)
}
