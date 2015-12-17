import sbt._
import Keys._

object Projects extends Build {
  
  lazy val glaux = project.in(file("."))
    .configs(Testing.Integration)
    .settings(moduleName := "root")
    .aggregate(linearAlgebra, neuralNetwork, reinforcementLearning, statistics, akkaInterfaceService, interfaceAPI, persistenceMongoDB )
    .settings(Common.settings:_*)
    .settings(Testing.witIntegrationSettings:_*)

    .settings(noPublishing: _*)

  lazy val linearAlgebra = project.in(file("linear-algebra"))
    .dependsOn(statistics)
    .aggregate(statistics)
    .settings(moduleName := "glaux-linear-algebra")
    .settings(
      libraryDependencies ++= Dependencies.nd4j
    )
    .settings(coreModuleSettings:_*)

  lazy val neuralNetwork = project.in(file("neural-network"))
    .dependsOn( linearAlgebra )
    .aggregate( linearAlgebra)
    .settings(moduleName := "glaux-neural-network")
    .settings(coreModuleSettings:_*)

  lazy val reinforcementLearning = project.in(file("reinforcement-learning"))
    .dependsOn( neuralNetwork )
    .aggregate( neuralNetwork )
    .settings(moduleName := "glaux-reinforcement-learning")
    .settings(coreModuleSettings:_*)

  lazy val statistics = project.in(file("statistics"))
    .settings(moduleName := "glaux-statistics")
    .settings(
      libraryDependencies ++= Dependencies.apacheCommonMath
    )
    .settings(coreModuleSettings:_*)

  lazy val akkaInterfaceService = project.in(file("akka-interface-service"))
    .dependsOn(interfaceAPI)
    .aggregate(interfaceAPI)
    .settings(moduleName := "glaux-akka-interface")
    .settings(
      libraryDependencies ++= Dependencies.akka
    )
    .settings(coreModuleSettings:_*)

  lazy val interfaceAPI = project.in(file("interface-api"))
    .dependsOn( reinforcementLearning)
    .aggregate( reinforcementLearning)
    .settings(moduleName := "glaux-akka-interface-api")
    .settings(coreModuleSettings:_*)

  lazy val persistenceMongoDB = project.in(file("persistence-mongodb"))
    .configs(Testing.Integration)
    .dependsOn( interfaceAPI )
    .aggregate( interfaceAPI )
    .settings(moduleName := "glaux-persistence-mongodb")
    .settings(
      libraryDependencies ++= Dependencies.mongodb
    )
    .settings(coreModuleSettings:_*)
    .settings(Testing.witIntegrationSettings:_*)

  val coreModuleSettings = Common.settings ++
                           Publish.settings ++ 
                           Format.settings ++ 
                           Testing.settings ++ 
                           Dependencies.coreModuleSettings

  val noPublishing = Seq(publish := (), publishLocal := (), publishArtifact := false)
}
