import sbt._
import Keys._

object Projects extends Build {
  
  lazy val glaux = project.in(file("."))
    .settings(moduleName := "root")
    .aggregate(linearAlgebra, neuralNetwork, reinforcementLearning, statistics )
    .settings(Common.settings:_*)
    .settings(Common.noPublishing: _*)

  lazy val linearAlgebra = project.in(file("linear-algebra"))
    .dependsOn(statistics)
    .settings(moduleName := "glaux-linear-algebra")
    .settings(
      libraryDependencies ++= Dependencies.nd4j
    )
    .settings(coreModuleSettings:_*)

  lazy val neuralNetwork = project.in(file("neural-network"))
    .dependsOn( linearAlgebra % "compile->compile;test->test", statistics)
    .settings(moduleName := "glaux-neural-network")
    .settings(coreModuleSettings:_*)

  lazy val reinforcementLearning = project.in(file("reinforcement-learning"))
    .dependsOn( neuralNetwork % "compile->compile;test->test", statistics)
    .settings(moduleName := "glaux-reinforcement-learning")
    .settings(coreModuleSettings:_*)

  lazy val statistics = project.in(file("statistics"))
    .settings(moduleName := "glaux-statistics")
    .settings(
      libraryDependencies ++= Dependencies.apacheCommonMath
    )
    .settings(coreModuleSettings:_*)

  val coreModuleSettings = Common.settings ++
                           Publish.settings ++ 
                           Format.settings ++ 
                           Testing.settings ++ 
                           Dependencies.settings
}
