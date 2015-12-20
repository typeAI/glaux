import org.scoverage.coveralls.Imports.CoverallsKeys._
import sbt._
import sbt.Keys._

object Testing {
  lazy val Integration = config("integration").extend(Test)

  def isIntegrationTest(name: String): Boolean = name.endsWith("Integration")
  def isUnitTest(name: String): Boolean = !isIntegrationTest(name)

  lazy val settings = {
    Seq(
      scalacOptions in Test ++= Seq("-Yrangepos"),
      testOptions in Test := Seq(Tests.Filter(isUnitTest))
    )
  }

  lazy val witIntegrationSettings = settings ++ Seq(
    testOptions in Integration := Seq(Tests.Filter(isIntegrationTest)),
    libraryDependencies ++= Dependencies.integration
  ) ++ inConfig(Integration)(Defaults.testTasks)
}
