import org.scoverage.coveralls.Imports.CoverallsKeys._


name := """glaux"""

version := "1.0"

scalaVersion := "2.11.6"

resolvers ++= Dependencies.resolvers

libraryDependencies ++= Dependencies.math ++ Dependencies.test

scalacOptions in Test ++= Seq("-Yrangepos")

lazy val glaux = project in file(".")



coverallsToken := Some("unXzL1sDsWymqcb7JwJNqzAfBXSrLpgpA")
