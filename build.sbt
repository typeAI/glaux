
name := """glaux"""

version := "1.0"

scalaVersion := "2.11.6"

resolvers ++= Dependencies.resolvers

libraryDependencies ++= Dependencies.math ++ Dependencies.test

scalacOptions in Test ++= Seq("-Yrangepos")
