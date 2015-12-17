import sbt._, Keys._
import bintray.BintrayKeys._


object Publish {

  val bintraySettings = Seq(
    bintrayOrganization := Some("org.typeAI"),
    bintrayPackageLabels := Seq("glaux")
  )

  val publishingSettings = Seq(
    organization in ThisBuild := "glaux",
    publishMavenStyle := true,
    licenses := Seq("Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0.html")),
    homepage := Some(url("http://typeAI.github.io/glaux")),
    scmInfo := Some(ScmInfo(url("https://github.com/typeAI/glaux"),
      "git@github.com:typeAI/glaux.git")),
    pomIncludeRepository := { _ => false },
    publishArtifact in Test := false
  )

  val settings = bintraySettings ++ publishingSettings
}

