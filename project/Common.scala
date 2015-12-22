import sbt.Keys._
import sbt._

object Common {

  val settings = Seq(
    Helpers.gcTask,
    scalacOptions ++= Seq(
      "-deprecation",
      "-unchecked",
      "-Xlint"
    )
  ) 

}
