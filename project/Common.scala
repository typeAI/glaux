import sbt.Keys._
import sbt._

object Common {


  val settings = Seq(
    scalacOptions ++= Seq(
      "-deprecation",
      "-unchecked",
      "-Xlint"
    )
  ) 

}
