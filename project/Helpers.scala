import sbt.TaskKey

object Helpers {
  val gc = TaskKey[Unit]("gc", "runs garbage collector")
  val gcTask = gc := {
    println("requesting garbage collection")
    System gc()
  }
}
