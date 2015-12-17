package glaux.interfaces.akka.service

import akka.actor.ActorSystem

object AkkaApp extends App {
  val system = ActorSystem("glaux")
}
