import akka.actor.ActorSystem
import glaux.reinforcementlearning.DeepMindQLearner

object AkkaApp extends App {
  val system = ActorSystem("shifu")
}
