package glaux.reinforcement

import java.time.ZonedDateTime

import glaux.linalg.RowVector
import glaux.nn.Net
import glaux.reinforcement.QLearner.Transition


trait Agent {
  val qLearner : QLearner
  val policy: Policy
}

trait QLearner {
  protected val targetQFunction : Net
  val learningQFunction : Net
  val memory: Seq[Transition] //needs random access here
  val gamma: Double
  val actionResults: Map[Action, Q]
  def iterate(reward: Reward, recentHistory: History, isTerminal: Boolean): QLearner
}

object QLearner {
  case class TemporalState(inputs: RowVector, time: Time)
  case class State(fullHistory: History, isTerminal: Boolean)
  case class Transition(before: State, action: Action, reward: Reward, after: State)
}

