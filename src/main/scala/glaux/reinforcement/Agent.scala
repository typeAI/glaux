package glaux.reinforcement

import glaux.linalg.RowVector
import glaux.nn.Net
import glaux.reinforcement.Agent.Policy
import glaux.reinforcement.QLearner.{State, Observation, Iteration}


trait Agent {
  val qLearner: QLearner
  val policy: Policy
}

object Agent {
  type Policy = (State, Action => Q) => Action
}

trait QLearner {
  def iterate(lastIteration: Iteration, observation: Observation): Iteration
}


object QLearner {
  type History = Seq[TemporalState]

  case class Iteration(targetQFunction: Net,
                       learningQFunction: Net,
                       memory: Seq[Transition], //Seq because we need random access here
                       actionResults: Map[Action, Q])

  case class Observation(reward: Reward,
                         recentHistory: History,
                         isTerminal: Boolean)

  case class TemporalState(inputs: RowVector, time: Time)

  case class State(fullHistory: History, isTerminal: Boolean)

  case class Transition(before: State,
                        action: Action,
                        reward: Reward,
                        after: State)


}

