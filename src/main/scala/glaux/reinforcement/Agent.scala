package glaux.reinforcement

import glaux.linalg.{Tensor, RowVector}
import glaux.nn.trainers.BatchTrainer
import glaux.nn.{Loss, Net}
import glaux.reinforcement.Agent.Policy
import glaux.reinforcement.QLearner.{Memory, State, Observation}


trait Agent {
  val qLearner: QLearner[_]
  val policy: Policy
}

object Agent {
  type Policy = (State, Action => Q) => Action
}

trait QNet[IT <: Tensor] <: Net {
  type Input = IT
}

trait QLearner[NetInput <: Tensor] {

  type Net = QNet[NetInput]
  type NetOutput = Net#Output
  type Trainer = BatchTrainer[Net]
  protected val trainer: Trainer
  type TrainerResult = trainer.BatchResult

  case class Iteration(targetNet: Net,
                       memory: Memory, //Seq because we need random access here
                       trainingResult: TrainerResult,
                       targetNetHitCount: Int = 0 ) {

    lazy val learningNet = trainingResult.net
    lazy val actionQs: Map[Action, Q] = learningNet.predict(memory.last.after).seqView.zipWithIndex.map(_.swap).toMap
    lazy val loss = trainingResult.lossInfo.cost
  }

  implicit def toInput(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration
}

object QLearner {

  type History = Seq[TemporalState]
  type Memory = Seq[Transition]


  case class Observation( lastAction: Action,
                          reward: Reward,
                          recentHistory: History,
                          isTerminal: Boolean)

  case class TemporalState(inputs: RowVector, time: Time)

  case class State(fullHistory: History, isTerminal: Boolean)

  case class Transition(before: State,
                        action: Action,
                        reward: Reward,
                        after: State)


}

