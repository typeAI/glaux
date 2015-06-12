package glaux.reinforcement

import glaux.linalg.{Tensor, RowVector}
import glaux.nn.trainers.BatchTrainer
import glaux.nn.{Loss, Net}


trait Agent {
  val qLearner: QLearner
  import qLearner.State
  type Policy = (State, Action => Q) => Action

  val policy: Policy
}

trait QNet[IT <: Tensor] <: Net {
  type Input = IT
}

trait QLearner {
  type NetInput <: Tensor
  type Input <: Tensor
  type Net = QNet[NetInput] //Need to fix input to the type level
  type NetOutput = Net#Output
  type Trainer = BatchTrainer[Net]
  protected val trainer: Trainer
  type TrainerResult = trainer.BatchResult
  type History = Seq[TemporalState]
  type Memory = Seq[Transition]


  case class Observation( lastAction: Action,
                          reward: Reward,
                          recentHistory: History,
                          isTerminal: Boolean)

  case class TemporalState(readings: Input, time: Time)

  case class State(fullHistory: History, isTerminal: Boolean)

  case class Transition(before: State,
                        action: Action,
                        reward: Reward,
                        after: State)

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

  def init: Iteration = {
    val initNet = buildNet
    Iteration(initNet, Nil, trainer.init(initNet))
  }

  def buildNet: Net
}



