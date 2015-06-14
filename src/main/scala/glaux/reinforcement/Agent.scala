package glaux.reinforcement

import glaux.linalg.{Tensor, RowVector}
import glaux.nn.trainers.BatchTrainer
import glaux.nn.{NetOf, Loss, Net}


trait Agent {
  val qLearner: QLearner
  import qLearner.State
  type Policy = (State, Action => Q) => Action

  val policy: Policy
}


trait QLearner {
  type NetInput <: Tensor
  type Input <: Tensor
  type Net = NetOf[NetInput] //Need to fix input to the type level
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

  trait GenIteration {
    val trainingResult: TrainerResult
    val net: Net
    val memory: Memory
    lazy val actionQs: Map[Action, Q] = net.predict(memory.last.after).seqView.zipWithIndex.map(_.swap).toMap
    lazy val loss = trainingResult.lossInfo.cost
  }

  type Iteration <: GenIteration

  implicit def toInput(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration

  def init(inputDimension: Input#Dimensionality, numOfActions: Int): Iteration
}



