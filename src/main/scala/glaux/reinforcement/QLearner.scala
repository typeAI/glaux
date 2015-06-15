package glaux.reinforcement

import glaux.linalg.Tensor
import glaux.nn.NetOf
import glaux.nn.trainers.BatchTrainer


trait QLearner {
  type NetInput <: Tensor
  type Input <: Tensor
  type Net <: NetOf[NetInput] //Need to fix input to the type level
  type NetOutput = Net#Output
  type Trainer = BatchTrainer[Net]
  protected val trainer: Trainer
  type TrainingResult = trainer.BatchResult
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

  trait IterationLike {
    val trainingResult: TrainingResult
    val net: Net
    val memory: Memory
    lazy val latestState = memory.last.after
    lazy val actionQs: Map[Action, Q] = net.predict(latestState).seqView.zipWithIndex.map(_.swap).toMap
    lazy val loss = trainingResult.lossInfo.cost
    lazy val inputDimension = net.inputDimension
  }

  type Iteration <: IterationLike

  implicit def inputToNet(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration

  def init(inputDimension: Input#Dimensionality, numOfActions: Int): Iteration
}



