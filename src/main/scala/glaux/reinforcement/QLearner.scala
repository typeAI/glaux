package glaux
package reinforcement

import glaux.linalg.Tensor
import glaux.nn.Loss
import glaux.nn.trainers.BatchTrainer


trait QLearner {
  type NetInput <: Tensor
  type Input <: Tensor
  type Net <: nn.Net { type Input = NetInput } //Need to fix input to the type level

  type NetOutput = Net#Output
  type Trainer = BatchTrainer[Net]
  protected val trainer: Trainer
  type TrainingResult = trainer.BatchResult
  type History = Seq[TemporalState]
  type Memory = Seq[Transition]

  case class Observation( lastAction: Action,
                          reward: Reward,
                          recentHistory: History,
                          isTerminal: Boolean) {
    lazy val startTime = recentHistory.head.time
  }

  case class TemporalState(readings: Input, time: Time)

  case class State(fullHistory: History, isTerminal: Boolean)

  case class Transition(before: State,
                        action: Action,
                        reward: Reward,
                        after: State)

  trait IterationLike {
    def trainingResult: TrainingResult
    def net: Net
    def memory: Memory
    def isTerminal: Boolean
    def latestState: Option[State]

    lazy val actionQs: Map[Action, Q] = {
      val void = Map.empty[Action, Q]
      if(isTerminal)
        void
      else
        latestState.fold(void){ net.predict(_).seqView.zipWithIndex.map(_.swap).toMap }
    }

    lazy val loss: Loss = trainingResult.lossInfo.cost
    lazy val inputDimension = net.inputDimension
  }

  type Iteration <: IterationLike

  implicit def inputToNet(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration

  def init(inputDimension: Input#Dimensionality, numOfActions: Int): Iteration
}



