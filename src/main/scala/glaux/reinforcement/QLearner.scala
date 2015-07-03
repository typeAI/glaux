package glaux
package reinforcement

import glaux.linalg.Tensor
import glaux.nn.Loss
import glaux.nn.trainers.BatchTrainer

import scala.util.Try


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


  type InputDimension = Input#Dimensionality

  case class Observation( lastAction: Action,
                          reward: Reward,
                          recentHistory: History,
                          isTerminal: Boolean) {
    def startTime = recentHistory.head.time
  }

  case class TemporalState(readings: Input, time: Time)

  case class State(fullHistory: History, isTerminal: Boolean) {
    def endTime = fullHistory.last.time
    lazy val inputDimension: InputDimension = fullHistory.head.readings.dimension
  }

  case class Transition(before: State,
                        action: Action,
                        reward: Reward,
                        after: State)

  trait IterationLike {
    def trainingResult: TrainingResult
    def net: Net
    def memory: Memory
    def isTerminal: Boolean
    def state: State

    lazy val actionQs: Map[Action, Q] = if (isTerminal)
        Map.empty[Action, Q] //doesn't make sense to give Q function for terminal state
      else
        net.predict(state).seqView.zipWithIndex.map(_.swap).toMap

    lazy val loss: Loss = trainingResult.lossInfo.cost
  }

  type Iteration <: IterationLike

  private[reinforcement] def updateInit(iteration: Iteration, newHistory: History): Iteration

  implicit protected def inputToNet(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration


  /**
   *
   * @param initHistory initial history to construct the fist state, MUST NOT BE Terminal
   * @param numOfActions
   * @return
   */
  def init(initHistory: History, numOfActions: Int): Either[String, Iteration]


  protected def inputDimensionOfHistory(history: History): Option[InputDimension] =
    history.headOption.flatMap { head =>
      val inputDim = head.readings.dimension
      if (history.map(_.readings).exists(_.dimension != inputDim)) None else Some(inputDim)
    }

}



