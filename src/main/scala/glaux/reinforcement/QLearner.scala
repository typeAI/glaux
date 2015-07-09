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
  type Trainer <: BatchTrainer[Net]

  protected val trainer: Trainer
  type TrainingResult = trainer.BatchResult
  type History = Seq[TemporalState]
  type Memory = Seq[Transition]


  type InputDimension = Input#Dimensionality

  case class Observation( lastAction: Action,
                          reward: Reward,
                          recentHistory: History,
                          isTerminal: Boolean) {
    assert(!recentHistory.isEmpty, "Cannot create an observation without recent history")
    def startTime = recentHistory.head.time

  }

  case class TemporalState(readings: Input, time: Time)

  val historyLength: Int

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

    lazy val actionQs: Map[Action, Q] = qMap(state)
    lazy val loss: Loss = trainingResult.lossInfo.cost

    def stateActionQ(s: State = state): Action => Q = {
      assert(!s.isTerminal)
      qMap(s).apply
    }

    private def qMap(s: State): Map[Action, Q] = if(state.isTerminal) Map.empty else net.predict(s).seqView.zipWithIndex.map(_.swap).toMap

  }

  type Iteration <: IterationLike

  implicit protected def inputToNet(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {
    assert(observation.recentHistory.forall(_.readings.dimension == lastIteration.state.inputDimension),
      s"input readings doesn't conform to preset reading dimension ${lastIteration.state.inputDimension}")

    val relevantHistory = if(lastIteration.isTerminal) observation.recentHistory else concat(lastIteration.state.fullHistory, observation.recentHistory)

    val currentState = stateFromHistory(relevantHistory, observation.isTerminal)

    doIterate(lastIteration, observation, currentState)

  }

  protected def doIterate(lastIteration: Iteration, observation: Observation, currentState: State): Iteration
  /**
   *
   * @param initHistory initial history to construct the fist state, MUST NOT BE Terminal
   * @param numOfActions
   * @return
   */
  def init(initHistory: History, numOfActions: Int): Iteration =
    doInit(stateFromHistory(initHistory, false), numOfActions, inputDimensionOfHistory(initHistory).get)


  protected def doInit(initState: State, numOfActions: Action, inputDim: InputDimension): Iteration

  protected def concat(previous: History, newHistory: History): History = {
    val relevantPreviousHistory = previous.filter(_.time.isBefore(newHistory.head.time))
    (relevantPreviousHistory ++ newHistory)
  }

  protected def inputDimensionOfHistory(history: History): Option[InputDimension] =
    history.headOption.flatMap { head =>
      val inputDim = head.readings.dimension
      if (history.map(_.readings).exists(_.dimension != inputDim)) None else Some(inputDim)
    }

  private[reinforcement] def stateFromHistory(history: History, isTerminal: Boolean): State = {
    assert(canBuildStateFrom(history), "incorrect history length or dimension to create a state")
    State(history.takeRight(historyLength), isTerminal)
  }

  def canBuildStateFrom(history: History): Boolean = 
    history.size >= historyLength && inputDimensionOfHistory(history).isDefined

}



