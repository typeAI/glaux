package glaux
package reinforcementlearning

import glaux.linearalgebra.Tensor
import glaux.neuralnetwork.Loss
import glaux.neuralnetwork.trainers.BatchTrainer
import glaux.reinforcementlearning.QLearner.{History ⇒ QHistory, Observation ⇒ QObservation, State ⇒ QState, Transition}

trait QLearner {
  type NetInput <: Tensor
  type Input <: Tensor
  type Net <: neuralnetwork.Net { type Input = NetInput } //Need to fix input to the type level

  type Trainer <: BatchTrainer[Net, _]
  type NetOutput = Net#Output

  protected val trainer: Trainer

  type TrainingResult = trainer.BatchResult

  type History = QHistory[Input]

  type InputDimension = Input#Dimensionality

  def historyLength: Int

  type Observation = QObservation[Input]
  type State = QState[Input]

  type Memory = Seq[Transition[Input]]

  trait IterationLike {
    def trainingResult: TrainingResult
    def net: Net
    def memory: Memory
    def isTerminal: Boolean
    def state: State

    lazy val actionQs: Map[Action, Q] = qMap(state)
    lazy val loss: Loss = trainingResult.lossInfo.cost

    def stateActionQ(s: State = state, action: Action): Q = {
      assert(!s.isTerminal)
      qMap(s).apply(action)
    }

    private def qMap(s: State): Map[Action, Q] = if (state.isTerminal) Map.empty else net.predict(s).seqView.zipWithIndex.map(_.swap).toMap

  }

  type Iteration <: IterationLike

  implicit protected def inputToNet(state: State): NetInput

  def iterate(lastIteration: Iteration, observation: Observation): Iteration = {
    assert(
      observation.recentHistory.forall(_.readings.dimension == lastIteration.state.inputDimension),
      s"input readings doesn't conform to preset reading dimension ${lastIteration.state.inputDimension}"
    )

    val relevantHistory = if (lastIteration.isTerminal) observation.recentHistory else concat(lastIteration.state.fullHistory, observation.recentHistory)

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
    history.headOption.flatMap { head ⇒
      val inputDim = head.readings.dimension
      if (history.map(_.readings).exists(_.dimension != inputDim)) None else Some(inputDim)
    }

  private[reinforcementlearning] def stateFromHistory(history: History, isTerminal: Boolean): State = {
    assert(canBuildStateFrom(history), "incorrect history length or dimension to create a state")
    QState(history.takeRight(historyLength), isTerminal)
  }

  def canBuildStateFrom(history: History): Boolean = {
    assert(inputDimensionOfHistory(history).isDefined, "history has inconsistent dimension")
    history.size >= historyLength
  }

}

object QLearner {

  case class TemporalState[Input <: Tensor](readings: Input, time: Time)

  type History[Input <: Tensor] = Seq[TemporalState[Input]]

  case class Observation[Input <: Tensor](
    lastAction:    Action,
    reward:        Reward,
    recentHistory: History[Input],
    isTerminal:    Boolean
  ) {
    assert(!recentHistory.isEmpty, "Cannot create an observation without recent history")
    def startTime = recentHistory.head.time

  }

  case class State[Input <: Tensor](fullHistory: History[Input], isTerminal: Boolean) {
    def endTime = fullHistory.last.time
    lazy val inputDimension: Input#Dimensionality = fullHistory.head.readings.dimension
  }

  case class Transition[Input <: Tensor](
    before: State[Input],
    action: Action,
    reward: Reward,
    after:  State[Input]
  )

}

