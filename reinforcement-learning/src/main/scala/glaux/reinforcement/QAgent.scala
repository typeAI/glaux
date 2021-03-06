package glaux.reinforcementlearning

import glaux.linearalgebra.{Vol, RowVector}
import glaux.neuralnetwork.Net
import glaux.neuralnetwork.Net.{DefaultNet, Updater}
import glaux.neuralnetwork.trainers.VanillaSGD
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.reinforcementlearning
import glaux.reinforcementlearning.DeepMindQLearner.{ConvolutionBased, Simplified}
import glaux.reinforcementlearning.Policy.DecisionContext
import glaux.reinforcementlearning.QAgent.{Session ⇒ QSession}
import glaux.reinforcementlearning.QLearner.{Observation, TemporalState}

import scala.util.Random

trait QAgent {
  type Learner <: QLearner
  val qLearner: Learner

  import qLearner.{Iteration, State}

  type Policy <: reinforcementlearning.Policy[State]

  val numOfActions: Int
  val policy: Policy
  type Session = QSession[Iteration, policy.Context]

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input

  protected def toHistory(readings: Iterable[Reading]): qLearner.History =
    readings.toSeq.map {
      case (r, time) ⇒ TemporalState(readingsToInput(r), time)
    }

  def start(initReadings: Iterable[Reading], previous: Option[Session]): Either[String, Session] = {
    val initHistory = toHistory(initReadings)

    if (qLearner.canBuildStateFrom(initHistory))
      Right(QSession(
        iteration = previous.map(_.iteration).getOrElse(qLearner.init(initHistory, numOfActions)),
        currentReward = 0,
        currentReadings = initReadings.toVector,
        decisionContext = policy.init
      ))
    else
      Left("Not enough initial history to start a session")
  }

  def report(reading: Reading, reward: Reward, session: Session): Session = {
    assert(!session.isClosed)
    session.copy(
      currentReadings = session.currentReadings :+ reading,
      currentReward = session.currentReward + reward
    )
  }

  def requestAction(session: Session): (Action, Session) = {
    assert(!session.isClosed)
    val currentHistory = toHistory(session.currentReadings)
    session.status match {
      case session.Status.ReadyToForward ⇒
        val newIteration = forward(session, false)
        val (action, decisionContext) = policy.decide(newIteration.state, newIteration.stateActionQ, session.decisionContext)
        (action, QSession(newIteration, 0, Vector.empty, decisionContext, Some(action)))

      case session.Status.PendingFirstAction ⇒
        val currentState = qLearner.stateFromHistory(currentHistory, false)
        val (firstAction, decisionContext) = policy.decide(currentState, session.iteration.stateActionQ, session.decisionContext)
        (firstAction, session.copy(
          lastAction = Some(firstAction),
          currentReadings = Vector.empty,
          decisionContext = decisionContext
        ))

      case session.Status.PendingReadingAfterAction ⇒
        (session.lastAction.get, session) //simply repeat the last action

      case _ ⇒ throw new NotImplementedError(s"request action not implemented for ${session.status}")
    }
  }

  def close(session: Session): Session = {
    assert(!session.isClosed)
    if (session.canForward) {
      session.copy(iteration = forward(session, true), isClosed = true)
    } else
      session.copy(isClosed = true)
  }

  private def forward(session: Session, terminal: Boolean): Iteration = {
    val observation = Observation(session.lastAction.get, session.currentReward, toHistory(session.currentReadings), terminal)
    qLearner.iterate(session.iteration, observation).asInstanceOf[Iteration]
  }
}

object QAgent {
  case class Session[IterationT <: QLearner#IterationLike, PolicyContextT <: Policy.DecisionContext](
    iteration:       IterationT,
    currentReward:   Reward,
    currentReadings: Vector[Reading],
    decisionContext: PolicyContextT,
    lastAction:      Option[Action]  = None,
    isClosed:        Boolean         = false
  ) {
    def canForward: Boolean = !isClosed && lastAction.isDefined && !currentReadings.isEmpty

    object Status extends Enumeration {
      val ReadyToForward, PendingReadingAfterAction, PendingFirstAction, Closed = Value
    }

    def status: Status.Value = if (isClosed) Status.Closed
    else if (lastAction.isDefined)
      if (currentReadings.isEmpty)
        Status.PendingReadingAfterAction
      else
        Status.ReadyToForward
    else if (currentReadings.isEmpty)
      throw new Exception("session should not be in this state")
    else
      Status.PendingFirstAction
  }
}

trait DeepMindQAgent[LT <: DeepMindQLearner] extends QAgent {
  type Learner = LT
  import qLearner.State

  type Policy = Policy.Annealing[State]
  val policy: Policy = Policy.Annealing[State](numOfActions, 0.05, 10000)
  implicit val updater: Net.Updater[LT#Net]
}

case class SimpleQAgent(numOfActions: Int, historyLength: Int = 10) extends DeepMindQAgent[DeepMindQLearner.Simplified] {

  val trainer = VanillaSGD[Learner#Net](SGDSettings(learningRate = 0.05))

  val qLearner = DeepMindQLearner.Simplified(historyLength = historyLength, batchSize = 20, trainer = trainer)

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input = RowVector(readings: _*)
  implicit lazy val updater = implicitly[Updater[DefaultNet[RowVector]]]

}

case class AdvancedQAgent(
  numOfActions:    Int,
  learnerSettings: ConvolutionBased.Settings,
  trainerSettings: SGDSettings
) extends DeepMindQAgent[DeepMindQLearner.ConvolutionBased] {

  val trainer = VanillaSGD[Learner#Net](trainerSettings)

  val qLearner = ConvolutionBased(trainer, learnerSettings)

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input = RowVector(readings: _*)
  implicit lazy val updater = implicitly[Updater[DefaultNet[Vol]]]

}
