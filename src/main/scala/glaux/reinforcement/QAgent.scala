package glaux.reinforcement

import glaux.linalg.{Vol, RowVector}
import glaux.nn.Net
import glaux.nn.Net.{DefaultNet, Updater}
import glaux.nn.trainers.VanillaSGD
import glaux.nn.trainers.SGD.SGDSettings
import glaux.reinforcement
import glaux.reinforcement.DeepMindQLearner.{ConvolutionBased, Simplified}
import glaux.reinforcement.Policy.DecisionContext
import glaux.reinforcement.QAgent.{Session => QSession}
import glaux.reinforcement.QLearner.{Observation, TemporalState}

import scala.util.Random

trait QAgent {
  type Learner <: QLearner
  val qLearner: Learner

  import qLearner.{Iteration, State}

  type Policy <: reinforcement.Policy[State]

  val numOfActions: Int
  val policy: Policy
  type Session = QSession[Iteration, policy.Context]

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input

  protected def toHistory(readings: Iterable[Reading]): qLearner.History =
    readings.toSeq.map {
      case (r, time) => TemporalState(readingsToInput(r), time)
    }


  def start(initReadings: Iterable[Reading], previous: Option[Session]): Either[String, Session] = {
    val initHistory = toHistory(initReadings)

    if(qLearner.canBuildStateFrom(initHistory))
      Right(QSession(iteration = previous.map(_.iteration).getOrElse(qLearner.init(initHistory, numOfActions)),
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
      case session.Status.ReadyToForward =>
        val newIteration = forward(session, false)
        val (action, decisionContext) = policy.decide(newIteration.state, newIteration.stateActionQ, session.decisionContext)
        (action, QSession(newIteration, 0, Vector.empty, decisionContext, Some(action)))

      case session.Status.PendingFirstAction =>
        val currentState = qLearner.stateFromHistory(currentHistory, false)
        val (firstAction, decisionContext) = policy.decide(currentState, session.iteration.stateActionQ, session.decisionContext)
        (firstAction, session.copy( lastAction = Some(firstAction),
                                    currentReadings = Vector.empty,
                                    decisionContext = decisionContext))

      case session.Status.PendingReadingAfterAction =>
        (session.lastAction.get, session)  //simply repeat the last action

      case _ => throw new NotImplementedError(s"request action not implemented for ${session.status}")
    }
  }

  def close(session: Session): Session = {
    assert(!session.isClosed)
    if(session.canForward) {
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
  case class Session[ IterationT <: QLearner#IterationLike,
                      PolicyContextT <: Policy.DecisionContext]
                                (iteration: IterationT,
                                 currentReward: Reward,
                                 currentReadings: Vector[Reading],
                                 decisionContext: PolicyContextT,
                                 lastAction: Option[Action] = None,
                                 isClosed: Boolean = false) {
    def canForward: Boolean = !isClosed && lastAction.isDefined && !currentReadings.isEmpty

    object Status extends Enumeration {
      val ReadyToForward, PendingReadingAfterAction, PendingFirstAction, Closed = Value
    }

    def status: Status.Value =  if (isClosed) Status.Closed
                                else
                                  if(lastAction.isDefined)
                                    if(currentReadings.isEmpty)
                                      Status.PendingReadingAfterAction
                                    else
                                      Status.ReadyToForward
                                  else
                                    if(currentReadings.isEmpty)
                                      throw new Exception("session should not be in this state")
                                    else
                                      Status.PendingFirstAction
  }
}

trait DeepMindQAgent[LT <: DeepMindQLearner] extends QAgent {
  type Learner = LT
  import qLearner.State
  implicit val updater: Net.Updater[LT#Net]

<<<<<<< HEAD
  val trainer = VanillaSGD[Learner#Net](SGDSettings(learningRate = 0.05))
=======
>>>>>>> 34f089f3b1a4e1b7659d51cec2dc271a01d8b60d

  type Policy = Policy.Annealing[State]
  val policy: Policy = Policy.Annealing[State](numOfActions, 0.05, 10000)
}

case class SimpleQAgent(numOfActions: Int, historyLength: Int = 10) extends DeepMindQAgent[DeepMindQLearner.Simplified] {

<<<<<<< HEAD
=======
  val trainer = VanillaSGD[Learner#Net](SGDSettings(learningRate = 0.05))

>>>>>>> 34f089f3b1a4e1b7659d51cec2dc271a01d8b60d
  val qLearner = DeepMindQLearner.Simplified(historyLength = historyLength, batchSize = 20, trainer = trainer)

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input = RowVector(readings :_*)

  implicit val updater = implicitly[Updater[DefaultNet[RowVector]]]
}

<<<<<<< HEAD
case class AdvancedQAgent(numOfActions: Int, historyLength: Int = 50) extends DeepMindQAgent[DeepMindQLearner.ConvolutionBased] {

  val qLearner = DeepMindQLearner.ConvolutionBased(historyLength = historyLength, batchSize = 20, trainer = trainer)
=======
case class AdvancedQAgent(numOfActions: Int,
                          learnerSettings: ConvolutionBased.Settings,
                          trainerSettings: SGDSettings) extends DeepMindQAgent[DeepMindQLearner.ConvolutionBased] {

  val trainer = VanillaSGD[Learner#Net](trainerSettings)

  val qLearner = ConvolutionBased(trainer, learnerSettings)
>>>>>>> 34f089f3b1a4e1b7659d51cec2dc271a01d8b60d

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input = RowVector(readings :_*)

  implicit val updater = implicitly[Updater[DefaultNet[Vol]]]
}
