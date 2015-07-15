package glaux.reinforcement

import glaux.linalg.Dimension.Row
import glaux.linalg.RowVector
import glaux.linalg.Tensor.TensorBuilder
import glaux.nn.trainers.VanillaSGD
import glaux.nn.trainers.SGD.SGDSettings
import glaux.reinforcement.DeepMindQLearner.Simplified
import glaux.reinforcement.QAgent.{Session => QSession}
import glaux.reinforcement.QLearner.{Observation, TemporalState}

import scala.util.Random

trait QAgent {
  type Learner <: QLearner
  val qLearner: Learner

  import qLearner.{Iteration, State}

  type Policy = (State, State => Action => Q ) => Action

  val numOfActions: Int
  val policy: Policy
  type Session = QSession[Iteration]

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input

  protected def toHistory(readings: Iterable[Reading]): qLearner.History = {
    readings.toSeq.map {
      case (r, time) => TemporalState(readingsToInput(r), time)
    }
  }


  def start(initReadings: Iterable[Reading], previous: Option[Session]): Either[String, Session] = {
    val initHistory = toHistory(initReadings)

    if(qLearner.canBuildStateFrom(initHistory))

      Right(QSession(iteration = previous.map(_.iteration).getOrElse(qLearner.init(initHistory, numOfActions)),
              currentReward = 0,
              currentReadings = initReadings.toVector))
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
        val action = policy(newIteration.state, newIteration.stateActionQ)
        (action, QSession(newIteration, 0, Vector.empty, Some(action)))
      case session.Status.PendingFirstAction =>
        val firstAction = policy(qLearner.stateFromHistory(currentHistory, false), session.iteration.stateActionQ)
        (firstAction, session.copy(lastAction = Some(firstAction), currentReadings = Vector.empty))
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
  case class Session[IterationT](iteration: IterationT,
                                 currentReward: Reward,
                                 currentReadings: Vector[Reading],
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


case class SimpleQAgent(numOfActions: Int, historyLength: Int = 10) extends QAgent {
  type Learner = DeepMindQLearner.Simplified
  val trainer = VanillaSGD[Simplified#Net](SGDSettings(learningRate = 0.05))
  val qLearner = DeepMindQLearner.Simplified(historyLength = historyLength, batchSize = 20, trainer = trainer)

  protected def readingsToInput(readings: Seq[Double]): qLearner.Input = RowVector(readings :_*)

  val policy: Policy = (state, _) => Random.nextInt(numOfActions) //todo: implement a real policy


}
