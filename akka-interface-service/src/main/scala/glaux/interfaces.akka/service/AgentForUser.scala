package glaux.interfaces.akka
package service

import akka.actor.{Props, Actor, ActorLogging}
import Protocols.{Response, Confirmed}
import glaux.interfaces.api.domain.{SessionId, ProfileId, Reading}
import Protocols.Agent._
import glaux.interfaces.api.persistence.SessionPersistence
import glaux.reinforcementlearning._
import scala.concurrent.Future
import scala.util.{Failure, Success}

class AgentForUser[AT <: QAgent: SessionPersistence](qAgent: AT, sessionId: SessionId) extends Actor with ActorLogging {

  import qAgent.Session

  private val repo = implicitly[SessionPersistence[AT]] //todo: use cached implicit from shapeless here.
  import context.dispatcher

  private lazy val previousSessionF: Future[Option[Session]] = repo.get(qAgent, sessionId)

  def receive: Receive = initializing(Vector.empty)

  def initializing(initReadings: Vector[Reading]): Receive = {
    case Report(reading, _) ⇒
      val (newContext, response) = tryStart(initReadings :+ reading)
      sender ! response
      context become newContext

    case m @ (RequestAction | QueryStatus) ⇒ tryStart(initReadings) match {
      case (newContext, ActionsAvailable) ⇒
        self forward m
        context become newContext
      case (newContext, response) ⇒
        sender ! response
        context become newContext
    }

  }

  def inSession(session: Session): Receive = {
    case Report(reading, reward) ⇒
      context become inSession(qAgent.report(reading, reward, session))
      sender ! ActionsAvailable

    case RequestAction ⇒
      val (action, newSession) = qAgent.requestAction(session)
      sender ! ActionResult(action)
      context become inSession(newSession)

    case ReportTermination(reading, reward) ⇒
      val newS = qAgent.report(reading, reward, session)
      val closed = qAgent.close(newS)
      val replyTo = sender
      storeSession(closed).map { _ ⇒
        replyTo ! Confirmed
      }.onFailure {
        case e: Throwable ⇒ throw e
      }
      context stop self

    case QueryStatus ⇒
      sender ! AgentStatus(session.iteration.memory.size, session.iteration.loss)
  }

  private def tryStart(readings: Vector[Reading]): (Receive, Response) = {

    def startAgent(previousSession: Option[Session]): (Receive, Response) =
      qAgent.start(readings, previousSession) match {
        case Left(m) ⇒
          (initializing(readings), PendingMoreReadings)

        case Right(session) ⇒
          (inSession(session), ActionsAvailable)
      }

    previousSessionF.value match {
      case Some(Success(previousSession)) ⇒
        startAgent(previousSession)
      case Some(Failure(e)) ⇒
        throw e
      case None ⇒ //future not completed yet
        (initializing(readings), Initializing)
    }

  }

  private def storeSession(session: Session): Future[Unit] = repo.upsert(qAgent, sessionId)(session)

}

object AgentForUser {
  def props[AT <: QAgent: SessionPersistence](qAgent: AT, sessionId: SessionId): Props =

    Props(new AgentForUser(qAgent, sessionId))

}
