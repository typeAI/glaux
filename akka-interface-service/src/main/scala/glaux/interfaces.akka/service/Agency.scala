package glaux.interfaces.akka
package service

import Protocols.Agency._

import akka.actor._
import glaux.interfaces.akka.service.Agency.AgentCreated
import Protocols.{ Confirmed, Rejected }
import glaux.interfaces.api.domain._
import glaux.interfaces.api.persistence._
import glaux.reinforcementlearning.{ AdvancedQAgent, QAgent, SimpleQAgent }

import scala.concurrent.Future

class Agency(persistenceImpl: PersistenceImpl) extends Actor with ActorLogging {
  import Agency.AgentActor
  import context.dispatcher

  val settingsRepo = persistenceImpl.agentSettingsPersistence

  def receive = running(Map.empty)

  def running(agents: Map[SessionId, AgentActor]): Receive = {
    case GetAgentForUser(sessionId) ⇒
      val agentResult = agents.get(sessionId)
      if (agentResult.isDefined)
        sender ! AgentRef(agentResult.get)
      else {
        val replyTo = sender
        createAgentForUser(sessionId).foreach {
          case Some(newAgent) ⇒ self ! AgentCreated(sessionId, newAgent, replyTo)
          case None           ⇒ replyTo ! Rejected
        }
      }

    case AgentCreated(sessionId, newAgent, replyTo) ⇒
      val toUse = agents.getOrElse(sessionId, newAgent) //discard the newly created agent if there is already one created at this moment
      replyTo ! AgentRef(toUse)
      if (toUse == newAgent)
        context become running(agents + (sessionId → toUse))

    case Terminated(agent) ⇒
      val aidResult = agents.collectFirst {
        case (aid, `agent`) ⇒ aid
      }
      aidResult.foreach { agentId ⇒
        context become running(agents - agentId)
      }

    case CreateAgentSettings(settings) ⇒
      val replyTo = sender
      settingsRepo.upsert(settings).map[Any](_ ⇒ Confirmed).recover { case _ ⇒ Rejected }.foreach { replyMsg ⇒
        replyTo ! replyMsg
      }

  }

  def createAgentForUser(sessionId: SessionId): Future[Option[ActorRef]] = {
    settingsRepo.get(sessionId.agentName).map(_.map {
      case AdvancedAgentSettings(_, na, ls, ts) ⇒
        val agent: AdvancedQAgent = AdvancedQAgent(na, ls, ts)
        AgentForUser.props(agent, sessionId)(persistenceImpl.advanceQAgentSessionPersistence)
    }.map { agentProps ⇒
      val agent = context.actorOf(agentProps)
      context watch agent
      agent
    })
  }

}

object Agency {
  type AgentActor = ActorRef

  def props(implicit p: PersistenceImpl): Props = Props(new Agency(p))

  private case class AgentCreated(sessionId: SessionId, agentActor: AgentActor, replyTo: ActorRef)
}

