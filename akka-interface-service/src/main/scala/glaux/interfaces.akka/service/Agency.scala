package glaux.interfaces.akka
package service

import api._
import Protocols.Agency._

import akka.actor._
import glaux.interfaces.akka.service.Agency.AgentCreated
import glaux.interfaces.akka.api.Protocols.{ Confirmed, Rejected }
import glaux.interfaces.akka.api.domain._
import glaux.persistence.mongodb.AgentSettingsRepo
import glaux.persistence.mongodb.AgentSettingsRepo
import glaux.reinforcementlearning.{ AdvancedQAgent, QAgent, SimpleQAgent }
import glaux.persistence.mongodb.QSessionHandler._

import scala.concurrent.Future

class Agency(repo: AgentSettingsRepo) extends Actor with ActorLogging {
  import Agency.AgentActor
  import context.dispatcher

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
      repo.upsert(settings).map[Any](_ ⇒ Confirmed).recover { case _ ⇒ Rejected }.foreach { replyMsg ⇒
        replyTo ! replyMsg
      }

  }

  def createAgentForUser(sessionId: SessionId): Future[Option[ActorRef]] = {
    repo.get(sessionId.agentName).map(_.map {
      case AdvancedAgentSettings(_, na, ls, ts) ⇒ AgentForUser.props(AdvancedQAgent(na, ls, ts), sessionId)
    }.map { agentProps ⇒
      val agent = context.actorOf(agentProps)
      context watch agent
      agent
    })
  }

}

object Agency {
  type AgentActor = ActorRef

  def props(repo: AgentSettingsRepo = AgentSettingsRepo.apply): Props = Props(new Agency(repo))

  private case class AgentCreated(sessionId: SessionId, agentActor: AgentActor, replyTo: ActorRef)
}

