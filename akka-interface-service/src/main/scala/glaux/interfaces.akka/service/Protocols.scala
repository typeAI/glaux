package glaux.interfaces.akka.service

import akka.actor.ActorRef
import glaux.interfaces.api.domain._

object Protocols {
  
  sealed trait Response
  sealed trait Request[T <: Response]
  sealed trait Request2[T1 <: Response, T2 <: Response]
  sealed trait Request3[T1 <: Response, T2 <: Response, T3 <: Response]

  case object Confirmed extends Response
  case class Rejected(reason: String) extends Response

  trait RequestWithConfirmation extends Request2[Confirmed.type, Rejected]

  object Agency {

    case class GetAgentForUser(sessionId: SessionId) extends Request2[AgentRef, Rejected]

    case class AgentRef(ref: ActorRef) extends Response

    case class CreateAgentSettings(settings: AgentSettings) extends RequestWithConfirmation

  }
  
  object Agent {

    case class Report(reading: Reading, reward: Reward) extends Request3[ActionsAvailable.type, PendingMoreReadings.type, Initializing.type]

    case class ReportTermination(reading: Reading, reward: Reward) extends RequestWithConfirmation

    case object RequestAction extends Request3[ActionResult, PendingMoreReadings.type, Initializing.type]
    
    case class ActionResult(action: Action) extends Response

    case object QueryStatus extends Request2[AgentStatus, Initializing.type]

    case class AgentStatus(memorySize: Int, currentLoss: Double) extends Response

    case object PendingMoreReadings extends Response
    case object Initializing extends Response
    case object ActionsAvailable extends Response
  }

}
