package glaux.persistence.mongodb

import glaux.interfaces.api.domain.{AdvancedAgentSettings, AgentSettings, SessionId}
import glaux.persistence.mongodb.GeneralHandlers._
import glaux.reinforcementlearning.QAgent
import reactivemongo.bson.Macros
import reactivemongo.bson.Macros.Options.AllImplementations
import GlauxHandlers._
import play.api.libs.functional._
import syntax._

object InterfaceHandlers {
  implicit val sessionIdH = Macros.handler[SessionId]
  implicit val aashandler = Macros.handler[AdvancedAgentSettings]
  implicit val agshandler = Macros.handlerOpts[AgentSettings, AllImplementations]

  implicit def sessionRecordFormat[Session <: QAgent.Session[_, _]: Handler]: Handler[SessionRecord[Session]] = {
    (field[SessionId]("id") ~
      field[Session]("session"))(SessionRecord.apply[Session], unlift(SessionRecord.unapply[Session]))
  }
}
