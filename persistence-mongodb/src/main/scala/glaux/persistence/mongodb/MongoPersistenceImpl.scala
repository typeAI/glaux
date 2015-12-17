package glaux.persistence.mongodb
import GeneralHandlers._
import GlauxHandlers._
import glaux.interfaces.api.domain.SessionId
import glaux.interfaces.api.persistence.{ AgentSettingsPersistence, SessionPersistence, PersistenceImpl }
import glaux.reinforcementlearning.{ AdvancedQAgent, QAgent, SimpleQAgent }
import reactivemongo.bson.BSONDocument

import scala.concurrent.Future

object MongoPersistenceImpl extends PersistenceImpl {

  implicit def advanceQAgentSessionPersistence: SessionPersistence[AdvancedQAgent] = new QAgentSessionPersistence[AdvancedQAgent] {
    implicit def sessionHandler(implicit agent: AdvancedQAgent): Handler[agent.Session] = agentSessionH(agent)
  }

  implicit def agentSettingsPersistence: AgentSettingsPersistence = AgentSettingsRepo()

  implicit def simpleQAgentSessionPersistence: SessionPersistence[SimpleQAgent] = new QAgentSessionPersistence[SimpleQAgent] {
    implicit def sessionHandler(implicit agent: SimpleQAgent): Handler[agent.Session] = agentSessionH(agent)
  }
}
