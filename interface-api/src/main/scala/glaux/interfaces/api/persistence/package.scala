package glaux.interfaces.api

import glaux.interfaces.api.domain.{SessionId, AgentSettings, AgentName}
import glaux.reinforcementlearning.{SimpleQAgent, AdvancedQAgent, QAgent}

import scala.concurrent.Future

package object persistence {

  type AgentSettingsPersistence = Persistence[AgentSettings, AgentName]

}

package persistence {

  trait Persistence[T, Key] {
    def get(k: Key): Future[Option[T]]
    def upsert(t: T): Future[Unit]
  }
  trait SessionPersistence[A <: QAgent] {
    def get(agent: A, id: SessionId): Future[Option[agent.Session]]
    def upsert(agent: A, id: SessionId)(session: agent.Session): Future[Unit]
  }

  trait PersistenceImpl {
    implicit def advanceQAgentSessionPersistence: SessionPersistence[AdvancedQAgent]
    implicit def simpleQAgentSessionPersistence: SessionPersistence[SimpleQAgent]
    implicit def agentSettingsPersistence: AgentSettingsPersistence
  }

}
