package glaux.interfaces.akka.service

import glaux.interfaces.api.domain.{SessionId, AgentName, AgentSettings}
import glaux.interfaces.api.persistence.{Persistence, SessionPersistence, PersistenceImpl}
import glaux.reinforcementlearning.{QAgent, AdvancedQAgent, SimpleQAgent}

import scala.concurrent.Future

trait MockPersistence extends PersistenceImpl {
  case class MapBasedMockPersistence[T, K](getKey: T ⇒ K) extends Persistence[T, K] {
    @volatile
    var store: Map[K, T] = Map()

    def get(id: K): Future[Option[T]] = Future.successful(store.get(id))

    def upsert(t: T): Future[Unit] = {
      store += getKey(t) → t
      Future.successful(())
    }
  }

  class MockSessionPersistence[A <: QAgent] extends SessionPersistence[A] {
    @volatile
    var store: Map[SessionId, QAgent.Session[_, _]] = Map()

    def get(agent: A, id: SessionId): Future[Option[agent.Session]] =
      Future.successful(store.get(id).map(_.asInstanceOf[agent.Session]))

    def upsert(agent: A, id: SessionId)(session: agent.Session): Future[Unit] = {
      store += id → session
      Future.successful(())
    }
  }

  implicit def advanceQAgentSessionPersistence: SessionPersistence[AdvancedQAgent] = new MockSessionPersistence[AdvancedQAgent]

  implicit def agentSettingsPersistence: Persistence[AgentSettings, AgentName] = new MapBasedMockPersistence[AgentSettings, AgentName](_.name)

  implicit def simpleQAgentSessionPersistence: SessionPersistence[SimpleQAgent] = new MockSessionPersistence[SimpleQAgent]
}

object MockPersistence extends MockPersistence
