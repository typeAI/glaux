package glaux.interfaces.akka.service

import akka.actor._
import akka.testkit.{ImplicitSender, TestKit}
import Protocols.Agency.{CreateAgentSettings, AgentRef, GetAgentForUser}
import Protocols.{Rejected, Confirmed}
import glaux.interfaces.api.persistence.{SessionPersistence, PersistenceImpl, AgentSettingsPersistence, Persistence}
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased
import glaux.reinforcementlearning.{AdvancedQAgent, SimpleQAgent}
import org.specs2.mutable.Specification
import org.specs2.specification.{AfterEach, Scope, AfterAll}
import glaux.interfaces.api.domain.{AdvancedAgentSettings, AgentName, AgentSettings, SessionId}

import scala.concurrent.{Future, Await}
import scala.concurrent.duration._

class AgencySpec extends Specification with AfterAll {
  sequential
  implicit lazy val system = ActorSystem()
  "GetAgent" >> {
    "starts new agent " in new AgencyScope {
      agency ! GetAgentForUser(SessionId("A Test", "testUser"))
      expectMsgType[AgentRef]
    }

    "returns existing agent if found" in new AgencyScope {
      val agentId = SessionId("A Test", "testUser")

      agency ! GetAgentForUser(agentId)
      val agent = expectMsgType[AgentRef].ref
      agency ! GetAgentForUser(agentId)
      expectMsgType[AgentRef].ref === agent
    }

    "returns existing agent if hit multiple times" in new AgencyScope {
      val agentId = SessionId("A Test", "testUser")

      agency ! GetAgentForUser(agentId)
      agency ! GetAgentForUser(agentId)
      val agent = expectMsgType[AgentRef].ref
      expectMsgType[AgentRef].ref === agent
    }

    "create new agent id is new" in new AgencyScope {
      val agentId = SessionId("A Test", "testUser")

      agency ! GetAgentForUser(agentId)
      val agent = expectMsgType[AgentRef].ref

      agency ! GetAgentForUser(SessionId("A Test", "anotherUser"))
      expectMsgType[AgentRef].ref !== agent
    }

    "recreate new agent after last agent is terminated " in new AgencyScope {
      val agentId = SessionId("A Test", "testUser")

      agency ! GetAgentForUser(agentId)
      val agent = expectMsgType[AgentRef].ref
      watch(agent)
      agent ! PoisonPill
      expectTerminated(agent)

      {
        agency ! GetAgentForUser(agentId)
        expectMsgType[AgentRef].ref
      } must be_!=(agent).eventually
    }

  }

  "CreateAgentSettings" >> {

    "creates one with confirmation" in new AgencyScope {
      val agency1 = system.actorOf(Agency.props(mockRepoWith()))
      agency1 ! CreateAgentSettings(AdvancedAgentSettings("test", 5, ConvolutionBased.Settings(), SGDSettings()))
      expectMsg(Confirmed)

    }

    "fails with rejection" in new AgencyScope {
      val agency1 = system.actorOf(Agency.props(mockRepoWith(Future.failed(new Exception()))))
      agency1 ! CreateAgentSettings(AdvancedAgentSettings("test", 5, ConvolutionBased.Settings(), SGDSettings()))
      expectMsg(Rejected)
    }
  }

  def afterAll(): Unit = {
    system.terminate()
  }
}

class AgencyScope(implicit system: ActorSystem) extends TestKit(system) with ImplicitSender with Scope {
  def mockRepoWith(
    upsertResult:     Future[Unit]          = Future.successful(()),
    getSettingResult: Option[AgentSettings] = None
  ) = new MockPersistence {
    override implicit def agentSettingsPersistence: AgentSettingsPersistence = new AgentSettingsPersistence {
      def get(name: AgentName): Future[Option[AgentSettings]] = Future.successful(getSettingResult)
      def upsert(settings: AgentSettings): Future[Unit] = upsertResult
    }
  }

  val testAgentSettings = Some(AdvancedAgentSettings("test", 8, ConvolutionBased.Settings(), SGDSettings()))

  lazy val agency = system.actorOf(Agency.props(mockRepoWith(getSettingResult = testAgentSettings)))

}
