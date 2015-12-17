package glaux.interfaces.akka.service

import akka.actor._
import akka.testkit.{ ImplicitSender, TestKit }
import glaux.interfaces.akka.api.Protocols.Agency.{ CreateAgentSettings, AgentRef, GetAgentForUser }
import glaux.interfaces.akka.api.Protocols.{ Rejected, Confirmed }
import glaux.persistence.mongodb.{ AgentSettingsRepo, SessionRepo }
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased
import org.specs2.mutable.Specification
import org.specs2.specification.{ AfterEach, Scope, AfterAll }
import glaux.interfaces.akka.api.domain.{ AdvancedAgentSettings, AgentName, AgentSettings, SessionId }

import scala.concurrent.{ Future, Await }
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

      agency ! GetAgentForUser(agentId)
      expectMsgType[AgentRef].ref !== agent
    }

  }

  "CreateAgentSettings" >> {
    def mockRepoWith(upsertResult: Future[Unit]) = new AgentSettingsRepo {
      def get(name: AgentName): Future[Option[AgentSettings]] = ???
      def upsert(settings: AgentSettings): Future[Unit] = upsertResult
    }

    "creates one with confirmation" in new AgencyScope {
      val agency1 = system.actorOf(Agency.props(mockRepoWith(Future.successful(()))))
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

  val mockRepo = new AgentSettingsRepo {
    def get(name: AgentName): Future[Option[AgentSettings]] = Future.successful(Some(AdvancedAgentSettings(name, 8, ConvolutionBased.Settings(), SGDSettings())))
    def upsert(settings: AgentSettings): Future[Unit] = ???
  }
  lazy val agency = system.actorOf(Agency.props(mockRepo))

}
