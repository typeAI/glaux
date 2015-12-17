package glaux.interfaces.akka.service

import java.time.ZonedDateTime

import akka.actor._
import akka.testkit.{ TestProbe, ImplicitSender, TestKit }
import glaux.interfaces.akka.api.Protocols.Agent._
import glaux.interfaces.akka.api.Protocols.{ Confirmed, Response }
import glaux.persistence.mongodb.SessionRepo
import glaux.reinforcementlearning.{ Reading, Reward, SimpleQAgent }
import org.specs2.mutable.Specification
import org.specs2.specification.{ AfterEach, Scope, AfterAll }

import scala.concurrent.Await
import scala.util.Random
import scala.concurrent.duration._

class AgentSpec extends Specification with AfterAll with AfterEach {
  sequential
  implicit lazy val system = ActorSystem()

  "start from new session" >> {
    "does not allow action until getting enough data" in new withAkka {
      report()
      report()
      waitForInitialization()
      agent ! RequestAction
      expectMsg(PendingMoreReadings)
    }

    "does allow action after getting enough data" in new withAkka {
      report()
      report()
      report()

      waitForInitialization()

      agent ! RequestAction
      expectMsgType[ActionResult]
    }

    "reports action available after initialized" in new withAkka {
      initialize()
      agent ! Report(randomReading, 3)
      expectMsg(ActionsAvailable)
    }

    "reports action result after initialized" in new withAkka {
      initialize()
      agent ! RequestAction
      expectMsgType[ActionResult]
    }

    "increase memory after one action feedback loop" in new withAkka {
      initialize()
      requestAction()
      agent ! QueryStatus
      val status = expectMsgType[AgentStatus]
      status.memorySize === 0

      report()
      report()
      requestAction()
      agent ! QueryStatus
      val newStatus = expectMsgType[AgentStatus]
      newStatus.memorySize === 1

    }

    "confirm after Report Termination" in new withAkka {
      initialize()
      agent ! ReportTermination(randomReading, 0)

      expectMsg(Confirmed)
    }

    "terminates after Report Termination" in new withAkka {
      initialize()
      watch(agent)
      ignoreMsg { case Confirmed ⇒ true }
      agent ! ReportTermination(randomReading, 0)
      expectTerminated(agent)
    }

    "start from exiting previous session" >> {
      "pick up previous session" in new withAkka {
        initialize()
        requestAction()
        agent ! ReportTermination(randomReading, 1)
        expectMsg(Confirmed)

        val newAgent = system.actorOf(agentProps)

        initialize(newAgent)
        newAgent ! QueryStatus
        val status = expectMsgType[AgentStatus]
        status.memorySize === 1
      }

    }
  }

  def after: Any =
    Await.result(SessionRepo.removeAll(), 5.seconds)

  def afterAll(): Unit = {
    system.terminate()
  }
}

class withAkka(implicit system: ActorSystem) extends TestKit(system) with ImplicitSender with Scope {
  import glaux.interfaces.akka.api.domain.SessionId

  def randomReading = (Seq[Double](Random.nextDouble(), Random.nextDouble(), Random.nextDouble()), ZonedDateTime.now)

  lazy val agentProps = AgentForUser.props(SimpleQAgent(8, historyLength = 3), SessionId("atest", Random.nextString(3)))

  lazy val agent = system.actorOf(agentProps)

  def waitForInitialization(a: ActorRef = agent): Unit = {
    a ! QueryStatus
    fishForMessage(3.seconds) {
      case Initializing ⇒
        a ! QueryStatus
        false
      case _ ⇒ true
    }
  }

  def report(a: ActorRef = agent, reading: Reading = randomReading, reward: Reward = Random.nextDouble()): Any = {
    a ! Report(reading, reward)
    expectReportResponse(a)
  }

  def expectReportResponse(a: ActorRef = agent): Any =
    expectMsgAnyOf(30.seconds, ActionsAvailable, Initializing, PendingMoreReadings)

  def requestAction(a: ActorRef = agent): ActionResult = {
    a ! RequestAction
    expectMsgType[ActionResult]
  }

  def initialize(a: ActorRef = agent): Unit = {
    report(a)
    report(a)
    report(a)
    waitForInitialization(a)
  }
}
