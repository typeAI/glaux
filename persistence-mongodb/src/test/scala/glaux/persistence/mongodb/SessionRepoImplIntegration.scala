package glaux.persistence.mongodb

import java.time.{ Clock, ZonedDateTime }

import glaux.interfaces.api.domain.SessionId
import glaux.linearalgebra.RowVector
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.neuralnetwork.trainers.{ SGD, VanillaSGD }
import glaux.persistence.mongodb.GeneralHandlers._
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased
import glaux.reinforcementlearning.QLearner.{ Observation, TemporalState }
import glaux.reinforcementlearning._
import org.specs2.mutable.Specification
import org.specs2.concurrent.ExecutionEnv
import org.specs2.specification.{ AfterAll, AfterEach }

import scala.concurrent.Await
import scala.concurrent.duration._

class SessionRepoImplIntegration extends Specification with AfterEach with AfterAll {
  sequential
  import GlauxHandlers._

  "SessionRepo with simpleAgent" should {

    val agent = SimpleQAgent(3, 3)
    val reading = (Seq(3d, 2d), ZonedDateTime.now)
    val result = agent.start(List(reading, reading, reading), None)
    val session = result.right.get
    implicit val h = agentSessionH(agent)
    val repo = SessionRepoImpl(agent)

    "Insert and retrieve" in { implicit ee: ExecutionEnv ⇒

      val sessionId = SessionId("a Test", "111")

      repo.insert(sessionId, session) must be_==(()).await

      repo.get(sessionId) must beSome(session).await.eventually
    }

    "upsert add new record" in { implicit ee: ExecutionEnv ⇒

      val agentId = SessionId("a Test", "112")
      repo.get(agentId) must beNone.await

      repo.upsert(agentId, session) must be_==(()).await

      repo.get(agentId) must beSome(session).await.eventually
    }

    "upsert update exisiting new record" in { implicit ee: ExecutionEnv ⇒

      val agentId = SessionId("a Test", "113")

      val (_, newSession) = agent.requestAction(session)

      repo.insert(agentId, session) must be_==(()).await
      repo.get(agentId) must beSome(session).await.eventually

      repo.upsert(agentId, newSession) must be_==(()).await
      repo.get(agentId) must beSome(newSession).await.eventually
    }

  }

  "Session Repo with Advanced Agent" should {
    implicit val agent = AdvancedQAgent(3, ConvolutionBased.Settings(), SGDSettings())
    val reading = (Seq(3d, 2d), ZonedDateTime.now)
    def mockReadings(n: Int) = (0 to n).map(i ⇒ reading.copy(_2 = reading._2.plusSeconds(i)))
    val result = agent.start(mockReadings(100), None)
    val init = result.right.get

    val (_, afterAction) = agent.requestAction(init)

    val last = (0 to 50).foldLeft(afterAction) { (session, i) ⇒
      agent.report((Seq(1d, 2d), ZonedDateTime.now.plusSeconds(i + 101)), 1d, session)
    }

    val (_, finalResult) = agent.requestAction(last)

    implicit val h = agentSessionH(agent)
    val repo = SessionRepoImpl(agent)

    "Insert and retrieve" in { implicit ee: ExecutionEnv ⇒
      val agentId = SessionId("a Test", "111")

      repo.insert(agentId, finalResult) must be_==(()).await

      repo.get(agentId) must beSome(finalResult).await.eventually
    }

  }

  def after: Any = Await.result(SessionRepoImpl.removeAll(), 30.seconds)

  def afterAll(): Unit = SessionRepoImpl.close()
}
