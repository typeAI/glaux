package glaux.reinforcement

import java.time.ZonedDateTime

import glaux.linalg.RowVector
import glaux.nn.trainers.VanillaSGD
import glaux.nn.trainers.SGD.SGDSettings
import glaux.reinforcement.DeepMindQLearner.Simplified
import glaux.reinforcement.Policy.DecisionContext
import org.specs2.matcher.Scope
import org.specs2.mutable.Specification



class QAgentSpec extends Specification {
  trait QAgentScope extends Scope {
    case class TestAgent(numOfActions: Int, fixedReturnAction: Int) extends QAgent {
      type Learner = DeepMindQLearner.Simplified
      val trainer = VanillaSGD[Simplified#Net](SGDSettings(learningRate = 0.05))
      val qLearner = DeepMindQLearner.Simplified(historyLength = 3, batchSize = 20, trainer = trainer)
      type Policy =  glaux.reinforcement.Policy[qLearner.State]

      val policy: Policy = new glaux.reinforcement.Policy[qLearner.State]{
        type Context = DecisionContext
        def init: Context = new DecisionContext {}
        def numOfActions: Action = numOfActions
        def decide(state: qLearner.State, qFunction: QFunction, context: Context): (Action, Context) = (fixedReturnAction, init)
      }

      protected def readingsToInput(readings: Seq[Reward]): RowVector =  RowVector(readings :_*)
    }

    lazy val agent = TestAgent(3, 2)
  }



  "Start" should {

    "does not start without enough history" in new QAgentScope {
      val result = agent.start(List((Seq(3d,2d), ZonedDateTime.now)), None)
      result must beLeft[String]
    }

    "start with enough history" in new QAgentScope {
      val reading = (Seq(3d, 2d), ZonedDateTime.now)
      val result = agent.start(List(reading, reading, reading), None)
      result must beRight
    }
  }

  "Request Action Without Report" should  {
    trait RequestActionScope extends QAgentScope {
      val testReading = (Seq(3d, 2d), ZonedDateTime.now)
      val result = agent.start(List(testReading, testReading, testReading), None)
      val session = result.right.get
      val (action, newSession) = agent.requestAction(session)
    }

    "return the action from policy" in new RequestActionScope {
      action === 2
    }

    "without empty recentHistory" in new RequestActionScope {
      newSession.currentReadings must beEmpty
    }

    "remember the action from policy" in new RequestActionScope {
      newSession.lastAction must beSome(action)
    }

    "not create memory for the first action request" in new RequestActionScope {
      newSession.iteration.memory must beEmpty
    }
  }

  "Request Action After Report" should {
    trait ReportScope extends QAgentScope {
      val testReading = (Seq(3d, 2d), ZonedDateTime.now)
      val start = agent.start(List(testReading, testReading, testReading), None)
      val startSession = start.right.get
      val (_, s) = agent.requestAction(startSession)
      val session = agent.report((Seq(9d, 5d), ZonedDateTime.now.plusMinutes(1)), 7, s)

    }

    "add reading and reward to the memory" in new ReportScope {
      val (_, result) = agent.requestAction(session)
      val memory = result.iteration.memory

      memory.size === 1
      memory.head.after.fullHistory.last.readings.seqView === Seq(9d, 5d)
      memory.head.reward === 7
    }

    "consecutive requests returns same action without changing session " in new ReportScope {
      val (a1, s1) = agent.requestAction(session)
      val (a2, s2) = agent.requestAction(s1)

      a1 === a2
      s1 === s2
    }
  }

  "close" should {
    trait CloseScope extends QAgentScope {
      val testReading = (Seq(3d, 2d), ZonedDateTime.now)
      val start = agent.start(List(testReading, testReading, testReading), None)
      val startSession = start.right.get
      val (_, session) = agent.requestAction(startSession)
    }

    "return a session that is closed" in new CloseScope {
      agent.close(session).isClosed must beTrue
    }

    "does not create extra memory if no further readings is provided" in new CloseScope {
      agent.close(session).iteration.memory must beEmpty
    }

    "creates extra memory if further readings is provided" in new CloseScope {

      val s = agent.report((Seq(9d, 5d), ZonedDateTime.now.plusMinutes(1)), 7, session)
      val closed = agent.close(s)

      closed.isClosed must beTrue
      val memory = closed.iteration.memory
      memory.size === 1
      memory.head.after.fullHistory.last.readings.seqView === Seq(9d, 5d)
      memory.head.reward === 7
    }
  }
}



