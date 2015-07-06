package glaux.reinforcement

import java.time.ZonedDateTime

import glaux.linalg.RowVector
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import glaux.reinforcement.DeepMindQLearner.Simplified
import org.specs2.matcher.Scope
import org.specs2.mutable.Specification



class QAgentSpec extends Specification {
  trait QAgentScope extends Scope {
    case class TestAgent(numOfActions: Int, fixedReturnAction: Int) extends QAgent {
      val trainer = VanillaSGD[Simplified#Net](SGDOptions(learningRate = 0.05))
      val qLearner = DeepMindQLearner.Simplified(historyLength = 3, batchSize = 20, trainer = trainer)
      val policy: Policy = (_, _) => fixedReturnAction

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

  "Request Action" should  {
    trait RequestActionScope extends QAgentScope {
      val reading = (Seq(3d, 2d), ZonedDateTime.now)
      val result = agent.start(List(reading, reading, reading), None)
      val session = result.right.get
      val (action, newSession) = agent.requestAction(session)
    }

    "return the action from policy" in new RequestActionScope {
      action === 2

    }
    "remember the action from policy" in new RequestActionScope {
      newSession.lastAction must beSome(action)
    }

    "not create memory for the first action request" in new RequestActionScope {
      newSession.iteration.memory must beEmpty
    }
  }
}



