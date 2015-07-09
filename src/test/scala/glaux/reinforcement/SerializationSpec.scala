package glaux.reinforcement

import java.io.{ByteArrayInputStream, ObjectInputStream, ByteArrayOutputStream}
import java.time.ZonedDateTime

import glaux.linalg.{Vol, Matrix, RowVector}
import glaux.nn.trainers.{SGDOptions, VanillaSGD}
import glaux.reinforcement.DeepMindQLearner.Simplified
import glaux.test.utils.DebuggingObjectOutputStream
import scala.collection.JavaConverters._
import org.specs2.mutable.Specification

class SerializationSpec extends Specification {

  def canSerialize(o: Any) = {
    val bos = new ByteArrayOutputStream()
    val oos = new DebuggingObjectOutputStream(bos)
    try {
      oos.writeObject(o)
    } catch {
      case e: Exception =>
        throw new RuntimeException(
          "Serialization error. Path to bad object: "
            + oos.getStack.asScala.mkString(" -> "), e);
    }
    oos.close

    val bytes = bos.toByteArray
    bytes.length must be_>(0)
    val deSerialized = new ObjectInputStream(new ByteArrayInputStream(bytes)).readObject()
    deSerialized === o
  }

  "linalg classes" should {
    "be serializable" in {
      canSerialize(RowVector(1d, 2d))
      canSerialize(Matrix(2,2, Seq(4d, 3d, 1d, 2d)))
      canSerialize(Vol(2,2,2, Seq(4d, 3d, 1d, 2d, 4d, 3d, 1d, 2d)))
    }
  }

  val trainer = VanillaSGD[Simplified#Net](SGDOptions(learningRate = 0.05))
  val learner = DeepMindQLearner.Simplified(historyLength = 1, batchSize = 20, trainer = trainer)
  import learner.TemporalState
  val start = ZonedDateTime.now
  val initHistory = Seq(TemporalState(RowVector(1, 0), start))

  "dqn classes" should {
    "be serializable" in {
      val iteration = learner.init(initHistory, 2)
      canSerialize(iteration)
    }
  }

  case class TestAgent(numOfActions: Int) extends QAgent {
    val policy: Policy = (_, _) => 0
    val qLearner = learner
    protected def readingsToInput(readings: Seq[Reward]): RowVector =  RowVector(readings :_*)
  }

  val agent = TestAgent(2)

  val testReading = (Seq(3d, 2d), ZonedDateTime.now)
  val session = agent.start(List(testReading), None).right.get

//
//  "QAgent#Session" should {
//    "be serializable" in {
//       canSerialize(session)
//    }
//  }
}
