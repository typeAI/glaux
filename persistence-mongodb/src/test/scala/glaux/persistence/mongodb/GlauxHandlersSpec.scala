package glaux.persistence.mongodb

import java.time.{Clock, LocalTime, LocalDate, ZonedDateTime}

import glaux.interfaces.api.domain.Action
import glaux.linearalgebra.Dimension._
import glaux.linearalgebra._
import glaux.neuralnetwork.Net.DefaultNet
import glaux.neuralnetwork.trainers.BatchTrainer.{BatchResult, LossInfo}
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.neuralnetwork.{Rectangle, LossLayer, HiddenLayer, InputLayer}
import glaux.neuralnetwork.layers._
import glaux.reinforcementlearning.DeepMindQLearner.ConvolutionBased
import glaux.reinforcementlearning.QLearner.{Observation, State, TemporalState}
import glaux.reinforcementlearning._
import org.specs2.mutable.Specification
import GlauxHandlers._
import GeneralHandlers._
import reactivemongo.bson._

case class Test(readings: Vector[Reading], lastAction: Option[Action])
case class TestHiddenLayer(layers: Seq[HiddenLayer])
case class TestLossLayer(layers: Seq[LossLayer])

class GlauxHandlersSpec extends Specification {

  def canHandle[T: Handler](t: T) =
    BSON.write(t).as[T] === t

  "Formats" should {

    "format specific dimensions" in {
      canHandle(Row(5))
      canHandle(TwoD(5, 3))
      canHandle(ThreeD(5, 2, 1))
    }

    "format general dimension" in {
      val d: Dimension = TwoD(5, 3)
      canHandle(d)
      val d2: Dimension = ThreeD(5, 2, 1)
      canHandle(d2)
    }

    "format tensors" in {
      canHandle(RowVector(2, 3, 4))
      canHandle(Matrix(2, 2, Seq(4d, 3d, 1d, 2d)))
      canHandle(Vol(2, 2, 2, Seq(4d, 3d, 1d, 2d, 4d, 3d, 1d, 2d)))
    }

    "format InputLayer" in {
      canHandle(InputLayer[RowVector](Row(3)))
    }

    "format FullyConnected" in {
      canHandle(FullyConnected(3, 4))
    }

    "format FullyConnected of Vol" in {
      canHandle(FullyConnected[Vol](ThreeD(3, 3, 3), 4))
    }

    "format Convolution" in {
      val inputDim = ThreeD(4, 4, 2)
      val layer = Convolution(3, Rectangle(3, 3), inputDim, 1)
      canHandle(layer)
    }

    "format Pool" in {
      canHandle(Pool(ThreeD(4, 4, 1), Rectangle(3, 3), 3))
    }

    "format Relu" in {
      canHandle(Relu[RowVector](Row(4)))
      canHandle(Relu[Matrix](TwoD(2, 3)))
      canHandle(Relu[Vol](ThreeD(4, 2, 3)))
    }

    "format HiddenLayer" in {
      val f: HiddenLayer = FullyConnected(3, 4)
      val fv: HiddenLayer = FullyConnected[Vol](ThreeD(3, 3, 3), 4)
      val r: Relu[RowVector] = Relu[RowVector](Row(4))
      val p: Pool = Pool(ThreeD(4, 4, 1), Rectangle(3, 3), 3)
      val c = Convolution(3, Rectangle(3, 3), ThreeD(3, 3, 3), 1)
      val r3: Relu[Vol] = Relu[Vol](ThreeD(3, 2, 3))
      val r2: Relu[Matrix] = Relu[Matrix](TwoD(2, 3))
      implicit val thf = Macros.handler[TestHiddenLayer]
      canHandle(TestHiddenLayer(Seq(f, fv, r, p, c, r3, r2)))
    }

    "format lossLayer" in {
      val r: LossLayer = Regression(Row(3))
      val s: LossLayer = Softmax(Row(3))
      implicit val thf = Macros.handler[TestLossLayer]
      canHandle(TestLossLayer(Seq(r, s)))
    }

    "format default net" in {
      val inputSize = 5
      val netInputDimension = Row(inputSize)
      val inputLayer = InputLayer[RowVector](netInputDimension)
      val fc1 = FullyConnected(inputSize, inputSize)
      val relu = Relu[RowVector](netInputDimension)
      val fc2 = FullyConnected(inputSize, 5)
      val lossLayer = Regression(5)
      val net = DefaultNet(inputLayer, Seq(fc1, relu, fc2), lossLayer)
      canHandle(net)
    }

    "format BatchResult" in {
      val loss = LossInfo(1, 3, 4)

      val inputSize = 5
      val inputLayer = InputLayer[RowVector](Row(inputSize))
      val fc1 = FullyConnected(inputSize, inputSize)
      val lossLayer = Regression(inputSize)
      val net = DefaultNet(inputLayer, Seq(fc1), lossLayer)
      canHandle(BatchResult(loss, net, 1, ()))
    }

    "format temporal state" in {
      val t = new TemporalState(Matrix(2, 2, Seq(1d, 1d, 1d, 3d)), ZonedDateTime.now)
      canHandle(t)
    }

    "format State" in {
      val t = new TemporalState(Matrix(2, 2, Seq(1d, 1d, 1d, 3d)), ZonedDateTime.now)
      val t2 = new TemporalState(Matrix(2, 2, Seq(1d, 2d, 4d, 3d)), ZonedDateTime.now.plusDays(1))
      canHandle(State(Seq(t, t2), true))
    }

    "format learner iteration" in {
      val start = ZonedDateTime.of(LocalDate.of(2015, 2, 14), LocalTime.of(14, 30), Clock.systemDefaultZone().getZone)
      val learner = DeepMindQLearner.Simplified(historyLength = 2)
      val initHistory = Seq(TemporalState(RowVector(1, 0), start), TemporalState(RowVector(2, 1), start.plusMinutes(1)))
      val init = learner.init(initHistory, 2)
      val iteration = learner.iterate(init, Observation(
        lastAction = 1,
        reward = 1,
        recentHistory = Seq(TemporalState(RowVector(3, 2), start.plusMinutes(2))),
        isTerminal = true
      ))
      implicit val iterH = learnerIterationH(learner)
      canHandle(iteration)
    }

    "format simple qAgent Session" in {
      val agent = SimpleQAgent(3, 3)
      val reading = (Seq(3d, 2d), ZonedDateTime.now)
      val result = agent.start(List(reading, reading, reading), None)
      val session = result.right.get

      implicit val sessionF = agentSessionH(agent)
      canHandle(session)
    }

    "format advanced qAgent Session" in {
      val agent = AdvancedQAgent(3, ConvolutionBased.Settings(), SGDSettings())
      val reading = (Seq(3d, 2d), ZonedDateTime.now)
      val result = agent.start((0 to 100).map(_ ⇒ reading), None)
      val session = result.right.get

      implicit val sessionF = agentSessionH(agent)
      canHandle(session)
    }

  }

}
