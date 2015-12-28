package glaux.reinforcementlearning

import glaux.linearalgebra.Dimension.{ThreeD, Row}
import glaux.linearalgebra.{Vol, RowVector}
import glaux.neuralnetwork.trainers.VanillaSGD
import glaux.neuralnetwork.trainers.SGD.SGDSettings
import glaux.neuralnetwork.{HiddenLayer, Rectangle, InputLayer, Net}
import Net.DefaultNet
import glaux.neuralnetwork.layers._
import glaux.reinforcementlearning.QLearner.Transition

import scala.util.Random

/**
 * QLearner based on deepmind algorithm
 */
trait DeepMindQLearner extends QLearner {

  def gamma: Double
  def batchSize: Int
  def targetNetUpdateFreq: Int //avg # of iterations before updating the target net
  def minMemorySizeBeforeTraining: Int

  type Net = DefaultNet[NetInput]
  type Trainer = VanillaSGD[Net]

  case class DeepMindIteration(
    targetNet:         Net,
    memory:            Memory, //Seq because we need random access here
    trainingResult:    TrainingResult,
    state:             State,
    isTerminal:        Boolean        = false,
    targetNetHitCount: Int            = 0
  ) extends IterationLike {
    lazy val net = trainingResult.net

  }

  assert(historyLength > 0)

  type Iteration = DeepMindIteration

  protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net

  override protected def doInit(initState: State, numOfActions: Action, inputDim: InputDimension): Iteration = {
    val initNet = buildNet(inputDim, numOfActions)

    DeepMindIteration(initNet, Nil, trainer.init(initNet), initState)
  }

  override protected def doIterate(lastIteration: Iteration, observation: Observation, currentState: State): Iteration = {
    val newMemory = if (lastIteration.isTerminal) lastIteration.memory
    else {
      val before = lastIteration.state
      lastIteration.memory :+ Transition(before, observation.lastAction, observation.reward, currentState)
    }

    val doTraining: Boolean = newMemory.size > minMemorySizeBeforeTraining
    val updateTarget: Boolean = doTraining && lastIteration.targetNetHitCount > targetNetUpdateFreq

    val targetNet = lastIteration.targetNet
    lazy val newResult = train(newMemory, lastIteration.trainingResult, targetNet)

    lastIteration.copy(
      targetNet = if (updateTarget) newResult.net else targetNet,
      memory = newMemory,
      trainingResult = if (doTraining) newResult else lastIteration.trainingResult,
      state = currentState,
      isTerminal = observation.isTerminal,
      targetNetHitCount = if (updateTarget) 0 else lastIteration.targetNetHitCount + 1
    )
  }

  protected def validate(): Unit = {
    assert(minMemorySizeBeforeTraining > batchSize, "must have enough transitions in memory before training")
  }

  private def train(memory: Memory, lastResult: TrainingResult, targetNet: Net): TrainingResult = {
    def randomExamples: Memory = {
      (1 to batchSize).map { _ ⇒
        memory(Random.nextInt(memory.size))
      }
    }

    def toTrainingInput(transition: Transition[Input]): (NetInput, Trainer#ScalarOutputInfo) = {
      val regressionOnAction = if (transition.after.isTerminal) transition.reward else
        transition.reward + targetNet.predict(transition.after).seqView.max * gamma

      (transition.before, (regressionOnAction, transition.action))
    }
    trainer.trainBatchWithScalaOutputInfo(lastResult)(randomExamples.map(toTrainingInput))
  }

}

object DeepMindQLearner {
  case class Simplified(
    override protected val trainer: Simplified#Trainer = VanillaSGD[Simplified#Net](SGDSettings()),
    historyLength:                  Int                = 50,
    gamma:                          Double             = 0.95,
    batchSize:                      Int                = 40,
    targetNetUpdateFreq:            Int                = 10, //avg # of iterations before updating the target net
    minMemorySizeBeforeTraining:    Int                = 100
  ) extends DeepMindQLearner {
    type NetInput = RowVector
    type Input = RowVector

    validate()

    implicit def inputToNet(state: State): NetOutput = {
      RowVector(state.fullHistory.flatMap(_.readings.seqView): _*)
    }

    protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net = {
      val inputSize = inputDimension.size * historyLength
      val netInputDimension = Row(inputSize)
      val inputLayer = InputLayer[RowVector](netInputDimension)
      val fc1 = FullyConnected(inputSize, inputSize)
      val relu = Relu[RowVector](netInputDimension)
      val fc2 = FullyConnected(inputSize, numOfActions)
      val lossLayer = Regression(numOfActions)
      DefaultNet(inputLayer, Seq(fc1, relu, fc2), lossLayer)
    }
  }

  case class ConvolutionBased(
    override protected val trainer: ConvolutionBased#Trainer  = VanillaSGD[ConvolutionBased#Net](SGDSettings()),
    settings:                       ConvolutionBased.Settings
  ) extends DeepMindQLearner {
    type NetInput = Vol
    type Input = RowVector

    def gamma = settings.gamma
    def historyLength = settings.historyLength
    def batchSize = settings.batchSize
    def targetNetUpdateFreq = settings.targetNetUpdateFreq
    def minMemorySizeBeforeTraining = settings.minMemorySizeBeforeTraining

    validate()
    assert(historyLength > settings.filterSize * 2, "too short history makes convolution useless")

    implicit def inputToNet(state: State): NetInput = {
      Vol(netInputDimension(state.fullHistory.head.readings.dimension), state.fullHistory.flatMap(_.readings.seqView))
    }

    protected def buildNet(inputDimension: Input#Dimensionality, numOfActions: Int): Net = {
      val netInputDim = netInputDimension(inputDimension)
      val inputLayer = InputLayer[Vol](netInputDim)
      type ConvLayerCombo = (Convolution, Relu[Vol])
      type FullyConnectedLayerCombo = (FullyConnected[_], Relu[RowVector])

      def flatten(seq: Seq[(HiddenLayer, HiddenLayer)]): Seq[HiddenLayer] = seq.flatMap(p ⇒ Seq(p._1, p._2))

      val convolutions = (1 to settings.numOfConvolutions).foldLeft(Vector.empty[ConvLayerCombo]) { (convs, _) ⇒
        val iDim: ThreeD = convs.lastOption.map(_._2.outDimension).getOrElse(netInputDim)
        val conv = Convolution(
          numOfFilters = settings.numOfFilters,
          filterSize = Rectangle(settings.filterSize, 1),
          inputDimension = iDim,
          padding = true
        )
        val relu = Relu[Vol](conv.outDimension)
        convs :+ ((conv, relu))
      }

      val midFc = FullyConnected[Vol](convolutions.last._2.outDimension, settings.numOfFullyConnectedNeurons)
      val midRelu = Relu[RowVector](midFc.outDimension)

      val fullyConnecteds = (1 until settings.numOfFullyConnected).foldLeft(Vector[FullyConnectedLayerCombo]((midFc, midRelu))) { (fcs, _) ⇒
        val fc = FullyConnected(fcs.last._2.outDimension.size, settings.numOfFullyConnectedNeurons)
        val relu = Relu[RowVector](fc.outDimension)
        fcs :+ ((fc, relu))
      }

      val lastFc = FullyConnected(fullyConnecteds.last._2.outDimension.totalSize, numOfActions)
      val lastRelu = Relu[RowVector](lastFc.outDimension)

      val lossLayer = Regression(numOfActions)
      DefaultNet(inputLayer, flatten(convolutions) ++ flatten(fullyConnecteds :+ ((lastFc, lastRelu))), lossLayer)
    }

    private def netInputDimension(inputDimension: Row): ThreeD = ThreeD(historyLength, 1, inputDimension.size)
  }

  object ConvolutionBased {
    case class Settings(
      historyLength:               Int    = 50,
      filterSize:                  Int    = 5,
      gamma:                       Double = 0.95,
      batchSize:                   Int    = 20,
      numOfFilters:                Int    = 10,
      numOfFullyConnectedNeurons:  Int    = 30,
      numOfConvolutions:           Int    = 2,
      numOfFullyConnected:         Int    = 2,
      targetNetUpdateFreq:         Int    = 10, //avg # of iterations before updating the target net
      minMemorySizeBeforeTraining: Int    = 100
    )
  }
}
